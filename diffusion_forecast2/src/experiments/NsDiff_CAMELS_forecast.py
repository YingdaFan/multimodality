
from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os
import torch
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.NsDiff0 import NsDiff
import src.layer.mu_backbone as ns_Transformer
import argparse
import src.layer.g_backbone as G
from src.experiments.prob_forecast import ProbForecastExp, update_metrics
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from torch.optim import *
from tqdm import tqdm
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
import torch.multiprocessing as mp
from torch_timeseries.utils.parse_type import parse_type

from torch_timeseries.utils.early_stop import EarlyStopping
from src.layer.nsdiff_utils import q_sample, p_sample_loop, cal_sigma12, cal_sigma_tilde, cal_forward_noise
import yaml
import numpy as np
import torch.distributed as dist
import torch
# autocast/GradScaler removed: pure float32 training to avoid NaN overflow
from tqdm import tqdm
import concurrent.futures
from types import SimpleNamespace
from src.utils.sigma import wv_sigma, wv_sigma_trailing

# Import forecast-specific dataset and loader (single-layer sliding window)
from src.datasets.forecast_dataset import ForecastMeta
from src.dataloader.forecast_loader import ForecastLoader


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


EPS = 1e-8


class NSDiffEarlyStopping(EarlyStopping):
    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model['model'].state_dict(), os.path.join(self.path, 'model.pth'))
        torch.save(model['cond_pred_model'].state_dict(), os.path.join(self.path, 'cond_pred_model.pth'))
        torch.save(model['cond_pred_model_g'].state_dict(), os.path.join(self.path, 'cond_pred_model_g.pth'))
        self.val_loss_min = val_loss


def log_normal(x, mu, var):
    """Logarithm of normal distribution with mean=mu and variance=var
    log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

    Args:
       x: (array) corresponding array containing the input
       mu: (array) corresponding array containing the mean
       var: (array) corresponding array containing the variance

    Returns:
       output: (array/float) depending on average parameters the result will be the mean
                            of all the sample losses or an array with the losses per sample
    """
    eps = 1e-8
    if eps > 0.0:
        var = var + eps
    return 0.5 * torch.mean(
        np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)



@dataclass
class NsDiffParameters:
    num_samples: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.01
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 1024
    diffusion_steps: int = 20
    moving_avg: int = 25
    factor: int = 3
    distil: bool = True
    dropout: float = 0.05
    activation: str = 'gelu'
    k_z: float = 1e-2
    k_cond: int = 1
    d_z: int = 8
    CART_input_x_embed_dim: int = 32
    p_hidden_layers: int = 2
    rolling_length: int = 96
    load_pretrain: bool = False
    npz_path: str = '../data_processing/data/prepped.npz'


@dataclass
class NsDiffCAMELSForecast(ProbForecastExp, NsDiffParameters):
    model_type: str = "NsDiff_CAMELS_forecast"
    dataset_type: str = "CAMELS"
    stride: int = 1

    def _init_dataset(self):
        """Initialize CAMELS dataset"""
        self.dataset = ForecastMeta(npz_path=self.npz_path)

    def _init_data_loader(self, shuffle=True, fast_test=True, fast_val=True):
        """
        Use CAMELSLoader for forecasting (shuffle=True for training)
        """
        self._init_dataset()

        from torch_timeseries.scaler import StandardScaler
        self.scaler = StandardScaler()

        # Forecasting: no masking
        self.dataloader = ForecastLoader(
            dataset=self.dataset,
            scaler=self.scaler,
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len,
            shuffle_train=shuffle,
            freq=self.dataset.freq,
            batch_size=None,  # Automatically uses n_segs
            num_worker=self.num_worker,
            fast_test=fast_test,
            fast_val=fast_val,
            npz_path=self.npz_path,
            masked_basins=None,  # Forecasting: no masking
            stride=self.stride,
        )

        self.train_loader, self.val_loader, self.test_loader = (
            self.dataloader.train_loader,
            self.dataloader.val_loader,
            self.dataloader.test_loader,
        )

    def _init_model(self):
        self.label_len = self.windows // 2
        args_dict = {
            "seq_len": self.windows,
            "device": self.device,
            "pred_len": self.pred_len,
            "label_len": self.label_len,
            "features": 'MS',
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "enc_in": self.dataset.num_features,  # X + Y(1), dynamically read from npz
            "dec_in": self.dataset.num_features,
            "c_out": 1,  # MS mode: only predict Y (1-dim)
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "e_layers": self.e_layers,
            "d_layers": self.d_layers,
            "d_ff": self.d_ff,
            "moving_avg": self.moving_avg,
            "timesteps": self.diffusion_steps,
            "factor": self.factor,
            "distil": self.distil,
            "beta_schedule": "linear",
            "embed": 'timeF',
            "dropout": self.dropout,
            "activation": self.activation,
            "output_attention": False,
            "do_predict": True,
            "k_z": self.k_z,
            "k_cond": self.k_cond,
            "p_hidden_dims": [64, 64],
            "freq": self.dataset.freq,
            "CART_input_x_embed_dim": self.CART_input_x_embed_dim,
            "p_hidden_layers": self.p_hidden_layers,
            "d_z": self.d_z,
            "diffusion_config_dir": "./configs/nsdiff.yml",
            "time_embed": False,  # CAMELS time features already encoded in X
        }

        with open("./configs/nsdiff.yml", "r") as f:
            config = yaml.unsafe_load(f)
            self.diffusion_config = dict2namespace(config)

        self.args = SimpleNamespace(**args_dict)
        self.model = NsDiff(self.args, self.device).to(self.device)
        self.cond_pred_model = ns_Transformer.Model(self.args).float().to(self.device)
        # g(x): 从 Y 历史预测方差，enc_in=1（只看流量维度）
        self.cond_pred_model_g = G.SigmaEstimation(
            self.windows, self.pred_len, 1, 512, self.rolling_length
        ).float().to(self.device)

        if self.load_pretrain:
            model_f_path = f"./results/runs/F/{self.dataset_type}/w{self.windows}h1s{self.pred_len}/1/best_model.pth"
            model_g_path = f"./results/runs/G/{self.dataset_type}/w{self.windows}h1s{self.pred_len}/1/best_model.pth"
            print("using pretrained model...")
            print(f"f(x): {model_f_path}")
            print(f"g(x): {model_g_path}")
            if os.path.exists(model_f_path):
                self.cond_pred_model.load_state_dict(torch.load(model_f_path, map_location=self.device, weights_only=True))
            if os.path.exists(model_g_path):
                self.cond_pred_model_g.load_state_dict(torch.load(model_g_path, map_location=self.device, weights_only=True))

    def _init_optimizer(self):
        # 3 models jointly optimized
        self.model_optim = parse_type(self.optm_type, globals=globals())(
            [{'params': self.model.parameters()},
             {'params': self.cond_pred_model.parameters()},
             {'params': self.cond_pred_model_g.parameters()},
             ],
            lr=self.lr,
        )
        # GradScaler removed: pure float32 training

    def _setup_early_stopper(self):
        self.best_checkpoint_filepath = os.path.join(
            self.run_save_dir, "model.pth"
        )
        self.best_cond_checkpoint_filepath = os.path.join(
            self.run_save_dir, "cond_pred_model.pth"
        )
        self.best_cond_g_checkpoint_filepath = os.path.join(
            self.run_save_dir, "cond_pred_model_g.pth"
        )
        self.early_stopper = NSDiffEarlyStopping(
            self.patience, verbose=True, path=self.run_save_dir
        )

    def _save_run_check_point(self, seed):
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
        print(f"Saving run checkpoint to '{self.run_save_dir}'.")

        self.run_state = {
            "model": self.model.state_dict(),
            "cond_pred_model": self.cond_pred_model.state_dict(),
            "cond_pred_model_g": self.cond_pred_model_g.state_dict(),
            "current_epoch": self.current_epoch,
            "optimizer": self.model_optim.state_dict(),
            "rng_state": torch.get_rng_state(),
            "early_stopping": self.early_stopper.get_state(),
        }

        torch.save(self.run_state, f"{self.run_checkpoint_filepath}")
        print("Run state saved ... ")

    def _load_best_model(self):
        self.model.load_state_dict(
            torch.load(self.best_checkpoint_filepath, map_location=self.device)
        )
        self.cond_pred_model.load_state_dict(
            torch.load(self.best_cond_checkpoint_filepath, map_location=self.device)
        )
        self.cond_pred_model_g.load_state_dict(
            torch.load(self.best_cond_g_checkpoint_filepath, map_location=self.device)
        )

    def _resume_run(self, seed):
        check_point = torch.load(self.run_checkpoint_filepath, map_location=self.device)

        self.model.load_state_dict(check_point["model"])
        self.cond_pred_model.load_state_dict(check_point["cond_pred_model"])
        self.cond_pred_model_g.load_state_dict(check_point["cond_pred_model_g"])
        self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]

        self.early_stopper.set_state(check_point["early_stopping"])

    def _train(self):
        self.model.train()
        self.cond_pred_model.train()
        self.cond_pred_model_g.train()

        with torch.enable_grad():
            train_loss = []
            for i, (
                batch_x,
                batch_y,
                x_future,
                origin_x,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
                is_masked,  # forecasting 下始终为 False，忽略
            ) in enumerate(self.train_loader):
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                x_future = x_future.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                # Sample-level NaN filtering: only remove basins with NaN, keep the rest
                nan_mask = (torch.isnan(batch_x).any(dim=(1, 2)) |
                            torch.isnan(batch_y).any(dim=(1, 2)) |
                            torch.isnan(x_future).any(dim=(1, 2)))
                if nan_mask.all():
                    continue
                if nan_mask.any():
                    valid = ~nan_mask
                    batch_x = batch_x[valid]
                    batch_y = batch_y[valid]
                    x_future = x_future[valid]
                    batch_x_date_enc = batch_x_date_enc[valid]
                    batch_y_date_enc = batch_y_date_enc[valid]

                try:
                    loss = self._process_train_batch(
                        batch_x, batch_y, x_future, batch_x_date_enc, batch_y_date_enc
                    )
                except RuntimeError as e:
                    if "NaN/Inf DETECTED" in str(e) or "mu_backbone NaN" in str(e):
                        print(f"\n[SKIP] epoch={self.current_epoch}, step={i}, "
                              f"NaN/Inf in computation, skipping batch")
                        continue
                    raise

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)

                train_loss.append(loss.item())
                self.model_optim.step()
                self.model_optim.zero_grad()

        self.model.eval()
        self.cond_pred_model.eval()
        self.cond_pred_model_g.eval()
        return train_loss

    @staticmethod
    def _nan_check(name, tensor):
        """Check tensor for NaN/Inf. If found, raise RuntimeError with diagnostics."""
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan or has_inf:
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            total = tensor.numel()
            finite = tensor[torch.isfinite(tensor)]
            if finite.numel() > 0:
                stats = f"finite range=[{finite.min().item():.6g}, {finite.max().item():.6g}], mean={finite.mean().item():.6g}"
            else:
                stats = "ALL values are NaN/Inf"
            msg = (
                f"\n{'='*60}\n"
                f"NaN/Inf DETECTED in '{name}'\n"
                f"  shape: {list(tensor.shape)}, dtype: {tensor.dtype}\n"
                f"  NaN: {nan_count}/{total}, Inf: {inf_count}/{total}\n"
                f"  {stats}\n"
                f"{'='*60}"
            )
            raise RuntimeError(msg)

    def _process_train_batch(self, batch_x, batch_y, x_future, batch_x_mark, batch_y_mark):
        """
        Forecasting training batch processing

        Data format:
        - batch_x:  (B, T, n_feat+1) = [X(n_feat dims), Y_history(1 dim)]
        - batch_y:  (B, O, 1) = Y target (future streamflow)
        - x_future: (B, O, 42) = future forcing (known covariates)

        3 losses:
        - loss1: f(x) prediction MSE (y_0_hat vs batch_y)
        - loss2: g(x) sigma estimation MSE (gx vs y_sigma)
        - kl_loss: diffusion KL divergence
        """
        chk = self._nan_check  # shorthand

        chk("batch_x", batch_x)
        chk("batch_y", batch_y)

        batch_y_target = batch_y  # (B, O, 1)

        # 从 batch_x 提取 Y 历史（最后一列）
        y_history = batch_x[:, :, -1:]  # (B, T, 1)

        # 计算 y_sigma：从观测+目标 Y 的滑动方差
        y_full = torch.cat([y_history, batch_y_target], dim=1)  # (B, T+O, 1)
        y_sigma = wv_sigma_trailing(y_full, self.rolling_length)[:, -self.pred_len:, :] + EPS  # (B, O, 1)
        chk("y_sigma", y_sigma)

        # g(x)：从 Y 历史预测未来方差
        gx = self.cond_pred_model_g(y_history) + EPS  # (B, O, 1)
        chk("gx", gx)

        # loss2: sigma 估计损失
        loss2 = (torch.sqrt(gx) - torch.sqrt(y_sigma)).square().mean()
        chk("loss2", loss2)

        batch_y_mark_input = torch.concat([batch_x_mark[:, -self.label_len:, :], batch_y_mark], dim=1)

        # Decoder input: future X known, Y unknown (zeros)
        dec_inp_pred = torch.cat([
            x_future,
            torch.zeros([batch_x.size(0), self.pred_len, 1], device=self.device)
        ], dim=-1)  # (B, O, 43)
        dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        n = batch_x.size(0)
        t = torch.randint(
            low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)
        ).to(self.device)
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

        # f(x): 从完整 batch_x 预测 y_0_hat
        y_0_hat_batch, _ = self.cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark_input)
        chk("y_0_hat_batch (f(x) output)", y_0_hat_batch)

        # loss1: 条件预测损失
        loss1 = (y_0_hat_batch - batch_y_target).square().mean()
        chk("loss1", loss1)

        y_T_mean = y_0_hat_batch  # (B, O, 1)
        e = torch.randn_like(batch_y_target).to(self.device)  # (B, O, 1)

        forward_noise = cal_forward_noise(self.model.betas_tilde, self.model.betas_bar, gx, y_sigma, t)
        chk("forward_noise", forward_noise)
        noise = e * torch.sqrt(forward_noise)
        chk("noise", noise)
        sigma_tilde = cal_sigma_tilde(self.model.alphas, self.model.alphas_cumprod, self.model.alphas_cumprod_sum,
                                       self.model.alphas_cumprod_prev, self.model.alphas_cumprod_sum_prev,
                                       self.model.betas_tilde_m_1, self.model.betas_bar_m_1, gx, y_sigma, t)
        chk("sigma_tilde", sigma_tilde)

        # Diffusion forward process
        y_t_batch = q_sample(batch_y_target, y_T_mean, self.model.alphas_bar_sqrt,
                             self.model.one_minus_alphas_bar_sqrt, t, noise=noise)
        chk("y_t_batch", y_t_batch)

        # NsDiff model: condition + noisy Y -> noise prediction
        output, sigma_theta = self.model(batch_x, batch_x_mark, y_t_batch, y_0_hat_batch, gx, t)
        chk("output (NsDiff pred)", output)
        sigma_theta = sigma_theta + EPS
        chk("sigma_theta", sigma_theta)

        # KL loss (clamp ratio to prevent log(0) and extreme gradients)
        ratio = (sigma_tilde / sigma_theta).clamp(min=1e-6)
        chk("ratio", ratio)
        kl_loss = ((e - output)).square().mean() + \
                  ratio.mean() - torch.log(ratio).mean()
        chk("kl_loss", kl_loss)

        loss = kl_loss + loss1 + loss2
        chk("total_loss", loss)

        return loss

    def _process_val_batch(self, batch_x, batch_y, x_future, batch_x_mark, batch_y_mark):
        """
        Validation: generate probabilistic forecast samples via reverse diffusion

        Output: preds (B, O, 1, S), batch_y_target (B, O, 1)
        """
        b = batch_x.shape[0]
        minisample = self.diffusion_config.testing.minisample

        batch_y_mark_input = torch.concat([batch_x_mark[:, -self.label_len:, :], batch_y_mark], dim=1)

        # Decoder input: future X known, Y unknown (zeros)
        dec_inp_pred = torch.cat([
            x_future,
            torch.zeros([batch_x.size(0), self.pred_len, 1], device=self.device)
        ], dim=-1)  # (B, O, 43)
        dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        # f(x): predict y_0_hat
        y_0_hat_batch, _ = self.cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark_input)

        # g(x): predict gx from Y history
        y_history = batch_x[:, :, -1:]  # (B, T, 1)
        gx = self.cond_pred_model_g(y_history) + EPS  # (B, O, 1)

        preds = []
        for i in range(self.diffusion_config.testing.n_z_samples // minisample):
            repeat_n = int(minisample)
            y_0_hat_tile = y_0_hat_batch.repeat(repeat_n, 1, 1, 1)
            y_0_hat_tile = y_0_hat_tile.transpose(0, 1).flatten(0, 1).to(self.device)
            y_T_mean_tile = y_0_hat_tile
            x_tile = batch_x.repeat(repeat_n, 1, 1, 1)
            x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(self.device)

            x_mark_tile = batch_x_mark.repeat(repeat_n, 1, 1, 1)
            x_mark_tile = x_mark_tile.transpose(0, 1).flatten(0, 1).to(self.device)

            gx_tile = gx.repeat(repeat_n, 1, 1, 1)
            gx_tile = gx_tile.transpose(0, 1).flatten(0, 1).to(self.device)
            gen_y_box = []
            for _ in range(self.diffusion_config.testing.n_z_samples_depart):
                y_tile_seq = p_sample_loop(self.model, x_tile, x_mark_tile, y_0_hat_tile, gx_tile, y_T_mean_tile,
                                           self.model.num_timesteps,
                                           self.model.alphas, self.model.one_minus_alphas_bar_sqrt,
                                           self.model.alphas_cumprod, self.model.alphas_cumprod_sum,
                                           self.model.alphas_cumprod_prev, self.model.alphas_cumprod_sum_prev,
                                           self.model.betas_tilde, self.model.betas_bar,
                                           self.model.betas_tilde_m_1, self.model.betas_bar_m_1,
                                           )
                gen_y = y_tile_seq[self.model.num_timesteps].reshape(
                    b, minisample, self.pred_len, self.model.args.c_out
                ).cpu().detach()
                gen_y_box.append(gen_y)
            outputs = torch.concat(gen_y_box, dim=1)  # (B, S, O, 1)

            outputs = outputs[:, :, -self.pred_len:, :]  # (B, S, O, 1)

            preds.append(outputs.detach().cpu())

        preds = torch.concat(preds, dim=1)  # (B, total_S, O, 1)

        batch_y_target = batch_y[:, -self.pred_len:, :].to(self.device)  # (B, O, 1)

        outs = preds.permute(0, 2, 3, 1)  # (B, O, 1, S)
        assert (outs.shape[1], outs.shape[2], outs.shape[3]) == (
            self.pred_len, 1, self.diffusion_config.testing.n_z_samples), \
            f"Expected shape ({self.pred_len}, 1, {self.diffusion_config.testing.n_z_samples}), got {outs.shape[1:]}"
        return outs, batch_y_target

    def run(self, seed=42) -> Dict[str, float]:

        self._setup_run(seed)
        self._check_run_exist(seed)

        self._run_print(f"run : {self.current_run} in seed: {seed}")

        parameter_tables, model_parameters_num = count_parameters(self.model)
        self._run_print(f"parameter_tables: {parameter_tables}")
        self._run_print(f"model parameters: {model_parameters_num}")

        while self.current_epoch < self.epochs:
            epoch_start_time = time.time()
            if self.early_stopper.early_stop is True:
                self._run_print(
                    f"val loss no decreased for patience={self.patience} epochs,  early stopping ...."
                )
                break

            reproducible(seed + self.current_epoch)
            train_losses = self._train()
            self._run_print(
                "Epoch: {} cost time: {}s".format(
                    self.current_epoch + 1, time.time() - epoch_start_time
                )
            )
            self._run_print(f"Training loss : {np.mean(train_losses)}")

            # val_result = self._val()
            # self._run_print(f"Validation result: {val_result}")

            self.current_epoch = self.current_epoch + 1
            # val_loss = val_result.get('CRPS', np.mean(train_losses))
            val_loss = np.mean(train_losses)
            self.early_stopper(val_loss,
                               model={'model': self.model, 'cond_pred_model': self.cond_pred_model,
                                      'cond_pred_model_g': self.cond_pred_model_g})

            self._save_run_check_point(seed)

        self._load_best_model()
        best_test_result = self._test_and_save()

        return best_test_result

    def _test_and_save(self):
        """Test evaluation + save predictions in a single inference pass."""
        print("Testing and saving predictions...")
        self.model.eval()
        self.cond_pred_model.eval()
        self.cond_pred_model_g.eval()
        self.metrics.reset()

        all_preds = []
        all_truths = []
        all_fx_preds = []  # f(x) Transformer-only predictions for diagnostic
        metric_results = []

        with torch.no_grad():
            for batch_x, batch_y, x_future, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, *rest in self.test_loader:
                origin_y = origin_y.to(self.device)
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                x_future = x_future.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                # Sample-level NaN filtering: keep valid basins, fill NaN for invalid ones
                nan_mask = (torch.isnan(batch_x).any(dim=(1, 2)) |
                            torch.isnan(batch_y).any(dim=(1, 2)) |
                            torch.isnan(x_future).any(dim=(1, 2)))
                if nan_mask.all():
                    nan_pred = np.full((batch_x.shape[0], self.pred_len, 1), np.nan)
                    all_preds.append(nan_pred)
                    all_truths.append(batch_y.cpu().numpy())
                    all_fx_preds.append(nan_pred)
                    continue
                has_nan = nan_mask.any()
                if has_nan:
                    # Save original size and indices for reassembly
                    full_size = batch_x.shape[0]
                    valid = ~nan_mask
                    batch_x = batch_x[valid]
                    batch_y = batch_y[valid]
                    x_future = x_future[valid]
                    origin_y = origin_y[valid]
                    batch_x_date_enc = batch_x_date_enc[valid]
                    batch_y_date_enc = batch_y_date_enc[valid]

                # --- f(x) diagnostic: Transformer-only prediction ---
                batch_y_mark_input = torch.concat([batch_x_date_enc[:, -self.label_len:, :], batch_y_date_enc], dim=1)
                dec_inp_pred = torch.cat([
                    x_future,
                    torch.zeros([batch_x.size(0), self.pred_len, 1], device=self.device)
                ], dim=-1)
                dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)
                dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)
                y_0_hat_batch, _ = self.cond_pred_model(batch_x, batch_x_date_enc, dec_inp, batch_y_mark_input)
                fx_out = y_0_hat_batch.cpu().numpy()

                # --- Diffusion prediction ---
                preds, truths = self._process_val_batch(
                    batch_x, batch_y, x_future, batch_x_date_enc, batch_y_date_enc
                )

                metric_results.append(self.task_pool.apply_async(
                    update_metrics,
                    (preds.contiguous().cpu().detach(),
                     truths.contiguous().cpu().detach(),
                     self.metrics)
                ))

                pred_mean = preds.mean(dim=-1).cpu().numpy()
                truths_out = truths.cpu().numpy()

                # Reassemble: fill NaN for filtered-out basins to keep alignment
                if has_nan:
                    full_pred = np.full((full_size, self.pred_len, 1), np.nan)
                    full_fx = np.full((full_size, self.pred_len, 1), np.nan)
                    full_truth = np.full((full_size, self.pred_len, 1), np.nan)
                    valid_np = valid.cpu().numpy()
                    full_pred[valid_np] = pred_mean
                    full_fx[valid_np] = fx_out
                    full_truth[valid_np] = truths_out
                    pred_mean = full_pred
                    fx_out = full_fx
                    truths_out = full_truth

                all_fx_preds.append(fx_out)
                all_preds.append(pred_mean)
                all_truths.append(truths_out)

        # Finalize metrics
        for r in metric_results:
            r.get()
        test_result = {name: float(metric.compute()) for name, metric in self.metrics.items()}
        self._run_print(f"test_results: {test_result}")

        # Save predictions to disk
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pred_dir = os.path.join(project_root, 'output', 'pred')
        os.makedirs(pred_dir, exist_ok=True)

        all_preds = np.concatenate(all_preds, axis=0)
        pred_path = os.path.join(pred_dir, 'tst.npy')
        np.save(pred_path, all_preds)
        print(f"Predictions saved: {pred_path}, shape: {all_preds.shape}")

        # Save f(x) Transformer-only predictions for diagnostic comparison
        all_fx_preds = np.concatenate(all_fx_preds, axis=0)
        fx_path = os.path.join(pred_dir, 'tst_fx.npy')
        np.save(fx_path, all_fx_preds)
        print(f"f(x) predictions saved: {fx_path}, shape: {all_fx_preds.shape}")

        return test_result


if __name__ == "__main__":
    import fire

    fire.Fire(NsDiffCAMELSForecast)
