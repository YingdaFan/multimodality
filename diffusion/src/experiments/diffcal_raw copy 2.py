"""
DiffCal RAW Pipeline (Stage 2 works in original Y space)

Key differences from NsDiff_CAMELS.py:
1. Uses CAMELSLoaderRaw which works with y_raw_* (original scale)
2. Applies GLOBAL Y normalization (same mean/std for all basins)
3. Saves predictions in ORIGINAL scale (denormalized using global stats)
4. Allows Diffusion to calibrate mean/std errors

Data flow:
- Input: y_raw_* = true observations (non-masked) + DENORMALIZED LSTM preds (masked)
- Normalization: Global mean/std (not per-basin)
- Output: Predictions in original scale (can compare across basins)

Usage:
    python diffcal_raw.py runs --seeds='[1]'
"""

from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os
import torch
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.NsDiff import NsDiff
import src.layer.mu_backbone as ns_Transformer
import argparse
import src.layer.g_backbone as G
from src.experiments.prob_forecast import ProbForecastExp
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
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import concurrent.futures
from types import SimpleNamespace
from src.utils.sigma import wv_sigma, wv_sigma_trailing

# Import CAMELS RAW dataset and loader
from src.datasets.camels_dataset_raw import CAMELSRaw, CAMELSNpzDatasetRaw
from src.dataloader.camels_loader_raw import CAMELSLoaderRaw


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


EPS = 10e-8


class DiffCalEarlyStopping(EarlyStopping):
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model['model'].state_dict(), os.path.join(self.path, 'model.pth'))
        torch.save(model['cond_pred_model'].state_dict(), os.path.join(self.path, 'cond_pred_model.pth'))
        self.val_loss_min = val_loss


def log_normal(x, mu, var):
    eps = 1e-8
    if eps > 0.0:
        var = var + eps
    return 0.5 * torch.mean(
        np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)


@dataclass
class DiffCalRawParameters:
    num_samples: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.01
    d_model: int = 512
    n_heads: int = 4
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
    # Wave Fusion: None=unidirectional, 'interference'/'scatter'/'scatterinterference'=bidirectional
    fusion_type: str = None


@dataclass
class DiffCalRaw(ProbForecastExp, DiffCalRawParameters):
    model_type: str = "diffusion_raw"
    dataset_type: str = "CAMELS"

    def _init_dataset(self):
        """Initialize CAMELS RAW dataset"""
        self.dataset = CAMELSRaw(npz_path=self.npz_path)
        # Store global Y stats for denormalization
        self.y_global_mean = self.dataset.y_global_mean
        self.y_global_std = self.dataset.y_global_std

    def _init_data_loader(self, shuffle=False, fast_test=True, fast_val=True):
        """
        Use CAMELSLoaderRaw with global Y normalization
        """
        self._init_dataset()

        from torch_timeseries.scaler import StandardScaler
        self.scaler = StandardScaler()

        # Get masked_basin_ids from global
        masked_basins_list = globals().get('_MASKED_BASIN_IDS', None)
        if masked_basins_list:
            print(f"Masked basins (by ID): {masked_basins_list}")

        print(f"\nRAW Pipeline Configuration:")
        print(f"  Using y_raw_* (original scale)")
        print(f"  Global Y normalization: mean={self.y_global_mean:.4f}, std={self.y_global_std:.4f}")

        self.dataloader = CAMELSLoaderRaw(
            dataset=self.dataset,
            scaler=self.scaler,
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len,
            shuffle_train=shuffle,
            freq=self.dataset.freq,
            batch_size=None,
            num_worker=self.num_worker,
            fast_test=fast_test,
            fast_val=fast_val,
            npz_path=self.npz_path,
            masked_basins=masked_basins_list
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
            "enc_in": self.dataset.num_features,
            "dec_in": self.dataset.num_features,
            "c_out": 1,
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
            "time_embed": False,
            # Wave fusion: None=unidirectional, 'interference'/'scatter'/'scatterinterference'=bidirectional
            "fusion_type": self.fusion_type,
        }

        with open("./configs/nsdiff.yml", "r") as f:
            config = yaml.unsafe_load(f)
            self.diffusion_config = dict2namespace(config)

        self.args = SimpleNamespace(**args_dict)
        self.model = NsDiff(self.args, self.device).to(self.device)
        self.cond_pred_model = ns_Transformer.Model(self.args).float().to(self.device)
        self.cond_pred_model_g = None

        if self.load_pretrain:
            model_f_path = f"./results/runs/F/{self.dataset_type}/w{self.windows}h1s{self.pred_len}/1/best_model.pth"
            print("using pretrained model...")
            print(f"f(x): {model_f_path}")
            if os.path.exists(model_f_path):
                self.cond_pred_model.load_state_dict(torch.load(model_f_path, map_location=self.device, weights_only=True))

    def _init_optimizer(self):
        self.model_optim = parse_type(self.optm_type, globals=globals())(
            [{'params': self.model.parameters()},
             {'params': self.cond_pred_model.parameters()},
             ],
            lr=self.lr,
        )
        self.grad_scaler = GradScaler('cuda')

    def _setup_early_stopper(self):
        self.best_checkpoint_filepath = os.path.join(self.run_save_dir, "model.pth")
        self.best_cond_checkpoint_filepath = os.path.join(self.run_save_dir, "cond_pred_model.pth")
        self.best_cond_g_checkpoint_filepath = os.path.join(self.run_save_dir, "cond_pred_model_g.pth")
        self.early_stopper = DiffCalEarlyStopping(self.patience, verbose=True, path=self.run_save_dir)

    def _save_run_check_point(self, seed):
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
        print(f"Saving run checkpoint to '{self.run_save_dir}'.")

        self.run_state = {
            "model": self.model.state_dict(),
            "cond_pred_model": self.cond_pred_model.state_dict(),
            "current_epoch": self.current_epoch,
            "optimizer": self.model_optim.state_dict(),
            "rng_state": torch.get_rng_state(),
            "early_stopping": self.early_stopper.get_state(),
            "y_global_mean": self.y_global_mean,  # Save global stats
            "y_global_std": self.y_global_std,
        }

        torch.save(self.run_state, f"{self.run_checkpoint_filepath}")
        print("Run state saved ... ")

    def _load_best_model(self):
        self.model.load_state_dict(torch.load(self.best_checkpoint_filepath, map_location=self.device))
        self.cond_pred_model.load_state_dict(torch.load(self.best_cond_checkpoint_filepath, map_location=self.device))

    def _resume_run(self, seed):
        check_point = torch.load(self.run_checkpoint_filepath, map_location=self.device)
        self.model.load_state_dict(check_point["model"])
        self.cond_pred_model.load_state_dict(check_point["cond_pred_model"])
        self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]
        self.early_stopper.set_state(check_point["early_stopping"])

    def _train(self):
        self.model.train()
        self.cond_pred_model.train()

        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            train_loss = []
            for i, (
                batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, is_masked,
            ) in enumerate(self.train_loader):
                origin_y = origin_y.to(self.device).float()
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                is_masked = is_masked.to(self.device).squeeze(-1)  # (B,)

                # Full precision training (no bf16)
                loss = self._process_train_batch(batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, is_masked)

                loss.backward()
                progress_bar.update(batch_x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )
                self.model_optim.step()
                self.model_optim.zero_grad()

        self.model.eval()
        self.cond_pred_model.eval()
        return train_loss

    def _process_train_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, is_masked=None):
        """
        Process a training batch with optional loss masking.

        Args:
            is_masked: (B,) bool tensor, True for masked basins (exclude from loss)

        Key change: Use y_obs (LSTM prediction) directly as y_0_hat instead of cond_pred_model.
        This lets Diffusion focus on learning residual correction rather than re-predicting.
        """
        batch_y_target = batch_y
        gx = torch.ones_like(batch_y_target).to(self.device) + EPS
        y_sigma = torch.ones_like(batch_y_target).to(self.device) + EPS

        n = batch_x.size(0)
        t = torch.randint(low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

        # === KEY CHANGE: Use y_obs (LSTM prediction) directly as y_0_hat ===
        # batch_x structure: [X_features, y_obs], y_obs is the last dimension
        # This skips cond_pred_model and uses Stage 1 LSTM prediction directly
        y_0_hat_batch = batch_x[:, :, -1:].clone()  # (B, T, 1) - extract y_obs

        # Note: loss1 removed since y_0_hat is now fixed (no gradients needed)
        # Diffusion only learns to correct LSTM's prediction

        y_T_mean = y_0_hat_batch
        e = torch.randn_like(batch_y_target).to(self.device)

        forward_noise = cal_forward_noise(self.model.betas_tilde, self.model.betas_bar, gx, y_sigma, t)
        noise = e * torch.sqrt(forward_noise)
        sigma_tilde = cal_sigma_tilde(self.model.alphas, self.model.alphas_cumprod, self.model.alphas_cumprod_sum,
                                       self.model.alphas_cumprod_prev, self.model.alphas_cumprod_sum_prev,
                                       self.model.betas_tilde_m_1, self.model.betas_bar_m_1, gx, y_sigma, t)

        y_t_batch = q_sample(batch_y_target, y_T_mean, self.model.alphas_bar_sqrt,
                             self.model.one_minus_alphas_bar_sqrt, t, noise=noise)

        output, sigma_theta = self.model(batch_x, batch_x_mark, y_t_batch, y_0_hat_batch, gx, t)
        sigma_theta = sigma_theta + EPS

        # Compute per-sample kl_loss
        kl_loss_mse = ((e - output)).square().mean(dim=(1, 2))  # (B,)
        kl_loss_sigma1 = (sigma_tilde / sigma_theta).mean(dim=(1, 2))  # (B,)
        kl_loss_sigma2 = torch.log(sigma_tilde / sigma_theta).mean(dim=(1, 2))  # (B,)
        kl_loss_per_sample = kl_loss_mse + kl_loss_sigma1 - kl_loss_sigma2  # (B,)

        # loss1 removed since y_0_hat is now fixed (LSTM prediction, no gradients)
        total_loss_per_sample = kl_loss_per_sample  # (B,)

        # Apply loss masking: exclude masked basins from loss
        if is_masked is not None and is_masked.any():
            # Create mask: True for samples to INCLUDE in loss (non-masked basins)
            loss_mask = ~is_masked  # (B,)
            n_unmasked = loss_mask.sum()

            if n_unmasked > 0:
                # Only average over non-masked samples
                loss = total_loss_per_sample[loss_mask].mean()
            else:
                # All samples are masked, return zero loss (no gradient)
                loss = total_loss_per_sample.mean() * 0.0
        else:
            # No masking, use all samples
            loss = total_loss_per_sample.mean()

        return loss

    def _process_val_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        b = batch_x.shape[0]
        gen_y_by_batch_list = [[] for _ in range(self.diffusion_steps + 1)]
        minisample = self.diffusion_config.testing.minisample

        def store_gen_y_at_step_t(config, config_diff, idx, y_tile_seq):
            current_t = self.diffusion_steps - idx
            gen_y = y_tile_seq[idx].reshape(b, minisample, (config.pred_len), config.c_out).cpu()
            if len(gen_y_by_batch_list[current_t]) == 0:
                gen_y_by_batch_list[current_t] = gen_y.detach().cpu()
            else:
                gen_y_by_batch_list[current_t] = torch.concat([gen_y_by_batch_list[current_t], gen_y], dim=0).detach().cpu()
            return gen_y

        n = batch_x.size(0)
        t = torch.randint(low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

        # === KEY CHANGE: Use y_obs (LSTM prediction) directly as y_0_hat ===
        y_0_hat_batch = batch_x[:, :, -1:].clone()  # (B, T, 1) - extract y_obs
        gx = torch.ones(batch_x.size(0), self.pred_len, 1).to(self.device) + EPS

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
                for _ in range(self.diffusion_config.testing.n_z_samples_depart):
                    y_tile_seq = p_sample_loop(self.model, x_tile, x_mark_tile, y_0_hat_tile, gx_tile, y_T_mean_tile,
                                               self.model.num_timesteps,
                                               self.model.alphas, self.model.one_minus_alphas_bar_sqrt,
                                               self.model.alphas_cumprod, self.model.alphas_cumprod_sum,
                                               self.model.alphas_cumprod_prev, self.model.alphas_cumprod_sum_prev,
                                               self.model.betas_tilde, self.model.betas_bar,
                                               self.model.betas_tilde_m_1, self.model.betas_bar_m_1,
                                               )
                gen_y = store_gen_y_at_step_t(config=self.model.args, config_diff=self.diffusion_config,
                                              idx=self.model.num_timesteps, y_tile_seq=y_tile_seq)
                gen_y_box.append(gen_y.detach().cpu())
            outputs = torch.concat(gen_y_box, dim=1)
            outputs = outputs[:, :, -self.pred_len:, :]
            pred = outputs
            preds.append(pred.detach().cpu())

        preds = torch.concat(preds, dim=1)
        batch_y_target = batch_y[:, -self.pred_len:, :].to(self.device)

        outs = preds.permute(0, 2, 3, 1)
        assert (outs.shape[1], outs.shape[2], outs.shape[3]) == (
            self.pred_len, 1, self.diffusion_config.testing.n_z_samples)
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
                self._run_print(f"val loss no decreased for patience={self.patience} epochs, early stopping ....")
                break

            reproducible(seed + self.current_epoch)
            train_losses = self._train()
            self._run_print("Epoch: {} cost time: {}s".format(self.current_epoch + 1, time.time() - epoch_start_time))
            self._run_print(f"Training loss : {np.mean(train_losses)}")

            self.current_epoch = self.current_epoch + 1
            self.early_stopper(np.mean(train_losses),
                               model={'model': self.model, 'cond_pred_model': self.cond_pred_model,
                                      'cond_pred_model_g': self.cond_pred_model_g})

            self._save_run_check_point(seed)


        self._load_best_model()
        # Skip _test() - it takes ~26 min and computes metrics in normalized space
        # The actual predictions are saved in _save() which denormalizes properly
        # best_test_result = self._test()
        best_test_result = {}  # Return empty dict, metrics will be computed in postprocess
        self._save()

        return best_test_result

    def _predict(self, loader, desc="Predicting"):
        """Generate predictions and DENORMALIZE using global stats"""
        self.model.eval()
        self.cond_pred_model.eval()

        all_preds = []
        all_truths = []

        with torch.no_grad():
            for batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, is_masked in tqdm(loader, desc=desc):
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                # is_masked is not used in prediction, all basins get predictions

                preds, truths = self._process_val_batch(batch_x, batch_y, batch_x_date_enc, batch_y_date_enc)
                pred_mean = preds.mean(dim=-1)  # (B, T, 1) - still normalized

                # DENORMALIZE using global stats
                pred_mean_denorm = pred_mean * self.y_global_std + self.y_global_mean

                all_preds.append(pred_mean_denorm.cpu().numpy())
                # origin_y is already in original scale
                all_truths.append(origin_y.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_truths = np.concatenate(all_truths, axis=0)
        return all_preds, all_truths

    def _save(self):
        """Save predictions in ORIGINAL scale (denormalized)"""
        print("Generating and saving predictions (RAW pipeline - original scale)...")
        print(f"  Denormalization: pred_raw = pred_norm * {self.y_global_std:.4f} + {self.y_global_mean:.4f}")

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(project_root, 'output')
        pred_dir = os.path.join(output_dir, 'pred')
        os.makedirs(pred_dir, exist_ok=True)

        from torch.utils.data import DataLoader
        train_loader_no_shuffle = DataLoader(
            self.dataloader.train_dataset,
            batch_size=self.dataloader.batch_size,
            shuffle=False,
            num_workers=self.dataloader.num_worker,
            drop_last=False
        )

        #for partition, loader in [('trn', train_loader_no_shuffle), ('tst', self.test_loader)]:
        for partition, loader in [('trn', train_loader_no_shuffle)]:  # skip tst prediction
            preds, truths = self._predict(loader, desc=f"Predicting [{partition}]")

            pred_path = os.path.join(pred_dir, f'{partition}.npy')
            np.save(pred_path, preds)
            print(f"[{partition}] Predictions saved: {pred_path}, shape: {preds.shape}")
            print(f"  Value range: [{preds.min():.2f}, {preds.max():.2f}] (original scale)")


if __name__ == "__main__":
    from src.utils.parse_args import parse_masked_basin_ids
    import fire

    globals()['_MASKED_BASIN_IDS'] = parse_masked_basin_ids()

    fire.Fire(DiffCalRaw)
