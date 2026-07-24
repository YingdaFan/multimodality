
from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os
import torch
from dataclasses import dataclass, asdict, field
from src.models.TMDM import TMDM
import src.nn.tmdm_ns_transformer as ns_Transformer
from src.experiments.prob_forecast import ProbForecastExp, update_metrics
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from torch.optim import *
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
import torch.multiprocessing as mp
from torch_timeseries.utils.parse_type import parse_type
from torch_timeseries.utils.early_stop import EarlyStopping
from src.nn.tmdm_diffusion_utils import q_sample, p_sample_loop
import numpy as np
import torch
import concurrent.futures
from types import SimpleNamespace

from src.datasets.forecast_dataset import ForecastMeta
from src.dataloader.forecast_loader import ForecastLoader


class TMDMEarlyStopping(EarlyStopping):
    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
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
class TMDMParameters:
    num_samples: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.5
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 1024
    diffusion_steps: int = 100
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
    npz_path: str = '../data_processing/data/prepped.npz'


@dataclass
class TMDMForecast(ProbForecastExp, TMDMParameters):
    model_type: str = "TMDM_forecast"
    dataset_type: str = "CAMELS"

    def _init_dataset(self):
        self.dataset = ForecastMeta(npz_path=self.npz_path)

    def _init_data_loader(self, shuffle=True, fast_test=True, fast_val=True):
        self._init_dataset()
        from torch_timeseries.scaler import StandardScaler
        self.scaler = StandardScaler()
        self.dataloader = ForecastLoader(
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
            masked_basins=None,
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
            "enc_in": self.dataset.num_features,  # 43
            "dec_in": self.dataset.num_features,
            "c_out": 1,  # MS mode: predict Y only
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "e_layers": self.e_layers,
            "d_layers": self.d_layers,
            "d_ff": self.d_ff,
            "moving_avg": self.moving_avg,
            "timesteps": self.diffusion_steps,
            "factor": self.factor,
            "distil": self.distil,
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
            "diffusion_config_dir": "./configs/tmdm.yml",
            "time_embed": False,
        }

        self.args = SimpleNamespace(**args_dict)
        self.model = TMDM(self.args, self.device).to(self.device)
        self.cond_pred_model = ns_Transformer.Model(self.args).float().to(self.device)

    def _init_optimizer(self):
        self.model_optim = parse_type(self.optm_type, globals=globals())(
            [{'params': self.model.parameters()},
             {'params': self.cond_pred_model.parameters()}],
            lr=self.lr,
        )

    def _setup_early_stopper(self):
        self.best_checkpoint_filepath = os.path.join(
            self.run_save_dir, "model.pth"
        )
        self.best_cond_checkpoint_filepath = os.path.join(
            self.run_save_dir, "cond_pred_model.pth"
        )
        self.early_stopper = TMDMEarlyStopping(
            self.patience, verbose=True, path=self.run_save_dir
        )

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

        with torch.enable_grad():
            train_loss = []
            for i, (
                batch_x, batch_y, x_future, origin_x, origin_y,
                batch_x_date_enc, batch_y_date_enc, is_masked,
            ) in enumerate(self.train_loader):
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                x_future = x_future.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                self.model_optim.zero_grad()
                loss = self._process_train_batch(
                    batch_x, batch_y, x_future, batch_x_date_enc, batch_y_date_enc
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                train_loss.append(loss.item())
                self.model_optim.step()

        self.model.eval()
        self.cond_pred_model.eval()
        return train_loss

    def _process_train_batch(self, batch_x, batch_y, x_future, batch_x_mark, batch_y_mark):
        """
        Training: TMDM uses encoder-decoder with diffusion.
        - batch_x:  (B, T, 43) = [X(42), Y_history(1)]
        - batch_y:  (B, O, 1)  = future streamflow target
        - x_future: (B, O, 42) = known future meteorological forcing
        """
        batch_y_target = batch_y  # (B, O, 1)

        # Construct label_len + pred_len target: [tail of history Y, future Y]
        # For TMDM, batch_y includes label_len prefix
        y_label = batch_x[:, -self.label_len:, -1:]  # (B, label_len, 1)
        batch_y_full = torch.cat([y_label, batch_y_target], dim=1)  # (B, label_len+O, 1)

        batch_y_mark_input = torch.cat([
            batch_x_mark[:, -self.label_len:, :], batch_y_mark
        ], dim=1)

        # Decoder input: [label from history(43), future_X + zeros_Y(43)]
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

        # f(x): VAE condition prediction
        _, y_0_hat_batch, KL_loss, z_sample = self.cond_pred_model(
            batch_x, batch_x_mark, dec_inp, batch_y_mark_input
        )
        loss_vae = log_normal(batch_y_full, y_0_hat_batch, torch.from_numpy(np.array(1)))
        loss_vae_all = loss_vae + self.k_z * KL_loss

        y_T_mean = y_0_hat_batch
        e = torch.randn_like(batch_y_full).to(self.device)

        y_t_batch = q_sample(
            batch_y_full, y_T_mean,
            self.model.alphas_bar_sqrt, self.model.one_minus_alphas_bar_sqrt,
            t, noise=e
        )

        output = self.model(batch_x, batch_x_mark, batch_y_full, y_t_batch, y_0_hat_batch, t)
        loss = (e - output).square().mean() + self.args.k_cond * loss_vae_all
        return loss

    def _process_val_batch(self, batch_x, batch_y, x_future, batch_x_mark, batch_y_mark):
        """
        Inference: generate probabilistic forecast samples via reverse diffusion.
        Output: preds (B, O, 1, S), batch_y_target (B, O, 1)
        """
        b = batch_x.shape[0]
        minisample = self.model.diffusion_config.testing.minisample if hasattr(
            self.model.diffusion_config.testing, 'minisample') else 10

        batch_y_mark_input = torch.cat([
            batch_x_mark[:, -self.label_len:, :], batch_y_mark
        ], dim=1)

        # Decoder input: [label, future_X + zeros_Y]
        dec_inp_pred = torch.cat([
            x_future,
            torch.zeros([batch_x.size(0), self.pred_len, 1], device=self.device)
        ], dim=-1)
        dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        # f(x): predict y_0_hat
        _, y_0_hat_batch, _, z_sample = self.cond_pred_model(
            batch_x, batch_x_mark, dec_inp, batch_y_mark_input
        )

        preds = []
        for i in range(self.model.diffusion_config.testing.n_z_samples // minisample):
            repeat_n = int(minisample)
            y_0_hat_tile = y_0_hat_batch.repeat(repeat_n, 1, 1, 1)
            y_0_hat_tile = y_0_hat_tile.transpose(0, 1).flatten(0, 1).to(self.device)
            y_T_mean_tile = y_0_hat_tile
            x_tile = batch_x.repeat(repeat_n, 1, 1, 1)
            x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(self.device)
            x_mark_tile = batch_x_mark.repeat(repeat_n, 1, 1, 1)
            x_mark_tile = x_mark_tile.transpose(0, 1).flatten(0, 1).to(self.device)

            gen_y_box = []
            for _ in range(self.model.diffusion_config.testing.n_z_samples_depart):
                y_tile_seq = p_sample_loop(
                    self.model, x_tile, x_mark_tile, y_0_hat_tile, y_T_mean_tile,
                    self.model.num_timesteps,
                    self.model.alphas, self.model.one_minus_alphas_bar_sqrt
                )
                gen_y = y_tile_seq[self.model.num_timesteps].reshape(
                    b, minisample,
                    (self.model.args.label_len + self.model.args.pred_len),
                    self.model.args.c_out
                ).cpu().detach()
                gen_y_box.append(gen_y)
            outputs = torch.cat(gen_y_box, dim=1)  # (B, S, label_len+O, 1)
            outputs = outputs[:, :, -self.pred_len:, :]  # (B, S, O, 1)
            preds.append(outputs.detach().cpu())

        preds = torch.cat(preds, dim=1)  # (B, total_S, O, 1)
        batch_y_target = batch_y[:, -self.pred_len:, :].to(self.device)

        outs = preds.permute(0, 2, 3, 1)  # (B, O, 1, S)
        assert (outs.shape[1], outs.shape[2], outs.shape[3]) == (
            self.pred_len, 1, self.model.diffusion_config.testing.n_z_samples), \
            f"Expected ({self.pred_len}, 1, {self.model.diffusion_config.testing.n_z_samples}), got {outs.shape[1:]}"
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
                    f"val loss no decreased for patience={self.patience} epochs, early stopping ...."
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

            self.current_epoch += 1
            val_loss = np.mean(train_losses)
            self.early_stopper(
                val_loss,
                model={'model': self.model, 'cond_pred_model': self.cond_pred_model}
            )
            self._save_run_check_point(seed)

        self._load_best_model()
        best_test_result = self._test_and_save()
        return best_test_result

    def _test_and_save(self):
        """Test evaluation + save predictions."""
        print("Testing and saving predictions...")
        self.model.eval()
        self.cond_pred_model.eval()
        self.metrics.reset()

        all_preds = []
        all_truths = []
        metric_results = []

        with torch.no_grad():
            for batch_x, batch_y, x_future, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, *rest in self.test_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                x_future = x_future.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                preds, truths = self._process_val_batch(
                    batch_x, batch_y, x_future, batch_x_date_enc, batch_y_date_enc
                )
                metric_results.append(self.task_pool.apply_async(
                    update_metrics,
                    (preds.contiguous().cpu().detach(),
                     truths.contiguous().cpu().detach(),
                     self.metrics)
                ))
                pred_mean = preds.mean(dim=-1)
                all_preds.append(pred_mean.cpu().numpy())
                all_truths.append(truths.cpu().numpy())

        for r in metric_results:
            r.get()
        test_result = {name: float(metric.compute()) for name, metric in self.metrics.items()}
        self._run_print(f"test_results: {test_result}")

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pred_dir = os.path.join(project_root, 'output', 'pred')
        os.makedirs(pred_dir, exist_ok=True)
        all_preds = np.concatenate(all_preds, axis=0)
        pred_path = os.path.join(pred_dir, 'tst.npy')
        np.save(pred_path, all_preds)
        print(f"Predictions saved: {pred_path}, shape: {all_preds.shape}")
        return test_result


if __name__ == "__main__":
    import fire
    fire.Fire(TMDMForecast)
