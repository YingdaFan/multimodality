"""
FlowCal Gx Pipeline with Pure Encoder Backbone (Flow Matching variant)

Source p_0 : x_0 = y_0_hat + sigma_src * eps
Target p_1 : x_1 = y_obs (NaN filled with y_0_hat.detach() for safety)
Path       : x_t = (1 - t) * x_0 + t * x_1   (linear interpolant)
Velocity   : v* = x_1 - x_0
Loss       : MSE(u_theta(x_t, t, cond), v*) on observed timesteps only
Sampling   : Deterministic Euler ODE from t=0 to t=1.

Usage:
    python fmcal_gx_enc.py runs --seeds='[1]'
"""

from dataclasses import dataclass, field
import sys
from typing import List, Dict, Optional
import os
import torch
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.NsDiff import NsDiff
import src.layer.mu_backbone_enc as ns_Transformer
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
import yaml
import numpy as np
import torch.distributed as dist
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import concurrent.futures
from types import SimpleNamespace
from src.utils.sigma import wv_sigma, wv_sigma_trailing

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


class BalanceLossWeighter:
    def __init__(self, sigma=0.5, min_weight=0.1, max_weight=2.0):
        self.sigma = sigma
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.target_mean_center = None
        self.target_std_center = None
        self.initialized = False

    def initialize_from_dataset(self, dataset, masked_basin_ids):
        if masked_basin_ids is None or len(masked_basin_ids) == 0:
            print("[BalanceLossWeighter] No target basins specified, using uniform weights")
            self.initialized = False
            return

        basin_ids = dataset.basin_ids
        target_indices = []
        for basin_id in masked_basin_ids:
            if basin_id in basin_ids:
                idx = basin_ids.index(basin_id)
                target_indices.append(idx)
            else:
                print(f"[BalanceLossWeighter] Warning: basin {basin_id} not found in dataset")

        if len(target_indices) == 0:
            print("[BalanceLossWeighter] No target basins found in dataset, using uniform weights")
            self.initialized = False
            return

        y_mean_vae_all = dataset.y_mean_vae
        y_std_vae_all = dataset.y_std_vae

        target_means = y_mean_vae_all[target_indices]
        target_stds = y_std_vae_all[target_indices]

        self.target_mean_center = float(np.mean(target_means))
        self.target_std_center = float(np.mean(target_stds))

        print(f"[BalanceLossWeighter] Initialized with {len(target_indices)} target basins")
        print(f"  Target distribution center: mean_vae={self.target_mean_center:.4f}, std_vae={self.target_std_center:.4f}")
        self.initialized = True

    def compute_weights(self, batch_x, device):
        if not self.initialized:
            return torch.ones(batch_x.size(0), device=device)

        y_mean_vae = batch_x[:, 0, -3]
        y_std_vae = batch_x[:, 0, -2]

        mean_diff = y_mean_vae - self.target_mean_center
        std_diff = y_std_vae - self.target_std_center
        distance_sq = mean_diff ** 2 + std_diff ** 2
        similarity = torch.exp(-distance_sq / (2 * self.sigma ** 2))
        weights = self.min_weight + (self.max_weight - self.min_weight) * similarity
        return weights


class FMCalEarlyStopping(EarlyStopping):
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model['model'].state_dict(), os.path.join(self.path, 'model.pth'))
        torch.save(model['cond_pred_model'].state_dict(), os.path.join(self.path, 'cond_pred_model.pth'))
        self.val_loss_min = val_loss


@dataclass
class FMCalRawParameters:
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
    fusion_type: str = None
    balance_sigma: float = 0.5
    balance_min_weight: float = 0.1
    balance_max_weight: float = 2.0
    # Flow-matching specific
    fm_source_sigma: float = 0.1            # set <=0 to auto-estimate from prior residual after warmup
    fm_source_sigma_warmup: float = 0.3     # value used during epoch 0 when auto-estimating
    fm_source_sigma_floor: float = 0.05
    fm_source_sigma_ceil: float = 1.0
    fm_steps: int = 20                      # ODE Euler steps at inference


@dataclass
class FMCalRaw(ProbForecastExp, FMCalRawParameters):
    model_type: str = "flowmatching_gx_enc"
    dataset_type: str = "CAMELS"

    def _init_dataset(self):
        self.dataset = CAMELSRaw(npz_path=self.npz_path)
        self.y_global_mean = self.dataset.y_global_mean
        self.y_global_std = self.dataset.y_global_std

    def _init_data_loader(self, shuffle=False, fast_test=True, fast_val=True):
        self._init_dataset()

        from torch_timeseries.scaler import StandardScaler
        self.scaler = StandardScaler()

        masked_basins_list = globals().get('_MASKED_BASIN_IDS', None)
        if masked_basins_list:
            print(f"Masked basins (by ID): {masked_basins_list}")

        print(f"\nGx Pipeline Configuration (Pure Encoder, FLOW MATCHING):")
        print(f"  Backbone: mu_backbone_enc (no Decoder, fully bidirectional)")
        print(f"  Source p_0: y_0_hat + sigma_src * N(0,I)")
        print(f"  Target p_1: y_obs (NaN filled with y_0_hat.detach())")
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

        self.balance_weighter = BalanceLossWeighter(
            sigma=self.balance_sigma,
            min_weight=self.balance_min_weight,
            max_weight=self.balance_max_weight
        )
        self.balance_weighter.initialize_from_dataset(self.dataset, masked_basins_list)


    @property
    def condition_dim(self):
        """Width of the condition tensor fed to the diffusion models.
        Defaults to the dataset's feature count; a subclass that widens the
        condition (e.g. with live embeddings) overrides this one property
        instead of mutating dataset.num_features."""
        return self.dataset.num_features
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
            "enc_in": self.condition_dim,
            "dec_in": self.condition_dim,
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
            "fusion_type": self.fusion_type,
        }

        with open("./configs/nsdiff.yml", "r") as f:
            config = yaml.unsafe_load(f)
            self.diffusion_config = dict2namespace(config)

        self.args = SimpleNamespace(**args_dict)
        # NsDiff's primary output is reinterpreted as velocity. Its sigma_theta
        # output is unsupervised in FM and intentionally ignored.
        self.model = NsDiff(self.args, self.device).to(self.device)
        self.cond_pred_model = ns_Transformer.Model(self.args).float().to(self.device)
        self.cond_pred_model_g = None

        self.fm_num_steps = int(self.fm_steps if self.fm_steps else self.model.num_timesteps)

        # Resolve fm_source_sigma: fixed value, or auto (<=0) with warmup-then-estimate.
        self._auto_estimate_sigma = (self.fm_source_sigma is None) or (self.fm_source_sigma <= 0)
        if self._auto_estimate_sigma:
            self.fm_source_sigma = float(self.fm_source_sigma_warmup)
            print(f"  fm_source_sigma: auto (warmup={self.fm_source_sigma_warmup}, "
                  f"re-estimates after epoch 0)")
        else:
            print(f"  fm_source_sigma: {self.fm_source_sigma:.4f} (fixed)")
        print(f"  cond_pred_model: mu_backbone_enc.Model (pure Encoder)")
        print(f"  FM Euler steps at inference: {self.fm_num_steps}")

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
        self.early_stopper = FMCalEarlyStopping(self.patience, verbose=True, path=self.run_save_dir)

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
            "y_global_mean": self.y_global_mean,
            "y_global_std": self.y_global_std,
            "fm_source_sigma_resolved": float(self.fm_source_sigma),
            "auto_estimate_sigma": self._auto_estimate_sigma,
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
        if "fm_source_sigma_resolved" in check_point:
            self.fm_source_sigma = check_point["fm_source_sigma_resolved"]
        if "auto_estimate_sigma" in check_point:
            self._auto_estimate_sigma = check_point["auto_estimate_sigma"]

    @torch.no_grad()
    def _estimate_prior_residual_std(self, n_batches: int = 20) -> float:
        """
        Run cond_pred_model on a few training batches and estimate the std of
        the residual (y_obs - y_0_hat) over observed timesteps. Used to set
        fm_source_sigma so the source perturbation matches the prior's typical
        error scale.
        """
        was_training = self.cond_pred_model.training
        self.cond_pred_model.eval()
        residuals = []
        for i, batch in enumerate(self.train_loader):
            if i >= n_batches:
                break
            batch_x, batch_y, _, _, batch_x_date_enc, batch_y_date_enc, _ = batch
            batch_x = batch_x.to(self.device).float()
            batch_y = batch_y.to(self.device).float()
            batch_x_date_enc = batch_x_date_enc.to(self.device).float()
            batch_y_date_enc = batch_y_date_enc.to(self.device).float()

            batch_y_mark_input = torch.concat(
                [batch_x_date_enc[:, -self.label_len:, :], batch_y_date_enc], dim=1)
            dec_inp_pred = torch.zeros(
                [batch_x.size(0), self.pred_len, self.condition_dim]
            ).to(self.device)
            dec_inp_label = batch_x[:, -self.label_len:, :]
            dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

            y_0_hat, _ = self.cond_pred_model(batch_x, batch_x_date_enc, dec_inp, batch_y_mark_input)
            valid = ~torch.isnan(batch_y)
            if valid.any():
                r = (batch_y - y_0_hat)[valid]
                residuals.append(r.detach().cpu())

        if was_training:
            self.cond_pred_model.train()

        if not residuals:
            print("[FM] residual estimation: no valid points found, falling back to warmup default")
            return float(self.fm_source_sigma_warmup)
        all_r = torch.cat(residuals, dim=0)
        std = float(all_r.std().item())
        std_clipped = max(self.fm_source_sigma_floor, min(std, self.fm_source_sigma_ceil))
        print(f"[FM] estimated prior residual std = {std:.4f} -> sigma_src = {std_clipped:.4f} "
              f"(clipped to [{self.fm_source_sigma_floor}, {self.fm_source_sigma_ceil}])")
        return std_clipped

    def _train(self):
        # Re-estimate sigma_src after epoch 0 so cond_pred_model has trained first.
        if self._auto_estimate_sigma and self.current_epoch >= 1:
            self.fm_source_sigma = self._estimate_prior_residual_std()
            self._auto_estimate_sigma = False

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
                is_masked = is_masked.to(self.device).squeeze(-1)

                with autocast('cuda', dtype=torch.bfloat16):
                    loss = self._process_train_batch(
                        batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, is_masked)

                if torch.isnan(loss) or torch.isinf(loss):
                    self.model_optim.zero_grad()
                    progress_bar.update(batch_x.size(0))
                    continue

                self.grad_scaler.scale(loss).backward()
                progress_bar.update(batch_x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    sigma_src=f"{self.fm_source_sigma:.3f}",
                    refresh=True,
                )
                self.grad_scaler.step(self.model_optim)
                self.grad_scaler.update()
                self.model_optim.zero_grad()

        self.model.eval()
        self.cond_pred_model.eval()
        return train_loss

    def _process_train_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, is_masked=None):
        batch_y_target = batch_y
        gx = torch.ones_like(batch_y_target).to(self.device) + EPS

        batch_y_mark_input = torch.concat(
            [batch_x_mark[:, -self.label_len:, :], batch_y_mark], dim=1)

        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.condition_dim]).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        n = batch_x.size(0)
        # Stratified t coverage: clamp BEFORE pairing to keep t<->1-t symmetric.
        half = (n + 1) // 2
        t_half = torch.rand(half, device=self.device).clamp(min=1e-3, max=1 - 1e-3)
        t_cont = torch.cat([t_half, 1.0 - t_half], dim=0)[:n]
        t_idx = (t_cont * self.model.num_timesteps).long().clamp(
            min=0, max=self.model.num_timesteps - 1)

        y_0_hat_batch, _ = self.cond_pred_model(
            batch_x, batch_x_mark, dec_inp, batch_y_mark_input)

        nan_mask = torch.isnan(batch_y_target)
        valid_mask = ~nan_mask
        valid_count_per_sample = valid_mask.sum(dim=(1, 2)).float()
        sample_has_obs = valid_count_per_sample > 0

        batch_y_filled = batch_y_target.clone()
        if nan_mask.any():
            batch_y_filled[nan_mask] = y_0_hat_batch[nan_mask].detach()

        loss1_diff = (y_0_hat_batch - batch_y_filled).square() * valid_mask.float()
        loss1_per_sample = loss1_diff.sum(dim=(1, 2)) / valid_count_per_sample.clamp(min=1.0)

        # Conditional Flow Matching with informed source.
        # y_0_hat is NOT detached: v_target itself depends on it, so FM loss
        # propagates into cond_pred_model through both target and v_pred paths.
        # loss1 keeps the joint optimization well-posed.
        eps = torch.randn_like(batch_y_filled)
        x_0 = y_0_hat_batch + self.fm_source_sigma * eps
        x_1 = batch_y_filled

        t_b = t_cont.view(-1, 1, 1)
        x_t = (1.0 - t_b) * x_0 + t_b * x_1
        v_target = x_1 - x_0

        v_pred, _ = self.model(batch_x, batch_x_mark, x_t, y_0_hat_batch, gx, t_idx)

        valid_f = valid_mask.float()
        fm_loss_per_sample = ((v_pred - v_target).square() * valid_f).sum(dim=(1, 2)) \
                             / valid_count_per_sample.clamp(min=1.0)

        total_loss_per_sample = fm_loss_per_sample + loss1_per_sample

        balance_weights = self.balance_weighter.compute_weights(batch_x, self.device)

        if is_masked is not None and is_masked.any():
            loss_mask = (~is_masked) & sample_has_obs
        else:
            loss_mask = sample_has_obs

        if loss_mask.sum() > 0:
            weighted_loss = total_loss_per_sample[loss_mask] * balance_weights[loss_mask]
            weight_sum = balance_weights[loss_mask].sum()
            loss = weighted_loss.sum() / (weight_sum + 1e-8)
        else:
            loss = total_loss_per_sample.mean() * 0.0

        return loss

    @torch.no_grad()
    def _fm_euler_sample(self, batch_x, batch_x_mark, y_0_hat, gx, source_eps=None):
        """Deterministic Euler ODE from t=0 to t=1. Ensemble spread comes from
        fresh source_eps per call; sigma_theta is unused (no FM supervision)."""
        N = self.fm_num_steps
        dt = 1.0 / N

        if source_eps is None:
            source_eps = torch.randn_like(y_0_hat)
        x = y_0_hat + self.fm_source_sigma * source_eps

        for i in range(N):
            t_cont_val = (i + 0.5) / N
            t_idx_val = int(t_cont_val * self.model.num_timesteps)
            t_idx_val = max(0, min(t_idx_val, self.model.num_timesteps - 1))
            t_idx = torch.full((x.size(0),), t_idx_val,
                               device=self.device, dtype=torch.long)
            v, _ = self.model(batch_x, batch_x_mark, x, y_0_hat, gx, t_idx)
            x = x + v * dt

        return x

    def _process_val_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        b = batch_x.shape[0]
        minisample = self.diffusion_config.testing.minisample

        batch_y_mark_input = torch.concat(
            [batch_x_mark[:, -self.label_len:, :], batch_y_mark], dim=1)

        dec_inp_pred = torch.zeros(
            [batch_x.size(0), self.pred_len, self.condition_dim]).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        y_0_hat_batch, _ = self.cond_pred_model(
            batch_x, batch_x_mark, dec_inp, batch_y_mark_input)
        gx = torch.ones(batch_x.size(0), self.pred_len, 1).to(self.device) + EPS

        preds = []
        for _ in range(self.diffusion_config.testing.n_z_samples // minisample):
            repeat_n = int(minisample)
            y_0_hat_tile = y_0_hat_batch.repeat(repeat_n, 1, 1, 1)
            y_0_hat_tile = y_0_hat_tile.transpose(0, 1).flatten(0, 1).to(self.device)
            x_tile = batch_x.repeat(repeat_n, 1, 1, 1)
            x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(self.device)
            x_mark_tile = batch_x_mark.repeat(repeat_n, 1, 1, 1)
            x_mark_tile = x_mark_tile.transpose(0, 1).flatten(0, 1).to(self.device)
            gx_tile = gx.repeat(repeat_n, 1, 1, 1)
            gx_tile = gx_tile.transpose(0, 1).flatten(0, 1).to(self.device)

            gen_y_box = []
            for _ in range(self.diffusion_config.testing.n_z_samples_depart):
                gen_y = self._fm_euler_sample(x_tile, x_mark_tile, y_0_hat_tile, gx_tile)
                gen_y = gen_y.reshape(b, minisample, self.pred_len, self.args.c_out).cpu()
                gen_y_box.append(gen_y.detach().cpu())

            outputs = torch.concat(gen_y_box, dim=1)
            outputs = outputs[:, :, -self.pred_len:, :]
            preds.append(outputs.detach().cpu())

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
            self._run_print("Epoch: {} cost time: {}s".format(
                self.current_epoch + 1, time.time() - epoch_start_time))
            self._run_print(f"Training loss : {np.mean(train_losses)}")

            self.current_epoch = self.current_epoch + 1
            self.early_stopper(np.mean(train_losses),
                               model={'model': self.model, 'cond_pred_model': self.cond_pred_model,
                                      'cond_pred_model_g': self.cond_pred_model_g})

            self._save_run_check_point(seed)

        self._load_best_model()
        best_test_result = {}
        self._save()

        return best_test_result

    def _predict(self, loader, desc="Predicting"):
        self.model.eval()
        self.cond_pred_model.eval()

        all_preds = []
        all_truths = []
        self.metrics.reset()
        metric_results = []

        with torch.no_grad():
            for batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, is_masked in tqdm(loader, desc=desc):
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                preds, truths = self._process_val_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc)

                preds_denorm = preds * self.y_global_std + self.y_global_mean
                metrics_truths = origin_y.unsqueeze(-1) if origin_y.dim() == 2 else origin_y
                metric_results.append(self.task_pool.apply_async(
                    update_metrics,
                    (preds_denorm.contiguous().cpu().detach(),
                     metrics_truths.contiguous().cpu().detach(),
                     self.metrics)
                ))

                pred_mean = preds.mean(dim=-1)
                pred_mean_denorm = pred_mean * self.y_global_std + self.y_global_mean

                all_preds.append(pred_mean_denorm.cpu().numpy())
                all_truths.append(origin_y.cpu().numpy())

        for r in metric_results:
            r.get()
        prob_metrics = {name: float(metric.compute()) for name, metric in self.metrics.items()}
        print(f"[{desc}] Probabilistic metrics (denormalized): {prob_metrics}")

        all_preds = np.concatenate(all_preds, axis=0)
        all_truths = np.concatenate(all_truths, axis=0)
        return all_preds, all_truths

    def _save(self):
        print("Generating and saving predictions (Pure Encoder backbone, Flow Matching)...")
        print(f"  Denormalization: pred_raw = pred_norm * {self.y_global_std:.4f} + {self.y_global_mean:.4f}")
        print(f"  Resolved fm_source_sigma: {self.fm_source_sigma:.4f}")

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # FM outputs isolated to output_fm/ so they don't collide with diffusion's output/.
        output_dir = os.path.join(project_root, 'output_fm')
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

        for partition, loader in [('trn', train_loader_no_shuffle)]:
            preds, truths = self._predict(loader, desc=f"Predicting [{partition}]")

            pred_path = os.path.join(pred_dir, f'{partition}.npy')
            np.save(pred_path, preds)
            print(f"[{partition}] Predictions saved: {pred_path}, shape: {preds.shape}")
            print(f"  Value range: [{preds.min():.2f}, {preds.max():.2f}] (original scale)")


if __name__ == "__main__":
    from src.utils.parse_args import parse_masked_basin_ids
    import fire

    globals()['_MASKED_BASIN_IDS'] = parse_masked_basin_ids()

    fire.Fire(FMCalRaw)
