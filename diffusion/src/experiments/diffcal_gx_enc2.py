"""
DiffCal Gx Pipeline with Dual-Head Encoder: f(x) + g(x)

Key difference from diffcal_gx_enc.py:
- Shared Encoder outputs both mean f(x) and variance g(x)
- g(x) is per-timestep (B, pred_len, 1), not a constant
- Adds loss2: explicit g(x) supervision from f(x) residuals (detached)
- gx flows through the full diffusion process (forward noise, reverse sampling)

Architecture:
    Encoder (bidirectional) -> mu_head -> y_0_hat (mean)
                            -> sigma_head -> gx (variance, via softplus)

Key design decisions:
- .detach() on y_0_hat in loss2: prevents g(x) from incentivizing worse f(x)
- lambda_g (default 0.1): small weight, KL loss is primary signal for g(x)
- softplus + zero init: gx starts near 0.693 (~1), then adapts; always positive
- Shared encoder: no extra encoder parameters, just one nn.Linear added

Usage:
    python diffcal_gx_enc2.py runs --seeds='[1]'
"""

from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.NsDiff import NsDiff
import src.layer.mu_backbone_enc as mu_backbone_enc_base
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
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import concurrent.futures
from types import SimpleNamespace
from src.utils.sigma import wv_sigma, wv_sigma_trailing

# Import CAMELS RAW dataset and loader
from src.datasets.camels_dataset_raw import CAMELSRaw, CAMELSNpzDatasetRaw
from src.dataloader.camels_loader_raw import CAMELSLoaderRaw


# ============================================================
# DualHeadModel: shared Encoder with mu_head + sigma_head
# ============================================================
class DualHeadModel(mu_backbone_enc_base.Model):
    """
    Extends mu_backbone_enc.Model with a sigma_head for g(x) prediction.

    Same Encoder produces both:
    - y_0_hat = f(x): mean prediction (via self.projection, inherited)
    - gx = g(x): variance prediction (via self.sigma_head, new)

    Returns (y_0_hat, gx) instead of (y_0_hat, output).
    """

    def __init__(self, configs):
        super().__init__(configs)
        # sigma_head: same input dim (d_model) as mu_head (self.projection)
        self.sigma_head = nn.Linear(configs.d_model, configs.c_out)
        # Zero init so initial gx = softplus(0) ≈ 0.693, close to 1
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.zeros_(self.sigma_head.bias)

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        Forward pass: shared encoder, dual output heads.

        Returns:
            y_0_hat: (B, pred_len, c_out) - mean prediction (de-normalized)
            gx: (B, pred_len, c_out) - variance prediction (always positive)
        """
        x_raw = x_enc.clone().detach()

        # Normalization (same as parent)
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        # De-stationary factors (same as parent)
        tau = self.tau_learner(x_raw, std_enc).exp()
        delta = self.delta_learner(x_raw, mean_enc)

        # Shared Encoder (same as parent)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, tau=tau, delta=delta)

        # === Head 1: f(x) mean prediction (same as parent) ===
        y_0_hat = self.projection(enc_out)
        if self.c_out == self.enc_in:
            y_0_hat = y_0_hat * std_enc + mean_enc
        else:
            std_out = std_enc[:, :, -self.c_out:]
            mean_out = mean_enc[:, :, -self.c_out:]
            y_0_hat = y_0_hat * std_out + mean_out

        # === Head 2: g(x) variance prediction (new) ===
        gx = F.softplus(self.sigma_head(enc_out)) + 1e-8

        return y_0_hat[:, -self.pred_len:, :], gx[:, -self.pred_len:, :]


# ============================================================
# Helper classes (same as diffcal_gx_enc.py)
# ============================================================
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
    """
    Computes loss weights for non-target basins based on their similarity to target basins.
    """
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


# ============================================================
# Parameters & Experiment class
# ============================================================
@dataclass
class DiffCalRaw2Parameters:
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
    lambda_g: float = 0.1  # weight for g(x) explicit supervision loss


@dataclass
class DiffCalRaw2(ProbForecastExp, DiffCalRaw2Parameters):
    model_type: str = "diffusion_gx_enc2"

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

        print(f"\nGx Dual-Head Pipeline Configuration:")
        print(f"  Backbone: DualHeadModel (shared Encoder, dual output)")
        print(f"  f(x): mu_head (mean prediction)")
        print(f"  g(x): sigma_head (variance prediction, per-timestep)")
        print(f"  lambda_g: {self.lambda_g}")
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

        self.balance_weighter = BalanceLossWeighter(
            sigma=self.balance_sigma,
            min_weight=self.balance_min_weight,
            max_weight=self.balance_max_weight
        )
        self.balance_weighter.initialize_from_dataset(self.dataset, masked_basins_list)

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
            "fusion_type": self.fusion_type,
        }

        with open("./configs/nsdiff.yml", "r") as f:
            config = yaml.unsafe_load(f)
            self.diffusion_config = dict2namespace(config)

        self.args = SimpleNamespace(**args_dict)
        self.model = NsDiff(self.args, self.device).to(self.device)
        # === Changed: use DualHeadModel instead of mu_backbone_enc.Model ===
        self.cond_pred_model = DualHeadModel(self.args).float().to(self.device)
        self.cond_pred_model_g = None

        print(f"  cond_pred_model: DualHeadModel (shared Encoder, mu_head + sigma_head)")
        print(f"  sigma_head params: {self.d_model} x 1 = {self.d_model}")

        if self.load_pretrain:
            model_f_path = f"./results/runs/F/{self.dataset_type}/w{self.windows}h1s{self.pred_len}/1/best_model.pth"
            print("using pretrained model...")
            print(f"f(x): {model_f_path}")
            if os.path.exists(model_f_path):
                self.cond_pred_model.load_state_dict(
                    torch.load(model_f_path, map_location=self.device, weights_only=True),
                    strict=False  # sigma_head won't be in pretrained weights
                )

    def _init_optimizer(self):
        self.model_optim = parse_type(self.optm_type, globals=globals())(
            [{'params': self.model.parameters()},
             {'params': self.cond_pred_model.parameters()},  # includes sigma_head
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
            "y_global_mean": self.y_global_mean,
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
                is_masked = is_masked.to(self.device).squeeze(-1)

                with autocast('cuda', dtype=torch.bfloat16):
                    loss = self._process_train_batch(batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, is_masked)

                # Skip NaN/Inf loss to prevent weight corruption
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

        batch_y_mark_input = torch.concat([batch_x_mark[:, -self.label_len:, :], batch_y_mark], dim=1)

        dec_inp_pred = torch.zeros([batch_x.size(0), self.pred_len, self.dataset.num_features]).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        n = batch_x.size(0)
        t = torch.randint(low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

        # === Changed: DualHeadModel returns (y_0_hat, gx) ===
        y_0_hat_batch, gx = self.cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark_input)
        y_sigma = gx.clone()

        # loss1: mean prediction
        loss1_per_sample = (y_0_hat_batch - batch_y_target).square().mean(dim=(1, 2))

        # === loss2: g(x) explicit supervision (detached) ===
        residual_sq = (batch_y_target - y_0_hat_batch.detach()).square()
        loss2_per_sample = (gx - residual_sq).square().mean(dim=(1, 2))

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

        kl_loss_mse = ((e - output)).square().mean(dim=(1, 2))
        kl_loss_sigma1 = (sigma_tilde / sigma_theta).mean(dim=(1, 2))
        kl_loss_sigma2 = torch.log(sigma_tilde / sigma_theta).mean(dim=(1, 2))
        kl_loss_per_sample = kl_loss_mse + kl_loss_sigma1 - kl_loss_sigma2

        # === Changed: total loss includes loss2 with lambda_g ===
        total_loss_per_sample = kl_loss_per_sample + loss1_per_sample + self.lambda_g * loss2_per_sample

        balance_weights = self.balance_weighter.compute_weights(batch_x, self.device)

        if is_masked is not None and is_masked.any():
            loss_mask = ~is_masked
            n_unmasked = loss_mask.sum()

            if n_unmasked > 0:
                weighted_loss = total_loss_per_sample[loss_mask] * balance_weights[loss_mask]
                weight_sum = balance_weights[loss_mask].sum()
                loss = weighted_loss.sum() / (weight_sum + 1e-8)
            else:
                loss = total_loss_per_sample.mean() * 0.0
        else:
            weighted_loss = total_loss_per_sample * balance_weights
            weight_sum = balance_weights.sum()
            loss = weighted_loss.sum() / (weight_sum + 1e-8)

        return loss

    def _process_val_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        b = batch_x.shape[0]
        gen_y_by_batch_list = [[] for _ in range(self.diffusion_steps + 1)]
        minisample = self.diffusion_config.testing.minisample

        batch_y_mark_input = torch.concat([batch_x_mark[:, -self.label_len:, :], batch_y_mark], dim=1)

        dec_inp_pred = torch.zeros([batch_x.size(0), self.pred_len, self.dataset.num_features]).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

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

        # === Changed: DualHeadModel returns (y_0_hat, gx) ===
        y_0_hat_batch, gx = self.cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark_input)
        gx = gx + EPS

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

            # === Changed: use learned gx instead of ones ===
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

                preds, truths = self._process_val_batch(batch_x, batch_y, batch_x_date_enc, batch_y_date_enc)

                # Probabilistic metrics (denormalized)
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
        print("Generating and saving predictions (Dual-Head Encoder backbone)...")
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

    fire.Fire(DiffCalRaw2)
