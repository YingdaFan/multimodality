"""
DiffCal Gx Pipeline with Basin-Adaptive gx

Inherits from diffcal_raw, only modifies the gx computation method:
- Original: gx = 1 (constant)
- This version: gx = BasinVarianceEstimator(X, Y_obs)

All other logic remains unchanged.

Usage:
    python diffcal_gx.py runs --seeds='[1]'
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *

# Inherits from the Raw version
from src.experiments.diffcal_raw import DiffCalRaw, DiffCalRawParameters


EPS = 10e-8


# ============================================================
# BasinVarianceEstimator (embedded in this file)
# ============================================================
class BasinVarianceEstimator(nn.Module):
    """
    Estimates basin-specific gx (variance scaling factor) from input statistical features.

    Input: batch_x (B, T, enc_in), where enc_in = X_dim + y_mean_vae + y_std_vae + Y_obs
    Output: gx_scale (B, 1, 1), broadcastable to (B, T, 1)

    Output is always positive (via Softplus) and centered at 1.0,
    to maintain compatibility with the original diffusion formula.

    batch_x structure: [X(42), y_mean_vae(1), y_std_vae(1), y_obs(1)] = 45 dims
    """

    def __init__(self, enc_in=45, hidden_dim=64, output_bias=1.0):
        super().__init__()
        self.enc_in = enc_in
        self.output_bias = output_bias

        # Statistical feature dimensions: 6(X) + 6(Y) + 6(interaction) + 2(basin stats) = 20
        stat_dim = 20

        self.net = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize output near 0 (so gx initial value is close to output_bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def compute_statistics(self, batch_x):
        """
        Compute statistical features from input time series

        batch_x structure: [X(42), y_mean_vae(1), y_std_vae(1), y_obs(1)] = 45 dims
        """
        B, T, C = batch_x.shape

        # Extract each component
        x_features = batch_x[:, :, :-3]      # (B, T, 42) - X features
        y_mean_vae = batch_x[:, 0, -3]       # (B,) - per-basin Y mean (constant along T)
        y_std_vae = batch_x[:, 0, -2]        # (B,) - per-basin Y std (constant along T)
        y_obs = batch_x[:, :, -1:]           # (B, T, 1) - LSTM predictions (global normalized)

        # X statistics
        x_mean = x_features.mean(dim=(1, 2))
        x_std = x_features.std(dim=(1, 2)) + 1e-6
        x_min = x_features.min(dim=1).values.min(dim=1).values
        x_max = x_features.max(dim=1).values.max(dim=1).values
        x_range = x_max - x_min + 1e-6
        x_cv = x_std / (x_mean.abs() + 1e-6)

        # Y_obs statistics
        y_mean = y_obs.mean(dim=1).squeeze(-1)
        y_std = y_obs.std(dim=1).squeeze(-1) + 1e-6
        y_min = y_obs.min(dim=1).values.squeeze(-1)
        y_max = y_obs.max(dim=1).values.squeeze(-1)
        y_range = y_max - y_min + 1e-6
        y_cv = y_std / (y_mean.abs() + 1e-6)

        # X-Y interaction statistics
        x_temporal_mean = x_features.mean(dim=2)
        y_temporal = y_obs.squeeze(-1)
        xy_cov = ((x_temporal_mean - x_temporal_mean.mean(dim=1, keepdim=True)) *
                  (y_temporal - y_temporal.mean(dim=1, keepdim=True))).mean(dim=1)
        xy_corr = xy_cov / (x_temporal_mean.std(dim=1) * y_temporal.std(dim=1) + 1e-6)
        var_ratio = y_std / (x_std + 1e-6)
        range_ratio = y_range / (x_range + 1e-6)
        mean_diff = (y_mean - x_mean).abs()
        std_diff = (y_std - x_std).abs()

        # Combine all statistics, including basin-specific stats
        stats = torch.stack([
            x_mean, x_std, x_min, x_max, x_range, x_cv,
            y_mean, y_std, y_min, y_max, y_range, y_cv,
            xy_corr, var_ratio, range_ratio, mean_diff, std_diff,
            y_mean_vae,   # per-basin Y mean (original scale information)
            y_std_vae,    # per-basin Y std (original scale information)
            torch.zeros_like(x_mean),  # padding
        ], dim=-1)

        return stats

    def forward(self, batch_x):
        """
        Compute basin-adaptive gx scale.

        Returns:
            gx_scale: (B, 1, 1) - variance scaling factor
        """
        stats = self.compute_statistics(batch_x)
        delta = self.net(stats)

        # Ensure output is positive, centered at output_bias
        gx_scale = self.output_bias + F.softplus(delta) - F.softplus(torch.zeros(1, device=delta.device))
        gx_scale = gx_scale.clamp(min=0.1, max=10.0)

        return gx_scale.unsqueeze(-1)  # (B, 1, 1)


# ============================================================
# DiffCalGx main class
# ============================================================
@dataclass
class DiffCalGx(DiffCalRaw, DiffCalRawParameters):
    """
    Basin-Adaptive gx version of DiffCal.

    The only difference from DiffCalRaw:
    - gx is no longer a constant 1, but learned by BasinVarianceEstimator from (X, Y_obs)
    """
    model_type: str = "diffusion_gx"

    def _init_model(self):
        """Initialize model, adding BasinVarianceEstimator"""
        # Call parent's _init_model
        super()._init_model()

        # Added: BasinVarianceEstimator
        self.basin_var_estimator = BasinVarianceEstimator(
            enc_in=self.dataset.num_features,
            hidden_dim=64,
            output_bias=1.0
        ).to(self.device)

        # Added: store gx values per epoch (for analysis)
        # gx_history list: each element is a (531,) array
        # When saving, transpose to (531, epochs), i.e., each row is a basin, each column is an epoch
        self.gx_history = []

    def _init_optimizer(self):
        """Initialize optimizer, including BasinVarianceEstimator parameters"""
        from torch_timeseries.utils.parse_type import parse_type
        from torch.amp import GradScaler

        self.model_optim = parse_type(self.optm_type, globals=globals())(
            [
                {'params': self.model.parameters()},
                {'params': self.cond_pred_model.parameters()},
                {'params': self.basin_var_estimator.parameters()},  # Added
            ],
            lr=self.lr,
        )
        self.grad_scaler = GradScaler('cuda')

    def _process_train_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, is_masked=None):
        """
        Process training batch using basin-adaptive gx.

        The only difference from parent: gx is computed by BasinVarianceEstimator.
        """
        from src.layer.nsdiff_utils import q_sample, cal_sigma_tilde, cal_forward_noise

        batch_y_target = batch_y

        # ===== Core change: use BasinVarianceEstimator to compute gx =====
        gx_scale = self.basin_var_estimator(batch_x)  # (B, 1, 1)
        gx = gx_scale.expand_as(batch_y_target) + EPS  # (B, T, 1)
        y_sigma = gx.clone()
        # ========================================================

        batch_y_mark_input = torch.concat([batch_x_mark[:, -self.label_len:, :], batch_y_mark], dim=1)

        dec_inp_pred = torch.zeros([batch_x.size(0), self.pred_len, self.dataset.num_features]).to(self.device)
        dec_inp_label = batch_x[:, -self.label_len:, :].to(self.device)
        dec_inp = torch.cat([dec_inp_label, dec_inp_pred], dim=1)

        n = batch_x.size(0)
        t = torch.randint(low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

        y_0_hat_batch, _ = self.cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark_input)

        # Per-sample loss for loss1
        loss1_per_sample = (y_0_hat_batch - batch_y_target).square().mean(dim=(1, 2))

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

        # Per-sample kl_loss
        kl_loss_mse = ((e - output)).square().mean(dim=(1, 2))
        kl_loss_sigma1 = (sigma_tilde / sigma_theta).mean(dim=(1, 2))
        kl_loss_sigma2 = torch.log(sigma_tilde / sigma_theta).mean(dim=(1, 2))
        kl_loss_per_sample = kl_loss_mse + kl_loss_sigma1 - kl_loss_sigma2

        total_loss_per_sample = kl_loss_per_sample + loss1_per_sample

        # Loss masking
        if is_masked is not None and is_masked.any():
            loss_mask = ~is_masked
            n_unmasked = loss_mask.sum()
            if n_unmasked > 0:
                loss = total_loss_per_sample[loss_mask].mean()
            else:
                loss = total_loss_per_sample.mean() * 0.0
        else:
            loss = total_loss_per_sample.mean()

        return loss

    def _process_val_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """
        Process validation batch using basin-adaptive gx.

        The only difference from parent: gx is computed by BasinVarianceEstimator.
        """
        from src.layer.nsdiff_utils import p_sample_loop

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

        y_0_hat_batch, _ = self.cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark_input)

        # ===== Core change: use BasinVarianceEstimator to compute gx =====
        gx_scale = self.basin_var_estimator(batch_x)  # (B, 1, 1)
        gx = gx_scale.expand(batch_x.size(0), self.pred_len, 1) + EPS
        # ========================================================

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

    def _compute_all_basin_gx(self):
        """
        Compute gx values for all basins (for analysis)

        Returns:
            gx_values: (n_basins,) numpy array
        """
        import numpy as np

        self.basin_var_estimator.eval()
        all_gx = []

        with torch.no_grad():
            # Iterate through the first batch of train_loader (containing all 531 basins)
            for batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, is_masked in self.train_loader:
                batch_x = batch_x.to(self.device).float()
                gx_scale = self.basin_var_estimator(batch_x)  # (B, 1, 1)
                all_gx.append(gx_scale.squeeze().cpu().numpy())
                break  # Only need the first batch

        self.basin_var_estimator.train()

        if len(all_gx) > 0:
            return all_gx[0]  # (531,)
        return np.array([])

    def _save_run_check_point(self, seed):
        """Save checkpoint, including basin_var_estimator and gx_history"""
        import os
        import numpy as np

        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
        print(f"Saving run checkpoint to '{self.run_save_dir}'.")

        # Compute and save gx values for current epoch
        current_gx = self._compute_all_basin_gx()
        self.gx_history.append(current_gx)
        print(f"  Epoch {self.current_epoch}: gx mean={current_gx.mean():.4f}, std={current_gx.std():.4f}, min={current_gx.min():.4f}, max={current_gx.max():.4f}")

        self.run_state = {
            "model": self.model.state_dict(),
            "cond_pred_model": self.cond_pred_model.state_dict(),
            "basin_var_estimator": self.basin_var_estimator.state_dict(),
            "current_epoch": self.current_epoch,
            "optimizer": self.model_optim.state_dict(),
            "rng_state": torch.get_rng_state(),
            "early_stopping": self.early_stopper.get_state(),
            "y_global_mean": self.y_global_mean,
            "y_global_std": self.y_global_std,
            "gx_history": self.gx_history,  # Added: save gx history
        }

        torch.save(self.run_state, f"{self.run_checkpoint_filepath}")

        # Also save gx_history as numpy file (convenient for analysis)
        # Transpose to (num_basins, epochs) shape, i.e., each row is a basin, each column is an epoch
        gx_history_path = os.path.join(self.run_save_dir, "gx_history.npy")
        gx_array = np.array(self.gx_history).T  # (epochs, 531) -> (531, epochs)
        np.save(gx_history_path, gx_array)
        print(f"  gx_history saved to {gx_history_path} (shape: {gx_array.shape})")

        print("Run state saved ... ")

    def _load_best_model(self):
        """Load best model, including basin_var_estimator"""
        import os

        self.model.load_state_dict(torch.load(self.best_checkpoint_filepath, map_location=self.device))
        self.cond_pred_model.load_state_dict(torch.load(self.best_cond_checkpoint_filepath, map_location=self.device))

        # Try to load basin_var_estimator
        basin_var_path = os.path.join(self.run_save_dir, "basin_var_estimator.pth")
        if os.path.exists(basin_var_path):
            self.basin_var_estimator.load_state_dict(torch.load(basin_var_path, map_location=self.device))

    def _resume_run(self, seed):
        """Resume training, including basin_var_estimator"""
        check_point = torch.load(self.run_checkpoint_filepath, map_location=self.device)
        self.model.load_state_dict(check_point["model"])
        self.cond_pred_model.load_state_dict(check_point["cond_pred_model"])
        if "basin_var_estimator" in check_point:
            self.basin_var_estimator.load_state_dict(check_point["basin_var_estimator"])
        self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]
        self.early_stopper.set_state(check_point["early_stopping"])


if __name__ == "__main__":
    from src.utils.parse_args import parse_masked_basin_ids
    import fire

    globals()['_MASKED_BASIN_IDS'] = parse_masked_basin_ids()

    fire.Fire(DiffCalGx)
