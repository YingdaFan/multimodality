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
from torch.optim import *

# Inherits from the Raw version
from imputation.diffusion.src.experiments.diffcal_raw_res import DiffCalRaw, DiffCalRawParameters

# Added BasinVarianceEstimator
from src.models.basin_variance import BasinVarianceEstimator


EPS = 10e-8


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

        Key changes:
        1. gx is computed by BasinVarianceEstimator
        2. y_0_hat directly uses LSTM prediction (y_obs), skipping cond_pred_model
        """
        from src.layer.nsdiff_utils import q_sample, cal_sigma_tilde, cal_forward_noise

        batch_y_target = batch_y

        # ===== Core change 1: use BasinVarianceEstimator to compute gx =====
        gx_scale = self.basin_var_estimator(batch_x)  # (B, 1, 1)
        gx = gx_scale.expand_as(batch_y_target) + EPS  # (B, T, 1)
        y_sigma = gx.clone()
        # ========================================================

        n = batch_x.size(0)
        t = torch.randint(low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

        # ===== Core change 2: directly use y_obs (LSTM prediction) as y_0_hat =====
        # batch_x structure: [X_features, y_mean_vae, y_std_vae, y_obs], y_obs is the last dimension
        y_0_hat_batch = batch_x[:, :, -1:].clone()  # (B, T, 1) - extract y_obs
        # loss1 removed since y_0_hat is now fixed (LSTM prediction, no gradients)
        # ============================================================

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

        # Per-sample kl_loss (loss1 removed)
        kl_loss_mse = ((e - output)).square().mean(dim=(1, 2))
        kl_loss_sigma1 = (sigma_tilde / sigma_theta).mean(dim=(1, 2))
        kl_loss_sigma2 = torch.log(sigma_tilde / sigma_theta).mean(dim=(1, 2))
        kl_loss_per_sample = kl_loss_mse + kl_loss_sigma1 - kl_loss_sigma2

        total_loss_per_sample = kl_loss_per_sample  # loss1 removed

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

        Key changes:
        1. gx is computed by BasinVarianceEstimator
        2. y_0_hat directly uses LSTM prediction (y_obs), skipping cond_pred_model
        """
        from src.layer.nsdiff_utils import p_sample_loop

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

        # ===== Core change 1: directly use y_obs (LSTM prediction) as y_0_hat =====
        # batch_x structure: [X_features, y_mean_vae, y_std_vae, y_obs], y_obs is the last dimension
        y_0_hat_batch = batch_x[:, :, -1:].clone()  # (B, T, 1) - extract y_obs
        # ============================================================

        # ===== Core change 2: use BasinVarianceEstimator to compute gx =====
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

    def _save_run_check_point(self, seed):
        """Save checkpoint, including basin_var_estimator"""
        import os

        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
        print(f"Saving run checkpoint to '{self.run_save_dir}'.")

        self.run_state = {
            "model": self.model.state_dict(),
            "cond_pred_model": self.cond_pred_model.state_dict(),
            "basin_var_estimator": self.basin_var_estimator.state_dict(),  # Added
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
