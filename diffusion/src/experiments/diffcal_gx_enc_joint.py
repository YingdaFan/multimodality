"""
Joint LSTM + Diffusion End-to-End Fine-tuning (Pure Encoder Backbone)

Inherits from diffcal_gx_enc.py (DiffCalRaw) and adds:
1. Upstream LSTM loaded into the computation graph
2. LSTM output replaces static y_obs in batch_x on every forward pass
3. Diffusion loss backpropagates through denormalization into LSTM weights
4. Separate learning rates: LSTM (small) vs Diffusion (normal)

Differentiable chain (all linear ops, gradients flow directly):
    LSTM(X_42) → denorm (per-basin) → renorm (global) → batch_x[:,:,-1]

Usage:
    python diffcal_gx_enc_joint.py runs --seeds='[1]'
"""

import sys
import os
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time

# Add LSTM model to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_LSTM_DIR = os.path.join(os.path.dirname(_PROJECT_ROOT), 'lstm')
sys.path.insert(0, _LSTM_DIR)
from model import LSTM

from src.experiments.diffcal_gx_enc import (
    DiffCalRaw as DiffCalRawEnc, DiffCalEarlyStopping, EPS
)
from src.experiments.prob_forecast import update_metrics


class JointEarlyStopping(DiffCalEarlyStopping):
    """Early stopping that also saves LSTM weights."""

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model['model'].state_dict(), os.path.join(self.path, 'model.pth'))
        torch.save(model['cond_pred_model'].state_dict(), os.path.join(self.path, 'cond_pred_model.pth'))
        if model.get('lstm_model') is not None:
            torch.save(model['lstm_model'].state_dict(), os.path.join(self.path, 'lstm_model.pth'))
        self.val_loss_min = val_loss


@dataclass
class DiffCalJoint(DiffCalRawEnc):
    """Joint LSTM + Diffusion training with end-to-end gradient flow (Pure Encoder)."""

    model_type: str = "diffusion_gx_enc_joint"

    # LSTM config
    lstm_weights_path: str = '../lstm/output/finetuned_weights.pth'
    lstm_hidden_dim: int = 20
    lstm_dropout: float = 0.2
    lstm_lr: float = 1e-5  # Small LR to preserve pre-trained features

    # --- Model init ---

    def _init_model(self):
        super()._init_model()

        n_x_features = self.dataset.num_features - 3  # 45 - 3 = 42
        self.lstm_model = LSTM(
            input_dim=n_x_features,
            hidden_dim=self.lstm_hidden_dim,
            dropout=self.lstm_dropout,
            device=str(self.device),
            seed=42
        ).to(self.device)

        if os.path.exists(self.lstm_weights_path):
            state = torch.load(self.lstm_weights_path, map_location=self.device, weights_only=True)
            self.lstm_model.load_state_dict(state)
            print(f"[Joint] Loaded pre-trained LSTM from {self.lstm_weights_path}")
        else:
            print(f"[Joint] WARNING: LSTM weights not found at {self.lstm_weights_path}")

        lstm_params = sum(p.numel() for p in self.lstm_model.parameters())
        print(f"[Joint] LSTM: input={n_x_features}, hidden={self.lstm_hidden_dim}, params={lstm_params}")
        print(f"[Joint] LR: diffusion={self.lr}, lstm={self.lstm_lr}")

    # --- Optimizer with separate LR for LSTM ---

    def _init_optimizer(self):
        self.model_optim = torch.optim.Adam(
            [
                {'params': self.model.parameters(), 'lr': self.lr},
                {'params': self.cond_pred_model.parameters(), 'lr': self.lr},
                {'params': self.lstm_model.parameters(), 'lr': self.lstm_lr},
            ],
            lr=self.lr,
        )
        self.grad_scaler = GradScaler('cuda')

    # --- Core: replace static y_obs with live LSTM output ---

    def _replace_y_obs_with_lstm(self, batch_x):
        """
        Replace static y_obs in batch_x with live LSTM forward pass.

        batch_x layout: [X(42), y_mean_vae(1), y_std_vae(1), y_obs(1)]
        The LSTM takes X(42) and outputs per-basin normalized predictions.
        We denormalize (per-basin) then re-normalize (global) to match y_obs format.
        """
        x_features = batch_x[:, :, :-3]       # (B, T, 42)
        y_mean_vae = batch_x[:, :, -3:-2]      # (B, T, 1)
        y_std_vae = batch_x[:, :, -2:-1]       # (B, T, 1)

        # LSTM forward (gradients flow through here)
        lstm_out = self.lstm_model(x_features)  # (B, T, 1)

        # Per-basin denormalization → global re-normalization
        y_denorm = lstm_out * (y_std_vae + 1e-10) + y_mean_vae
        y_obs_global = (y_denorm - self.y_global_mean) / (self.y_global_std + 1e-10)

        return torch.cat([x_features, y_mean_vae, y_std_vae, y_obs_global], dim=-1)

    # --- Training loop ---

    def _train(self):
        self.model.train()
        self.cond_pred_model.train()
        self.lstm_model.train()

        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            train_loss = []
            for i, (
                batch_x, batch_y, origin_x, origin_y,
                batch_x_date_enc, batch_y_date_enc, is_masked,
            ) in enumerate(self.train_loader):
                origin_y = origin_y.to(self.device).float()
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                is_masked = is_masked.to(self.device).squeeze(-1)

                with autocast('cuda', dtype=torch.bfloat16):
                    batch_x = self._replace_y_obs_with_lstm(batch_x)
                    loss = self._process_train_batch(
                        batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, is_masked
                    )

                if torch.isnan(loss) or torch.isinf(loss):
                    self.model_optim.zero_grad()
                    progress_bar.update(batch_x.size(0))
                    continue

                self.grad_scaler.scale(loss).backward()
                progress_bar.update(batch_x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    diff_lr=self.model_optim.param_groups[0]["lr"],
                    lstm_lr=self.model_optim.param_groups[2]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )
                self.grad_scaler.step(self.model_optim)
                self.grad_scaler.update()
                self.model_optim.zero_grad()

        self.model.eval()
        self.cond_pred_model.eval()
        self.lstm_model.eval()
        return train_loss

    # --- Prediction with live LSTM ---

    def _predict(self, loader, desc="Predicting"):
        self.model.eval()
        self.cond_pred_model.eval()
        self.lstm_model.eval()

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

                batch_x = self._replace_y_obs_with_lstm(batch_x)

                preds, truths = self._process_val_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )

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

    # --- Checkpoint management ---

    def _setup_early_stopper(self):
        self.best_checkpoint_filepath = os.path.join(self.run_save_dir, "model.pth")
        self.best_cond_checkpoint_filepath = os.path.join(self.run_save_dir, "cond_pred_model.pth")
        self.best_cond_g_checkpoint_filepath = os.path.join(self.run_save_dir, "cond_pred_model_g.pth")
        self.best_lstm_checkpoint_filepath = os.path.join(self.run_save_dir, "lstm_model.pth")
        self.early_stopper = JointEarlyStopping(self.patience, verbose=True, path=self.run_save_dir)

    def _save_run_check_point(self, seed):
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
        print(f"Saving run checkpoint to '{self.run_save_dir}'.")

        self.run_state = {
            "model": self.model.state_dict(),
            "cond_pred_model": self.cond_pred_model.state_dict(),
            "lstm_model": self.lstm_model.state_dict(),
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
        if os.path.exists(self.best_lstm_checkpoint_filepath):
            self.lstm_model.load_state_dict(torch.load(self.best_lstm_checkpoint_filepath, map_location=self.device))

    def _resume_run(self, seed):
        check_point = torch.load(self.run_checkpoint_filepath, map_location=self.device)
        self.model.load_state_dict(check_point["model"])
        self.cond_pred_model.load_state_dict(check_point["cond_pred_model"])
        if "lstm_model" in check_point:
            self.lstm_model.load_state_dict(check_point["lstm_model"])
        self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]
        self.early_stopper.set_state(check_point["early_stopping"])

    # --- Main run loop ---

    def run(self, seed=42) -> Dict[str, float]:
        self._setup_run(seed)
        self._check_run_exist(seed)

        self._run_print(f"[Joint] run : {self.current_run} in seed: {seed}")

        parameter_tables, model_parameters_num = count_parameters(self.model)
        self._run_print(f"parameter_tables: {parameter_tables}")
        self._run_print(f"diffusion parameters: {model_parameters_num}")
        lstm_params = sum(p.numel() for p in self.lstm_model.parameters())
        self._run_print(f"lstm parameters: {lstm_params}")

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

            self.current_epoch = self.current_epoch + 1
            self.early_stopper(
                np.mean(train_losses),
                model={
                    'model': self.model,
                    'cond_pred_model': self.cond_pred_model,
                    'cond_pred_model_g': self.cond_pred_model_g,
                    'lstm_model': self.lstm_model,
                },
            )
            self._save_run_check_point(seed)

        self._load_best_model()
        best_test_result = {}
        self._save()

        return best_test_result


if __name__ == "__main__":
    from src.utils.parse_args import parse_masked_basin_ids
    import fire

    globals()['_MASKED_BASIN_IDS'] = parse_masked_basin_ids()

    fire.Fire(DiffCalJoint)
