"""
CSDI for CAMELS Dataset - Calibration Mode (XEncoder version)

CSDI (Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation)
Core design principles:
- Conditional diffusion: uses cond_mask to distinguish condition (X) and target (Y)
- Dual-channel input: [conditional observed values, noisy target] concatenated and fed into the diffusion model

Adapted for CAMELS calibration task (XEncoder conditional injection):
- Input: X(32-dim) + Y(1-dim) = 33-dimensional features
- XEncoder: compresses X(32-dim) into a low-dimensional embedding of cond_dim(16-dim)
- Network input: compressed X(16-dim) + Y(1-dim) = 17-dim
- Output: Y(1-dim) diffusion target
- Conditional generation: X serves as condition (cond_mask=1, kept clean), Y is generated from noise (cond_mask=0)

This design prevents the model from "seeing" the real Y during training, ensuring training-inference consistency
"""

from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os

import torch
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.CSDI import CSDI_Forecasting
from src.experiments.prob_forecast import ProbForecastExp
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
import torch.multiprocessing as mp

import numpy as np
import torch.distributed as dist
import torch
from torch.amp import autocast, GradScaler  # BF16 mixed precision training
from tqdm import tqdm
import concurrent.futures
from torch_timeseries.utils.early_stop import EarlyStopping

# Import CAMELS dataset and loader
from src.datasets.camels_dataset import CAMELS, CAMELSNpzDataset
from src.dataloader.camels_loader import CAMELSLoader


class CSDIEarlyStopping(EarlyStopping):
    """CSDI-specific EarlyStopping, saves both model and x_encoder"""
    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model['csdi_model'].state_dict(), os.path.join(self.path, 'csdi_model.pth'))
        if 'x_encoder' in model:
            torch.save(model['x_encoder'].state_dict(), os.path.join(self.path, 'x_encoder.pth'))
        self.val_loss_min = val_loss


class XEncoder(torch.nn.Module):
    """X feature encoder - compresses high-dimensional X into a low-dimensional condition vector

    Encodes X (B, T, x_dim) into (B, T, cond_dim), then passes it as condition to the diffusion model.
    This avoids directly feeding all X features into the Transformer, which would cause memory explosion.
    """
    def __init__(self, x_dim, cond_dim, hidden_dim=64):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(x_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, cond_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, x_dim) raw X features
        Returns:
            (B, T, cond_dim) compressed condition vector
        """
        return self.encoder(x)


@dataclass
class CSDIParameters:
    layers: int = 4
    channels: int = 64
    nheads: int = 8
    diffusion_embedding_dim: int = 128
    beta_start: float = 0.0001
    beta_end: float = 0.5
    num_steps: int = 50
    schedule: str = "quad"
    is_linear: bool = True
    is_unconditional: int = 0
    timeemb: int = 128
    featureemb: int = 16
    num_samples: int = 10  # Reduced sampling count to save memory
    target_strategy: str = "test"
    num_sample_features: int = 1  # Only predict Y (1-dim)
    npz_path: str = '../data_processing/data/prepped.npz'
    # X condition encoder configuration
    cond_dim: int = 16  # Compressed dimension of X features (original 32-dim -> 16-dim)
    cond_hidden_dim: int = 64  # XEncoder hidden layer dimension


@dataclass
class CSDICAMELS(ProbForecastExp, CSDIParameters):
    model_type: str = "CSDI_CAMELS"
    dataset_type: str = "CAMELS"

    def _init_dataset(self):
        """Initialize CAMELS dataset"""
        self.dataset = CAMELS(npz_path=self.npz_path)

    def _init_data_loader(self, shuffle=False, fast_test=True, fast_val=True):
        """Use CAMELSLoader"""
        self._init_dataset()

        from torch_timeseries.scaler import StandardScaler
        self.scaler = StandardScaler()

        # Get masked_basin_ids from global variables
        masked_basins_list = globals().get('_MASKED_BASIN_IDS', None)
        if masked_basins_list:
            print(f"Masked basins (by ID): {masked_basins_list}")

        self.dataloader = CAMELSLoader(
            dataset=self.dataset,
            scaler=self.scaler,
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len,
            shuffle_train=shuffle,
            freq=self.dataset.freq,
            batch_size=64,  # Reduced batch_size to avoid OOM (originally 531)
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
        """
        Initialize CSDI model (XEncoder conditional injection version - following CSBI design)

        CAMELS adaptation (same design approach as CSBI):
        - X features (33-dim) are compressed to cond_dim (16-dim) via XEncoder
        - target_dim = cond_dim + 1 = 17 (compressed X + Y)
        - observed_data = [compressed X, Y]
        - cond_mask: X dimensions = 1 (condition, kept clean), Y dimension = 0 (target, noise added)

        This design allows the model to learn to predict Y from the X condition, rather than from Y itself.
        """
        # X feature dimension (num_features includes Y, so X dimension is num_features - 1)
        self.num_features = self.dataset.num_features  # 34 (33 X + 1 Y)
        self.x_dim = self.num_features - 1  # 33 (X only)
        print(f"X dim: {self.x_dim}, cond_dim: {self.cond_dim}")

        # Create X feature encoder
        self.x_encoder = XEncoder(
            x_dim=self.x_dim,
            cond_dim=self.cond_dim,
            hidden_dim=self.cond_hidden_dim
        ).to(self.device)

        # target_dim = compressed X (cond_dim) + Y (1)
        total_dim = self.cond_dim + 1  # 17

        configs = {
            "diffusion": {
                "layers": self.layers,
                "channels": self.channels,
                "nheads": self.nheads,
                "diffusion_embedding_dim": self.diffusion_embedding_dim,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "num_steps": self.num_steps,
                "schedule": self.schedule,
                "is_linear": self.is_linear,
            },
            "model": {
                "is_unconditional": self.is_unconditional,
                "timeemb": self.timeemb,
                "featureemb": self.featureemb,
                "target_strategy": self.target_strategy,
                "num_sample_features": total_dim  # 17-dim
            }
        }

        # CAMELS: target_dim = cond_dim + 1 (compressed X + Y)
        self.model = CSDI_Forecasting(
            config=configs,
            device=self.device,
            target_dim=total_dim
        )
        self.model = self.model.to(self.device)

    def _process_train_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        """
        Training batch processing (XEncoder conditional injection version - following CSBI design)

        Core idea (same as CSBI):
        - observed_data = [compressed X, Y] (K = cond_dim + 1 = 17)
        - cond_mask: X dimensions = 1 (condition, kept clean), Y dimension = 0 (target, noise added)
        - CSDI adds noise to target positions (Y), condition positions (X) are kept clean

        CAMELS data format:
        - batch_x: (B, T, 34) conditional input = [X(33-dim features), Y(1-dim)]
        - batch_y: (B, T, 1)  diffusion target = Y
        """
        B, T = batch_x.shape[:2]

        # Separate X and Y features
        x_raw = batch_x[:, :, :-1]  # (B, T, 33) - X features
        y_raw = batch_x[:, :, -1:]  # (B, T, 1) - Y features

        # Compress X features via XEncoder: (B, T, 33) -> (B, T, cond_dim=16)
        x_encoded = self.x_encoder(x_raw)  # (B, T, cond_dim)

        # Concatenate compressed X and Y: (B, T, cond_dim+1=17)
        combined = torch.cat([x_encoded, y_raw], dim=2)  # (B, T, 17)

        # Convert to CSDI expected format: (B, T, K) -> (B, K, L)
        K = self.cond_dim + 1  # 17
        observed_data = combined.permute(0, 2, 1)  # (B, K, L)

        # cond_mask: X dimensions = 1 (condition), Y dimension = 0 (target)
        cond_mask = torch.ones(B, K, T, device=self.device).float()
        cond_mask[:, -1, :] = 0  # Last dimension (Y) is the target, needs to be generated from noise

        # observed_mask: all positions are "observed"
        observed_mask = torch.ones(B, K, T, device=self.device).float()

        # Time points
        observed_tp = batch_x_date_enc[:, :, 0]  # (B, T)

        # Use CSDI internal method (bypass mask generation logic in forward)
        side_info = self.model.get_side_info(observed_tp, cond_mask)
        noise, pred_noise = self.model.train_forward(observed_data, cond_mask, observed_mask, side_info)

        return noise, pred_noise

    def _process_val_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        """
        Validation batch processing (XEncoder conditional injection version - following CSBI design)

        Core idea (same as _process_train_batch):
        - observed_data = [compressed X, Y] (K = cond_dim + 1 = 17)
        - cond_mask: X dimensions = 1 (condition, kept clean), Y dimension = 0 (target, generated from noise)
        - Uses model.impute() for sampling

        Returns: preds (B, T, 1, S), batch_y (B, T, 1)
        """
        B, T = batch_x.shape[:2]

        # Separate X and Y features
        x_raw = batch_x[:, :, :-1]  # (B, T, 33) - X features
        y_raw = batch_x[:, :, -1:]  # (B, T, 1) - Y features

        # Compress X features via XEncoder: (B, T, 33) -> (B, T, cond_dim=16)
        with torch.no_grad():
            x_encoded = self.x_encoder(x_raw)  # (B, T, cond_dim)

        # Concatenate compressed X and Y: (B, T, cond_dim+1=17)
        combined = torch.cat([x_encoded, y_raw], dim=2)  # (B, T, 17)

        # Convert to CSDI expected format: (B, T, K) -> (B, K, L)
        K = self.cond_dim + 1  # 17
        observed_data = combined.permute(0, 2, 1)  # (B, K, L)

        # cond_mask: X dimensions = 1 (condition), Y dimension = 0 (target)
        cond_mask = torch.ones(B, K, T, device=self.device).float()
        cond_mask[:, -1, :] = 0  # Last dimension (Y) is the target, needs to be generated from noise

        # Time points
        observed_tp = batch_x_date_enc[:, :, 0]  # (B, T)

        # Use CSDI internal method for sampling
        with torch.no_grad():
            side_info = self.model.get_side_info(observed_tp, cond_mask)
            samples = self.model.impute(observed_data, cond_mask, side_info, self.num_samples)

        # samples shape: (B, S, K=17, L) -> extract only Y (last dimension)
        y_samples = samples[:, :, -1:, :]  # (B, S, 1, L)
        y_samples = y_samples.permute(0, 3, 2, 1)  # (B, L, 1, S) = (B, T, 1, S)

        return y_samples, batch_y

    def _init_optimizer(self):
        """Initialize optimizer - includes both CSDI model and XEncoder"""
        from torch.optim.adam import Adam
        self.model_optim = Adam(
            list(self.model.parameters()) + list(self.x_encoder.parameters()),
            lr=self.lr
        )

    def _setup_early_stopper(self):
        """Set up EarlyStopping - uses custom CSDIEarlyStopping"""
        self.best_checkpoint_filepath = os.path.join(self.run_save_dir, "model.pth")
        self.early_stopper = CSDIEarlyStopping(
            self.patience, verbose=True, path=self.run_save_dir
        )

    def _load_best_model(self):
        """Load best model - includes both CSDI model and XEncoder"""
        self.model.load_state_dict(
            torch.load(os.path.join(self.run_save_dir, 'csdi_model.pth'), map_location=self.device)
        )
        # Load XEncoder
        x_encoder_path = os.path.join(self.run_save_dir, 'x_encoder.pth')
        if os.path.exists(x_encoder_path):
            self.x_encoder.load_state_dict(
                torch.load(x_encoder_path, map_location=self.device)
            )

    def _save_run_check_point(self, seed):
        """Save run checkpoint - includes XEncoder"""
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)

        print(f"Saving run checkpoint to '{self.run_save_dir}'.")

        self.run_state = {
            "model": self.model.state_dict(),
            "x_encoder": self.x_encoder.state_dict(),
            "current_epoch": self.current_epoch,
            "optimizer": self.model_optim.state_dict(),
            "rng_state": torch.get_rng_state(),
            "early_stopping": self.early_stopper.get_state(),
        }

        torch.save(self.run_state, f"{self.run_checkpoint_filepath}")
        print("Run state saved ... ")

    def _init_grad_scaler(self):
        """Initialize BF16 GradScaler"""
        if not hasattr(self, 'grad_scaler'):
            self.grad_scaler = GradScaler('cuda')

    def _train(self):
        """Train one epoch - BF16 mixed precision (XEncoder version)"""
        self._init_grad_scaler()
        self.model.train()
        self.x_encoder.train()  # XEncoder also needs training

        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            train_loss = []
            for i, (
                batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
                is_masked,
            ) in enumerate(self.train_loader):
                self.model_optim.zero_grad()

                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                # BF16 mixed precision training
                with autocast('cuda', dtype=torch.bfloat16):
                    noise, pred_noise = self._process_train_batch(
                        batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                    )
                    # CSDI loss: MSE between noise and predicted noise
                    loss = ((noise - pred_noise) ** 2).mean()

                # Skip NaN/Inf loss to prevent weight corruption
                if torch.isnan(loss) or torch.isinf(loss):
                    self.model_optim.zero_grad()
                    progress_bar.update(batch_x.size(0))
                    continue

                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.model_optim)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.x_encoder.parameters()), max_norm=1.0
                )
                self.grad_scaler.step(self.model_optim)
                self.grad_scaler.update()

                progress_bar.update(batch_x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )

        self.model.eval()
        self.x_encoder.eval()
        return train_loss

    def run(self, seed=42) -> Dict[str, float]:

        self._setup_run(seed)
        self._check_run_exist(seed)

        self._run_print(f"run : {self.current_run} in seed: {seed}")

        # Count CSDI model and XEncoder parameters
        csdi_params = sum(p.numel() for p in self.model.parameters())
        x_encoder_params = sum(p.numel() for p in self.x_encoder.parameters())
        total_params = csdi_params + x_encoder_params
        self._run_print(f"CSDI model parameters: {csdi_params}")
        self._run_print(f"XEncoder parameters: {x_encoder_params}")
        self._run_print(f"Total parameters: {total_params}")


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
            # Pass csdi_model and x_encoder to early_stopper
            self.early_stopper(np.mean(train_losses), model={'csdi_model': self.model, 'x_encoder': self.x_encoder})

            self._save_run_check_point(seed)


        self._load_best_model()
        best_test_result = self._test()

        # Save prediction results
        self._save()

        return best_test_result

    def _predict(self, loader, desc="Predicting"):
        """
        Generate predictions on the specified DataLoader (XEncoder version - following CSBI design)

        Core idea (same as _process_train_batch):
        - observed_data = [compressed X, Y] (K = cond_dim + 1 = 17)
        - cond_mask: X dimensions = 1 (condition, kept clean), Y dimension = 0 (target, generated from noise)
        - Uses model.impute() for sampling
        """
        self.model.eval()
        self.x_encoder.eval()

        all_preds = []
        all_truths = []

        with torch.no_grad():
            for batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, is_masked in tqdm(loader, desc=desc):
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                B, T = batch_x.shape[:2]

                # Separate X and Y features
                x_raw = batch_x[:, :, :-1]  # (B, T, 33) - X features
                y_raw = batch_x[:, :, -1:]  # (B, T, 1) - Y features

                # Compress X features via XEncoder: (B, T, 33) -> (B, T, cond_dim=16)
                x_encoded = self.x_encoder(x_raw)  # (B, T, cond_dim)

                # Concatenate compressed X and Y: (B, T, cond_dim+1=17)
                combined = torch.cat([x_encoded, y_raw], dim=2)  # (B, T, 17)

                # Convert to CSDI expected format: (B, T, K) -> (B, K, L)
                K = self.cond_dim + 1  # 17
                observed_data = combined.permute(0, 2, 1)  # (B, K, L)

                # cond_mask: X dimensions = 1 (condition), Y dimension = 0 (target)
                cond_mask = torch.ones(B, K, T, device=self.device).float()
                cond_mask[:, -1, :] = 0  # Last dimension (Y) is the target, needs to be generated from noise

                # Time points
                observed_tp = batch_x_date_enc[:, :, 0]  # (B, T)

                # Use CSDI internal method for sampling
                side_info = self.model.get_side_info(observed_tp, cond_mask)
                samples = self.model.impute(observed_data, cond_mask, side_info, n_samples=1)

                # samples shape: (B, S=1, K=17, L) -> extract only Y (last dimension)
                y_samples = samples[:, :, -1:, :]  # (B, S=1, 1, L)
                y_samples = y_samples.permute(0, 3, 2, 1)  # (B, L, 1, S) = (B, T, 1, S)
                pred_mean = y_samples.mean(dim=-1)  # (B, T, 1)

                all_preds.append(pred_mean.cpu().numpy())
                all_truths.append(batch_y.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_truths = np.concatenate(all_truths, axis=0)
        return all_preds, all_truths

    def _save(self):
        """Generate predictions on train and test sets and save

        Note: Training set predictions must use a shuffle=False loader
        to ensure prediction order matches the original npz file; otherwise postprocess alignment will be incorrect.
        """
        print("Generating and saving predictions...")

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(project_root, 'output')
        pred_dir = os.path.join(output_dir, 'pred')
        os.makedirs(pred_dir, exist_ok=True)

        # Create a shuffle=False loader for the training set (ensure order matches npz file)
        from torch.utils.data import DataLoader
        train_loader_no_shuffle = DataLoader(
            self.dataloader.train_dataset,
            batch_size=self.dataloader.batch_size,
            shuffle=False,  # Key: no shuffle during prediction
            num_workers=self.dataloader.num_worker,
            drop_last=False
        )

        for partition, loader in [('trn', train_loader_no_shuffle)]:  # ('tst', self.test_loader)]:
            preds, truths = self._predict(loader, desc=f"Predicting [{partition}]")

            pred_path = os.path.join(pred_dir, f'{partition}.npy')
            np.save(pred_path, preds)
            print(f"[{partition}] Predictions saved: {pred_path}, shape: {preds.shape}")


if __name__ == "__main__":
    from src.utils.parse_args import parse_masked_basin_ids
    import fire

    globals()['_MASKED_BASIN_IDS'] = parse_masked_basin_ids()
    fire.Fire(CSDICAMELS)
