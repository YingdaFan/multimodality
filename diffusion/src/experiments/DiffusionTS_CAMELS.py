"""
DiffusionTS for CAMELS Dataset - Conditional Generation Mode

DiffusionTS conditional generation version:
- Uses X as condition to generate Y
- Encoder encodes the condition X
- Decoder generates Y from condition via CrossAttention
- Supports K-fold cross-validation: excludes masked basins during training, pure generation during inference

Adapted for CAMELS calibration task:
- Conditional input: X (33-dimensional meteorological forcing data)
- Generation target: Y (1-dimensional streamflow)
- Training: learns p(Y|X)
- Inference: given X, generates Y from noise
"""

from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os

import torch
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.DiffusionTS_cond import Diffusion_TS_Cond  # Use conditional generation version
from src.experiments.prob_forecast import ProbForecastExp
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
import torch.multiprocessing as mp

from ema_pytorch import EMA
import numpy as np
import torch.distributed as dist
import torch
from torch.amp import autocast, GradScaler  # BF16 mixed precision training
from tqdm import tqdm
import concurrent.futures
from types import SimpleNamespace

# Import CAMELS dataset and loader
from src.datasets.camels_dataset import CAMELS, CAMELSNpzDataset
from src.dataloader.camels_loader import CAMELSLoader


@dataclass
class DiffusionTSParameters:
    num_samples: int = 100  # Multiple samples averaged, consistent with original version
    n_layer_enc: int = 3
    n_layer_dec: int = 6
    d_model: int = 64
    timesteps: int = 100
    sampling_timesteps: int = 100
    loss_type: str = 'l1'
    beta_schedule: str = 'cosine'
    n_heads: int = 4
    mlp_hidden_times: int = 4
    eta: float = 0.
    attn_pd: float = 0.
    resid_pd: float = 0.
    kernel_size: int = 1
    padding_size: int = 0
    use_ff: bool = True
    decay: float = 0.995
    update_interval: int = 10
    reg_weight: float = None
    npz_path: str = '../data_processing/data/prepped.npz'


@dataclass
class DiffusionTSCAMELS(ProbForecastExp, DiffusionTSParameters):
    model_type: str = "DiffusionTS_CAMELS"
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
            batch_size=None,  # Automatically uses n_segs=531
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
        Initialize DiffusionTS conditional generation model

        Conditional generation mode:
        - Condition X: 33-dimensional meteorological forcing data
        - Target Y: 1-dimensional streamflow
        - Encoder encodes condition X
        - Decoder generates Y via CrossAttention
        """
        # Conditional generation model
        # Note: batch_x shape is (B, T, num_features), where num_features = X feature count + 1 (Y_history)
        # Condition dimension = num_features - 1 (excluding Y_history)
        cond_size = self.dataset.num_features - 1

        self.model = Diffusion_TS_Cond(
            seq_length=self.windows,  # 365
            feature_size=1,           # Target Y (1-dimensional)
            cond_size=cond_size,      # Condition X dimension (dynamically obtained from data)
            n_layer_enc=self.n_layer_enc,
            n_layer_dec=self.n_layer_dec,
            d_model=self.d_model,
            timesteps=self.timesteps,
            sampling_timesteps=self.sampling_timesteps,
            loss_type='l2',
            beta_schedule=self.beta_schedule,
            n_heads=self.n_heads,
            mlp_hidden_times=self.mlp_hidden_times,
            eta=self.eta,
            attn_pd=self.attn_pd,
            resid_pd=self.resid_pd,
            kernel_size=self.kernel_size,
            padding_size=self.padding_size,
            use_ff=self.use_ff,
            reg_weight=self.reg_weight,
        )
        self.model = self.model.to(self.device)

    def _init_grad_scaler(self):
        """Initialize BF16 GradScaler"""
        if not hasattr(self, 'grad_scaler'):
            self.grad_scaler = GradScaler('cuda')

    def _train(self):
        """Train one epoch - BF16 mixed precision"""
        self._init_grad_scaler()
        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            self.model.train()
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
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                # BF16 mixed precision training
                with autocast('cuda', dtype=torch.bfloat16):
                    loss = self._process_train_batch(
                        batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                    )

                # Skip NaN/Inf loss to prevent weight corruption
                if torch.isnan(loss) or torch.isinf(loss):
                    self.model_optim.zero_grad()
                    progress_bar.update(batch_x.size(0))
                    continue

                self.grad_scaler.scale(loss).backward()

                # Gradient clipping to prevent gradient explosion
                self.grad_scaler.unscale_(self.model_optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)

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

            return train_loss

    def _process_train_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        """
        Training batch processing

        Data format:
        - batch_x: (B, T, cond_size+1) = [X(cond_size-dimensional features), Y_history(1-dimensional)]
        - batch_y: (B, T, 1)  diffusion target = Y

        Conditional generation:
        - condition: batch_x[:, :, :-1] i.e., X features
        - target: batch_y i.e., Y
        """
        # Extract condition X (all features except the last 1-dimensional Y_history)
        condition = batch_x[:, :, :-1]  # (B, T, cond_size)

        # Target Y
        target = batch_y  # (B, T, 1)

        # Conditional diffusion training: learn p(Y|X)
        loss = self.model(target, condition=condition)
        return loss

    def _process_val_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        """
        Validation batch processing - conditional generation mode

        Given condition X, generate Y from pure noise.
        Does not rely on any historical Y observations, suitable for K-fold cross-validation.

        Returns: preds (B, T, 1, S), batch_y (B, T, 1)
        """
        # Extract condition X (all features except the last 1-dimensional Y_history)
        condition = batch_x[:, :, :-1]  # (B, T, cond_size)

        # ===== Conditional generation: given X, generate Y from noise =====
        with torch.no_grad():
            sample = self.model.fast_sample(
                shape=batch_y.shape,
                condition=condition,
                clip_denoised=False
            )

        # reshape: (B, T, 1) -> (B, T, 1, 1) to maintain interface consistency
        sample = sample.unsqueeze(-1)  # (B, T, 1, S=1)

        return sample, batch_y

    def _evaluate(self, dataloader):
        """Evaluation function"""
        self.model.eval()
        self.metrics.reset()

        with tqdm(total=len(dataloader.dataset)) as progress_bar:
            for batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, is_masked in dataloader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                preds, truths = self._process_val_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )

                # Always denormalize for probabilistic metrics (real-scale evaluation)
                preds = self.scaler.inverse_transform(preds)
                truths = origin_y.to(self.device)

                self.metrics.update(
                    preds.contiguous().cpu().detach(),
                    truths.contiguous().cpu().detach()
                )
                progress_bar.update(batch_x.shape[0])

        result = {name: float(metric.compute()) for name, metric in self.metrics.items()}
        return result

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

            self.current_epoch = self.current_epoch + 1
            self.early_stopper(np.mean(train_losses), model=self.model)

            self._save_run_check_point(seed)


        self._load_best_model()
        best_test_result = self._test()

        # Save prediction results
        self._save()

        return best_test_result

    def _predict(self, loader, desc="Predicting"):
        """Generate predictions on the specified DataLoader"""
        self.model.eval()

        all_preds = []
        all_truths = []

        with torch.no_grad():
            for batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, is_masked in tqdm(loader, desc=desc):
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                preds, truths = self._process_val_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )

                pred_mean = preds.mean(dim=-1)  # (B, T, 1)

                all_preds.append(pred_mean.cpu().numpy())
                all_truths.append(truths.cpu().numpy())

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
    fire.Fire(DiffusionTSCAMELS)
