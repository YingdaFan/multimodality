"""
TimeDiff_X for CAMELS Dataset - X -> Y Mode (Degraded Y History Approach)

Differences from TimeDiff_CAMELS.py:
- For masked basins, Y_history is set to 0 (handled by camels_dataset.py)
- Loss computation for masked basins is skipped during training
- TimeDiff learns to generate Y from "zero Y history", applicable to ungauged basin scenarios

Note: This is not a pure X -> Y approach (the model still receives 43-dim input),
but it is effective for ungauged basin inference since the model has learned to generate from Y=0.

Usage:
    bash scripts/CAMELS/run_camels_kfold.sh TimeDiff_X
"""

from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os

import torch
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.TimeDiff import TimeDiff
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
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import concurrent.futures
from types import SimpleNamespace

# Import CAMELS dataset and loader (using original version, supports mask)
from src.datasets.camels_dataset import CAMELS, CAMELSNpzDataset
from src.dataloader.camels_loader import CAMELSLoader


@dataclass
class TimeDiffParameters:
    beta_start: float = 0.0001
    beta_end: float = 0.5
    num_steps: int = 100
    vis_ar_part: int = 0
    num_samples: int = 100
    vis_MTS_analysis: int = 1
    schedule: str = "quad"
    use_window_normalization: bool = True
    t0: float = 1e-4
    T: float = 1
    nfe: int = 100
    dim_LSTM: int = 64
    UNet_Type: str = 'CNN'
    D3PM_kernel_size: int = 5
    use_freq_enhance: int = 0
    type_sampler: str = 'dpm'
    parameterization: str = 'x_start'
    ddpm_inp_embed: int = 256
    ddpm_dim_diff_steps: int = 256
    ddpm_channels_conv: int = 256
    ddpm_channels_fusion_I: int = 256
    ddpm_layers_inp: int = 5
    ddpm_layers_I: int = 5
    ddpm_layers_II: int = 5
    cond_ddpm_num_layers: int = 5
    cond_ddpm_channels_conv: int = 64
    ablation_study_case: str = "none"
    weight_pred_loss: float = 0.0
    ablation_study_F_type: str = "CNN"
    ablation_study_masking_type: str = "none"
    ablation_study_masking_tau: float = 0.9
    ot_ode: bool = True
    npz_path: str = '../data_processing/data/prepped.npz'


@dataclass
class TimeDiff_X_CAMELS(ProbForecastExp, TimeDiffParameters):
    """
    TimeDiff_X: X -> Y Mode (Degraded Y History Approach)

    Differences from TimeDiffCAMELS:
    - For masked basins, Y_history = 0
    - Loss for masked basins is skipped during training
    - Uses mask_mode='zero' to ensure Y_history = 0
    """
    model_type: str = "TimeDiff_X_CAMELS"
    dataset_type: str = "CAMELS"

    def _init_dataset(self):
        """Initialize CAMELS dataset"""
        self.dataset = CAMELS(npz_path=self.npz_path)

    def _init_data_loader(self, shuffle=False, fast_test=True, fast_val=True):
        """Use CAMELSLoader with mask_mode='zero' to ensure Y_history = 0 for masked basins"""
        self._init_dataset()

        from torch_timeseries.scaler import StandardScaler
        self.scaler = StandardScaler()

        # Get masked_basin_ids from global variables
        masked_basins_list = globals().get('_MASKED_BASIN_IDS', None)
        if masked_basins_list:
            print(f"🎯 [TimeDiff_X] Masked basins (Y_history=0): {masked_basins_list}")

        # Use mask_mode='zero' to ensure Y_history = 0 for masked basins
        self.dataloader = CAMELSLoader(
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
            masked_basins=masked_basins_list,
            mask_mode='zero'  # Key: set Y_history to 0 for masked basins
        )

        self.train_loader, self.val_loader, self.test_loader = (
            self.dataloader.train_loader,
            self.dataloader.val_loader,
            self.dataloader.test_loader,
        )

    def _init_model(self):
        """Initialize TimeDiff model"""
        self.label_len = self.pred_len // 2

        args_dict = {
            "seq_len": self.windows,
            "device": self.device,
            "pred_len": self.pred_len,
            "label_len": self.label_len,
            "features": 'MS',
            "vis_ar_part": self.vis_ar_part,
            "vis_MTS_analysis": self.vis_MTS_analysis,
            "num_vars": 1,  # Only process Y (1-dim)
            "freq": self.dataset.freq,
            "interval": self.num_steps,
            "beta-max": self.beta_end,
            "use_window_normalization": self.use_window_normalization,
            "t0": self.t0,
            "T": self.T,
            "nfe": self.nfe,
            "dim_LSTM": self.dim_LSTM,
            "diff_steps": self.num_steps,
            "UNet_Type": self.UNet_Type,
            "D3PM_kernel_size": self.D3PM_kernel_size,
            "use_freq_enhance": self.use_freq_enhance,
            "type_sampler": self.type_sampler,
            "parameterization": self.parameterization,
            "ddpm_inp_embed": self.ddpm_inp_embed,
            "ddpm_dim_diff_steps": self.ddpm_dim_diff_steps,
            "ddpm_channels_conv": self.ddpm_channels_conv,
            "ddpm_channels_fusion_I": self.ddpm_channels_fusion_I,
            "ddpm_layers_inp": self.ddpm_layers_inp,
            "ddpm_layers_I": self.ddpm_layers_I,
            "ddpm_layers_II": self.ddpm_layers_II,
            "cond_ddpm_num_layers": self.cond_ddpm_num_layers,
            "cond_ddpm_channels_conv": self.cond_ddpm_channels_conv,
            "ablation_study_case": self.ablation_study_case,
            "weight_pred_loss": self.weight_pred_loss,
            "ablation_study_F_type": self.ablation_study_F_type,
            "ablation_study_masking_type": self.ablation_study_masking_type,
            "ablation_study_masking_tau": self.ablation_study_masking_tau,
            "ot-ode": self.ot_ode,
        }

        args = SimpleNamespace(**args_dict)
        self.model = TimeDiff(args)
        self.model = self.model.to(self.device)

    def _init_grad_scaler(self):
        """Initialize BF16 GradScaler"""
        if not hasattr(self, 'grad_scaler'):
            self.grad_scaler = GradScaler('cuda')

    def _train(self):
        """Train one epoch - supports masked basins"""
        self._init_grad_scaler()
        self.model.train()

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
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                is_masked = is_masked.to(self.device).squeeze(-1)  # (B,)

                self.model_optim.zero_grad()

                # BF16 mixed precision training
                with autocast('cuda', dtype=torch.bfloat16):
                    loss = self._process_train_batch(
                        batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, is_masked
                    )

                # If all samples are masked, skip this batch
                if loss is None:
                    progress_bar.update(batch_x.size(0))
                    continue

                # Skip NaN/Inf loss to prevent weight corruption
                if torch.isnan(loss) or torch.isinf(loss):
                    self.model_optim.zero_grad()
                    progress_bar.update(batch_x.size(0))
                    continue

                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.model_optim)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

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

        self.model.eval()
        return train_loss

    def _process_train_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, is_masked=None):
        """
        Training batch processing - X -> Y mode (Degraded Y History)

        For masked basins:
        - The Y_history portion of batch_x is already 0 (handled by the dataloader)
        - Loss computation is skipped for these basins

        TimeDiff forward:
        - Uses batch_y as both input and target (calibration mode)
        - For masked basins, since Y_history=0, the model learns to generate from zero
        """
        # Handle masked basins: only compute loss for non-masked basins
        if is_masked is not None:
            non_masked_idx = (is_masked == 0)
            if non_masked_idx.sum() == 0:
                # All samples are masked, skip this batch
                return None
            # Only keep non-masked samples
            batch_x = batch_x[non_masked_idx]
            batch_y = batch_y[non_masked_idx]
            batch_x_date_enc = batch_x_date_enc[non_masked_idx]
            batch_y_date_enc = batch_y_date_enc[non_masked_idx]

        # TimeDiff train_forward: input history, predict target
        # For the calibration task: input and target are the same Y sequence
        loss = self.model.train_forward(batch_y, batch_x_date_enc, batch_y, batch_y_date_enc)
        return loss

    def _process_val_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        """
        Validation batch processing

        Returns: preds (B, T, 1, S), batch_y (B, T, 1)
        """
        # TimeDiff forward returns multiple samples
        outs, x, y, _, _ = self.model(
            batch_y, batch_x_date_enc, batch_y, batch_y_date_enc,
            None, None, None, self.num_samples
        )

        # outs shape: (B, S, T, N) -> (B, T, N, S)
        outs = outs.permute(0, 2, 3, 1)

        return outs, batch_y

    def run(self, seed=42) -> Dict[str, float]:

        self._setup_run(seed)
        self._check_run_exist(seed)

        self._run_print(f"run : {self.current_run} in seed: {seed}")
        self._run_print(f"🎯 X → Y mode: masked basins have Y_history=0")

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
        """Generate and save predictions on training and test sets"""
        print("Generating and saving predictions...")

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(project_root, 'output')
        pred_dir = os.path.join(output_dir, 'pred')
        os.makedirs(pred_dir, exist_ok=True)

        # Create a shuffle=False loader for the training set
        from torch.utils.data import DataLoader
        train_loader_no_shuffle = DataLoader(
            self.dataloader.train_dataset,
            batch_size=self.dataloader.batch_size,
            shuffle=False,
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
    fire.Fire(TimeDiff_X_CAMELS)
