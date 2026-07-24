"""
TimeDiff for CAMELS Dataset - Calibration Mode

TimeDiff core design:
- Conditional diffusion model + LSTM encoder
- Supports multiple samplers (DPM, DDPM, etc.)
- Time series specific noise scheduling

Adapted for CAMELS calibration task:
- Input: [X(42-dim), Y_history(1-dim)] = 43-dim condition
- Output: Y(1-dim) diffusion target
- Calibration mode: window == pred_len == 365
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
from torch.amp import autocast, GradScaler  # BF16 mixed precision training
from tqdm import tqdm
import concurrent.futures
from types import SimpleNamespace

# Import CAMELS dataset and loader
from src.datasets.camels_dataset import CAMELS, CAMELSNpzDataset
from src.dataloader.camels_loader import CAMELSLoader


@dataclass
class TimeDiffParameters:
    beta_start: float = 0.0001
    beta_end: float = 0.5
    num_steps: int = 100
    vis_ar_part: int = 0
    num_samples: int = 100  # Average multiple samples, consistent with original version
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
class TimeDiffCAMELS(ProbForecastExp, TimeDiffParameters):
    model_type: str = "TimeDiff_CAMELS"
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
            batch_size=None,  # Auto-use n_segs=531
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
        Initialize TimeDiff model

        TimeDiff features:
        - LSTM encodes historical sequence
        - Conditional diffusion generates predictions
        - Supports DPM/DDPM sampling

        CAMELS adaptation:
        - num_vars = 1 (only process Y)
        - seq_len = pred_len = 365 (calibration mode)
        """
        self.label_len = self.pred_len // 2

        args_dict = {
            "seq_len": self.windows,
            "device": self.device,
            "pred_len": self.pred_len,
            "label_len": self.label_len,
            "features": 'MS',
            "vis_ar_part": self.vis_ar_part,
            "vis_MTS_analysis": self.vis_MTS_analysis,
            "num_vars": 1,  # CAMELS: only process Y (1-dim)
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
        """Train one epoch - BF16 mixed precision"""
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
                is_masked,  # Add is_masked support
            ) in enumerate(self.train_loader):
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                self.model_optim.zero_grad()

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

    def _process_train_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        """
        Train batch processing

        TimeDiff forward:
        - batch_x: (B, T, N) historical input
        - batch_x_date_enc: time encoding
        - batch_y: (B, O, N) prediction target
        - batch_y_date_enc: target time encoding

        CAMELS:
        - Use batch_y as both input and target (calibration mode)
        """
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
        """Generate predictions on training and test sets and save

        Note: Training set predictions require a shuffle=False loader,
        to ensure prediction order matches the original npz file, otherwise postprocess alignment will be incorrect.
        """
        print("Generating and saving predictions...")

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(project_root, 'output')
        pred_dir = os.path.join(output_dir, 'pred')
        os.makedirs(pred_dir, exist_ok=True)

        # Create a shuffle=False loader for training set (ensure order matches npz file)
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
    fire.Fire(TimeDiffCAMELS)
