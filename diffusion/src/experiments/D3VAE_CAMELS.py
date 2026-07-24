"""
D3VAE for CAMELS Dataset - Calibration Mode

D3VAE (Denoising Diffusion Variational AutoEncoder) core design principles:
- VAE + Diffusion hybrid architecture
- Encoder-Decoder structure for time series data
- Diffusion process operates in latent space

Adapted for CAMELS calibration task:
- Input: [X(42 dims), Y_history(1 dim)] = 43 dims conditional
- Output: Y(1 dim) diffusion target
- Calibration mode: window == pred_len == 365
"""

from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os

import torch
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.D3VAE import denoise_net
from src.experiments.prob_forecast import ProbForecastExp
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
from types import SimpleNamespace

import torch.multiprocessing as mp
from torch_timeseries.utils.parse_type import parse_type

import numpy as np
import torch.distributed as dist
import torch
from tqdm import tqdm
import concurrent.futures

# Import CAMELS dataset and loader
from src.datasets.camels_dataset import CAMELS, CAMELSNpzDataset
from src.dataloader.camels_loader import CAMELSLoader


@dataclass
class D3VAEParameters:
    # Parameters optimized for CAMELS seq_len=365, reducing GPU memory usage
    embedding_dimension: int = 32  # Originally 64, halved
    dropout_rate: float = 0.1
    beta_schedule: str = 'linear'
    beta_start: float = 0.0
    beta_end: float = 0.01
    diff_step: int = 100
    scale: float = 0.1
    mult: int = 1
    channel_mult: int = 2
    num_preprocess_blocks: int = 1
    num_preprocess_cells: int = 3
    arch_instance: str = 'res_mbconv'
    num_latent_per_group: int = 4  # Originally 8, halved
    num_channels_enc: int = 16  # Originally 32, halved
    num_channels_dec: int = 16  # Originally 32, halved
    num_postprocess_blocks: int = 1
    num_postprocess_cells: int = 2
    hidden_size: int = 64  # Originally 128, halved
    num_layers: int = 2
    groups_per_scale: int = 2
    psi: float = 0.5
    lambda1: float = 1
    gamma: float = 0.01
    num_samples: int = 100
    npz_path: str = '../data_processing/data/prepped.npz'


@dataclass
class D3VAECAMELS(ProbForecastExp, D3VAEParameters):
    model_type: str = "D3VAE_CAMELS"
    dataset_type: str = "CAMELS"
    batch_size: int = 400  # D3VAE has high GPU memory usage, maximizing ~48GB GPU memory
    windows: int = 365  # CAMELS calibration mode: 365-day input
    pred_len: int = 365  # CAMELS calibration mode: 365-day output
    use_amp: bool = True  # Use mixed precision training (BF16)

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
            batch_size=None,  # Automatically uses n_segs=531, preserving spatial relationships
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
        Initialize D3VAE model

        D3VAE characteristics:
        - VAE + Diffusion hybrid
        - Requires seq_len == pred_len (calibration mode satisfies this)
        - Requires even sequence length (internal convolution stride=2 requirement)
        - Input dimension = target_dim

        CAMELS adaptation:
        - input_dim = 1 (only processes Y)
        - target_dim = 1
        - sequence_length = prediction_length = 364 (truncated by 1 day to satisfy even length requirement)
        """
        # D3VAE requires even sequence length, subtract 1 if odd
        seq_len = self.windows if self.windows % 2 == 0 else self.windows - 1
        self._d3vae_seq_len = seq_len  # Save the actual sequence length used

        args = {
            "arch_instance": self.arch_instance,
            "beta_end": self.beta_end,
            "beta_schedule": self.beta_schedule,
            "beta_start": self.beta_start,
            "channel_mult": self.channel_mult,
            "detail_freq": self.dataset.freq,
            "diff_steps": self.diff_step,
            "dropout_rate": self.dropout_rate,
            "embedding_dimension": self.embedding_dimension,
            "features": 'MS',
            "freq": self.dataset.freq,
            "gamma": self.gamma,
            "groups_per_scale": self.groups_per_scale,
            "hidden_size": self.hidden_size,
            "input_dim": 1,  # CAMELS: only processes Y (1-dim)
            "inverse": False,
            "lambda1": self.lambda1,
            "loss_type": 'kl',
            "mult": self.mult,
            "num_channels_dec": self.num_channels_dec,
            "num_channels_enc": self.num_channels_enc,
            "num_latent_per_group": self.num_latent_per_group,
            "num_layers": self.num_layers,
            "num_postprocess_blocks": self.num_postprocess_blocks,
            "num_postprocess_cells": self.num_postprocess_cells,
            "num_preprocess_blocks": self.num_preprocess_blocks,
            "num_preprocess_cells": self.num_preprocess_cells,
            "prediction_length": seq_len,
            "psi": self.psi,
            "scale": self.scale,
            "sequence_length": seq_len,
            "target_dim": 1,  # CAMELS: only predicts Y (1-dim)
        }
        self.args = SimpleNamespace(**args)
        self.model = denoise_net(self.args)
        self.model = self.model.to(self.device)

    def _process_train_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        """
        Training batch processing

        D3VAE characteristics:
        - Requires even batch size
        - Requires even sequence length (internal convolution stride=2 requirement)

        CAMELS:
        - batch_y: (B, T, 1) as both input and target
        """
        b, seq_len, _ = batch_y.shape

        # D3VAE requires even batch
        if b % 2 != 0:
            batch_y = torch.cat([batch_y, batch_y[0:1, :, :]], dim=0)
            batch_x_date_enc = torch.cat([batch_x_date_enc, batch_x_date_enc[0:1, :, :]], dim=0)

        # D3VAE requires even sequence length, truncate last timestep if odd
        if seq_len % 2 != 0:
            batch_y = batch_y[:, :-1, :]  # (B, T-1, 1)
            batch_x_date_enc = batch_x_date_enc[:, :-1, :]  # (B, T-1, 4)

        # Random diffusion timestep
        t = torch.randint(0, self.diff_step, (batch_y.shape[0],), dtype=torch.int64, device=self.device)

        # D3VAE forward: input x and target y are the same (calibration mode)
        # Use full 4-dim time encoding [month, day, weekday, hour]
        output, y_noisy, total_c, loss = self.model(batch_y, batch_x_date_enc, batch_y, t)
        return output, y_noisy, total_c, loss

    def _process_val_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        """
        Validation batch processing

        Returns: preds (B, T, 1, S), batch_y (B, T, 1)
        Note: if original sequence length is odd, returns the truncated even length
        """
        b, seq_len, _ = batch_y.shape

        # D3VAE requires even sequence length, truncate last timestep if odd
        if seq_len % 2 != 0:
            batch_y = batch_y[:, :-1, :]  # (B, T-1, 1)
            batch_x_date_enc = batch_x_date_enc[:, :-1, :]  # (B, T-1, 4)

        B = batch_y.shape[0]
        # mini_sample: number of samples per iteration, validation GPU usage = batch_size x mini_sample
        # D3VAE's prob_pred generation is very memory-intensive, needs small mini_sample
        mini_sample = 10  # Original value

        outs = []
        for i in range(self.num_samples // mini_sample):
            repeat_batch_y = batch_y.repeat(mini_sample, 1, 1)
            repeat_time_enc = batch_x_date_enc.repeat(mini_sample, 1, 1)

            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        noisy_out, out, _ = self.model.prob_pred(repeat_batch_y, repeat_time_enc)
                else:
                    noisy_out, out, _ = self.model.prob_pred(repeat_batch_y, repeat_time_enc)

            out = out.reshape(B, mini_sample, out.shape[-2], out.shape[-1])
            outs.append(out.detach().cpu())

        outs = torch.concat(outs, dim=1)  # (B, S, T, 1)
        outs = outs.permute(0, 2, 3, 1)  # (B, T, 1, S)

        return outs, batch_y

    def _train(self):
        """Train one epoch, with BF16 mixed precision support"""
        self.model.train()

        # Initialize GradScaler (only when using AMP)
        scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            train_loss = []
            for i, (
                batch_x,
                batch_y,
                origin_x,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
            ) in enumerate(self.train_loader):
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                if self.use_amp:
                    # Use BF16 mixed precision
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        output, y_noisy, total_c, loss = self._process_train_batch(
                            batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                        )
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.model_optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                    scaler.step(self.model_optim)
                    scaler.update()
                    self.model_optim.zero_grad()
                else:
                    output, y_noisy, total_c, loss = self._process_train_batch(
                        batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                    self.model_optim.step()
                    self.model_optim.zero_grad()

                progress_bar.update(batch_x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )

        self.model.eval()
        return train_loss

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
            for batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc in tqdm(loader, desc=desc):
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
        to ensure prediction order matches the original npz file, otherwise postprocessing alignment will be incorrect.
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
    fire.Fire(D3VAECAMELS)
