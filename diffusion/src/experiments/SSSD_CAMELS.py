"""
SSSD for CAMELS Dataset - Calibration Mode

SSSD (Structured State Space Diffusion) core design:
- Uses S4 (Structured State Space) layers to handle long sequences
- Mask mechanism distinguishes observed vs. to-be-predicted portions
- Diffusion process only operates on missing portions

Adapted for CAMELS calibration task:
- Input: [X(42-dim), Y_history(1-dim)] = 43-dim condition (actually only Y is used)
- Output: Y(1-dim) diffusion target
- Mask mechanism: gt_mask specifies prediction target, observation_mask specifies condition
"""

from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os

import torch
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.SSSD import SSSDSAImputer, calc_diffusion_hyperparams
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

# Import CAMELS dataset and loader
from src.datasets.camels_dataset import CAMELS, CAMELSNpzDataset
from src.dataloader.camels_loader import CAMELSLoader


def std_normal(size):
    """Generate standard Gaussian variable"""
    return torch.normal(0, 1, size=size).cuda()


@dataclass
class SSSDParameters:
    beta_start: float = 0.0001
    beta_end: float = 0.02
    num_steps: int = 200
    num_samples: int = 10  # Reduce sample count to avoid OOM during inference
    d_model: int = 128
    n_layers: int = 6
    pool: List[int] = field(default_factory=lambda: [2, 2])
    expand: int = 2
    ff: int = 2
    glu: bool = True
    unet: bool = True
    dropout: float = 0.02
    diffusion_step_embed_dim_in: int = 128
    diffusion_step_embed_dim_mid: int = 512
    diffusion_step_embed_dim_out: int = 512
    label_embed_dim: int = 128
    label_embed_classes: int = 71
    bidirectional: bool = True
    s4_lmax: int = 1000
    s4_d_state: int = 64
    s4_dropout: float = 0.00
    only_generate_missing: int = 1
    s4_bidirectional: bool = True
    npz_path: str = '../data_processing/data/prepped.npz'


@dataclass
class SSSDCAMELS(ProbForecastExp, SSSDParameters):
    model_type: str = "SSSD_CAMELS"
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
        Initialize SSSD model

        SSSD characteristics:
        - S4 layers handle long-range dependencies
        - Input format: (B, C, L) where C is feature count, L is sequence length
        - Mask mechanism: gt_mask (target), observation_mask (condition)
        - Requires sequence length divisible by pool product (pool=[2,2] -> must be divisible by 4)

        CAMELS adaptation:
        - in_channels = 1 (only process Y)
        - out_channels = 1
        - If windows is not divisible by 4, truncate to the nearest multiple of 4
        """
        # SSSD requires sequence length divisible by pool product
        pool_product = 1
        for p in self.pool:
            pool_product *= p
        self._sssd_seq_len = (self.windows // pool_product) * pool_product
        if self._sssd_seq_len != self.windows:
            print(f"  SSSD: Truncating seq_len from {self.windows} to {self._sssd_seq_len} (must be divisible by {pool_product})")

        self.model = SSSDSAImputer(
            d_model=self.d_model,
            n_layers=self.n_layers,
            pool=self.pool,
            expand=self.expand,
            ff=self.ff,
            glu=self.glu,
            unet=self.unet,
            dropout=self.dropout,
            in_channels=1,  # Only process Y (1-dim)
            out_channels=1,
            diffusion_step_embed_dim_in=self.diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid=self.diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out=self.diffusion_step_embed_dim_out,
            label_embed_dim=self.label_embed_dim,
            label_embed_classes=self.label_embed_classes,
            bidirectional=self.bidirectional,
            s4_lmax=self.s4_lmax,
            s4_d_state=self.s4_d_state,
            s4_dropout=self.s4_dropout,
            s4_bidirectional=self.s4_bidirectional,
        )
        self.model = self.model.to(self.device)

        # Diffusion hyperparameters
        self.diffu_params = calc_diffusion_hyperparams(
            self.num_steps, self.beta_start, self.beta_end
        )

        # SSSD mask design (calibration mode)
        # gt_mask = True: this is the prediction target
        # observation_mask = False: this is not an observed condition
        # For calibration task: the entire Y sequence is the prediction target
        # Use truncated sequence length
        self.gt_mask = torch.ones(size=(self._sssd_seq_len, 1)).to(self.device).bool()
        self.observation_mask = ~self.gt_mask

    def _process_train_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        """
        Train batch processing

        SSSD input format:
        - audio: (B, C, L) raw data
        - cond: (B, C, L) condition data
        - mask: (B, C, L) condition mask (1=observed)
        - loss_mask: (B, C, L) loss mask (1=to be predicted)

        CAMELS:
        - batch_y: (B, T, 1) -> transpose -> (B, 1, T)
        - If T is not divisible by pool product, truncate the sequence
        """
        # Truncate sequence length to meet SSSD requirements
        if batch_y.shape[1] != self._sssd_seq_len:
            batch_y = batch_y[:, :self._sssd_seq_len, :]

        B = batch_y.shape[0]

        # SSSD requires (B, C, L) format
        # batch_y: (B, T, 1) -> (B, 1, T)
        audio = batch_y.transpose(1, 2).float()  # (B, 1, T)
        cond = audio.clone()  # Condition data
        # SSSD mask convention: mask=1 means observed (keep original value), mask=0 means to be predicted (use noise)
        # Calibration task: entire sequence needs prediction, so mask=0 (observation_mask)
        # loss_mask: compute loss at prediction positions, i.e., where gt_mask=1
        mask = self.observation_mask.transpose(0, 1).unsqueeze(0).expand(B, -1, -1)  # (B, 1, T) all zeros
        loss_mask = self.gt_mask.transpose(0, 1).unsqueeze(0).expand(B, -1, -1)  # (B, 1, T) all ones

        # SSSD forward
        T = self.num_steps
        Alpha_bar = self.diffu_params["Alpha_bar"].to(self.device)

        C, L = audio.shape[1], audio.shape[2]  # C=1, L=365
        diffusion_steps = torch.randint(T, size=(B, 1, 1)).to(self.device).long()

        z = std_normal(audio.shape).to(self.device)
        if self.only_generate_missing == 1:
            z = audio * mask.float() + z * (1 - mask.float())

        transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * audio + \
                        torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z

        epsilon_theta = self.model((transformed_X, cond, mask, diffusion_steps.view(B, 1)))

        # Return predicted and true noise for loss computation
        return epsilon_theta[loss_mask], z[loss_mask]

    def _process_val_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        """
        Validation batch processing

        Returns: preds (B, T, 1, S), batch_y (B, T, 1)
        Note: if original sequence length is not divisible by pool product, the returned length is truncated
        """
        # Truncate sequence length to meet SSSD requirements
        if batch_y.shape[1] != self._sssd_seq_len:
            batch_y = batch_y[:, :self._sssd_seq_len, :]

        B = batch_y.shape[0]

        # SSSD format
        audio = batch_y.transpose(1, 2).float()  # (B, 1, T)
        cond = audio.clone()
        # SSSD mask convention: mask=1 means observed, mask=0 means to be predicted
        # Calibration task: mask=0 (predict all)
        mask = self.observation_mask.transpose(0, 1).unsqueeze(0).expand(B, -1, -1)  # all zeros

        T = self.num_steps
        Alpha = self.diffu_params["Alpha"]
        Alpha_bar = self.diffu_params["Alpha_bar"]
        Sigma = self.diffu_params["Sigma"]

        # Expand to multiple samples
        size = (B * self.num_samples, audio.shape[1], audio.shape[2])

        X_expanded = [
            audio.repeat(self.num_samples, 1, 1),
            cond.repeat(self.num_samples, 1, 1),
            mask.repeat(self.num_samples, 1, 1),
        ]

        cond_exp = X_expanded[1]
        mask_exp = X_expanded[2]

        # Start sampling from noise
        x = std_normal(size).to(self.device)

        with torch.no_grad():
            for t in range(T - 1, -1, -1):
                if self.only_generate_missing == 1:
                    x = x * (1 - mask_exp.float()) + cond_exp * mask_exp.float()
                diffusion_steps = (t * torch.ones((size[0], 1))).to(self.device)
                epsilon_theta = self.model((x, cond_exp, mask_exp, diffusion_steps))

                # Update x (denoising)
                x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / \
                    torch.sqrt(Alpha[t])
                # Deterministic sampling: no random noise added, suitable for point prediction tasks
                # Original code (probabilistic sampling):
                # if t > 0:
                #     x = x + Sigma[t] * std_normal(size).to(self.device)

        # reshape: (B*S, 1, T) -> (B, S, 1, T) -> (B, T, 1, S)
        x = x.reshape(B, self.num_samples, 1, self._sssd_seq_len)  # (B, S, 1, T)
        x = x.permute(0, 3, 2, 1)  # (B, T, 1, S)

        return x, batch_y

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
                is_masked,
            ) in enumerate(self.train_loader):
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                # BF16 mixed precision training
                with autocast('cuda', dtype=torch.bfloat16):
                    pred_noise, true_noise = self._process_train_batch(
                        batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                    )
                    # SSSD loss: MSE between predicted and true noise
                    loss = ((pred_noise - true_noise) ** 2).mean()

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

        # SSSD truncates sequence from 365 to 364 (must be divisible by pool product 4)
        # Need to pad predictions back to 365 to match observations in the npz file
        target_seq_len = self.windows  # Original sequence length (365)

        for partition, loader in [('trn', train_loader_no_shuffle)]:  # ('tst', self.test_loader)]:
            preds, truths = self._predict(loader, desc=f"Predicting [{partition}]")

            # If prediction length < target length, pad with last valid value (nearest neighbor interpolation)
            if preds.shape[1] < target_seq_len:
                pad_len = target_seq_len - preds.shape[1]
                # preds shape: (N, T, 1) -> pad subsequent days with the value from day T-1
                last_values = preds[:, -1:, :]  # (N, 1, 1)
                pad_values = np.repeat(last_values, pad_len, axis=1)  # (N, pad_len, 1)
                preds = np.concatenate([preds, pad_values], axis=1)
                print(f"  Padded predictions from {preds.shape[1] - pad_len} to {preds.shape[1]} (filled {pad_len} day(s) with last value)")

            pred_path = os.path.join(pred_dir, f'{partition}.npy')
            np.save(pred_path, preds)
            print(f"[{partition}] Predictions saved: {pred_path}, shape: {preds.shape}")


if __name__ == "__main__":
    from src.utils.parse_args import parse_masked_basin_ids
    import fire

    globals()['_MASKED_BASIN_IDS'] = parse_masked_basin_ids()
    fire.Fire(SSSDCAMELS)
