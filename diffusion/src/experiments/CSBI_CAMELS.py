"""
CSBI for CAMELS Dataset - Calibration Mode (XEncoder version)

CSBI (Conditional Score-Based Imputation) core design principles:
- Score-based diffusion with SDE (Stochastic Differential Equation)
- Bidirectional networks: z_f (forward), z_b (backward)
- Conditional generation: cond_mask distinguishes conditions from targets

Adapted for CAMELS calibration task (XEncoder conditional injection):
- Input: X(32-dim) + Y(1-dim) = 33-dim features
- XEncoder: compresses X(32-dim) into low-dimensional embedding of cond_dim(16-dim)
- Network input: compressed X(16-dim) + Y(1-dim) = 17-dim
- Output: Y(1-dim) diffusion target
- Conditional generation: X as condition (cond_mask=True), Y generated from noise (cond_mask=False)

This design avoids memory explosion (O(K^2)) caused by directly processing 33-dim features in Transformer
"""

import argparse
from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os

import torch
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.nn.csbi_net import build_transformerv5
from src.experiments.prob_forecast import ProbForecastExp, update_metrics
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
import torch.multiprocessing as mp
from torch_timeseries.utils.parse_type import parse_type
from torch_timeseries.utils.early_stop import EarlyStopping
import torch.distributions as td

import numpy as np
import torch.distributed as dist
import torch
from torch.amp import autocast, GradScaler  # BF16 mixed precision training
from tqdm import tqdm
import concurrent.futures
import src.nn.csbi_sde as sde
import src.nn.csbi_policy as policy
import src.nn.csbi_util as util
import yaml
from torch.optim.adam import Adam

# Import CAMELS dataset and loader
from src.datasets.camels_dataset import CAMELS, CAMELSNpzDataset
from src.dataloader.camels_loader import CAMELSLoader


class CSBIEarlyStopping(EarlyStopping):
    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model['z_f'].state_dict(), os.path.join(self.path, 'z_f.pth'))
        torch.save(model['z_b'].state_dict(), os.path.join(self.path, 'z_b.pth'))
        if 'x_encoder' in model:
            torch.save(model['x_encoder'].state_dict(), os.path.join(self.path, 'x_encoder.pth'))
        self.val_loss_min = val_loss


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def freeze_policy(policy):
    for p in policy.parameters():
        p.requires_grad = False
    policy.eval()
    return policy


def activate_policy(policy):
    for p in policy.parameters():
        p.requires_grad = True
    policy.train()
    return policy


class PriorSampler:
    """Gaussian prior distribution sampler"""
    def __init__(self, prior, batch_size, device):
        self.prior = prior
        self.batch_size = batch_size
        self.device = device

    def sample(self, num_samples=None):
        n = num_samples if num_samples is not None else self.batch_size
        return self.prior.sample((n,)).to(self.device)


class XEncoder(torch.nn.Module):
    """X feature encoder - compresses high-dimensional X into low-dimensional condition vector

    Encodes X (B, T, x_dim) into (B, T, cond_dim), then passed as condition to the diffusion model.
    This avoids memory explosion caused by directly feeding all X features into Transformer.
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


class DataSampler:
    """CAMELS data sampler (for CSBI's p distribution)

    Only samples Y features for SDE; X features are processed separately via XEncoder.
    """
    def __init__(self, train_loader, windows, device):
        self.train_loader = train_loader
        self.windows = windows
        self.device = device
        self.iterator = iter(train_loader)

    def sample(self, num_samples=None, return_mask=False, return_all_mask=False):
        try:
            batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.train_loader)
            batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc = next(self.iterator)

        # Only use Y features: (B, T, 1) -> (B, 1, 1, T)
        x0 = batch_y.transpose(1, 2).unsqueeze(1).to(self.device)  # (B, 1, 1, T)
        B = batch_y.shape[0]

        if return_all_mask or return_mask:
            # Y is the only target
            obs_mask = torch.zeros(B, 1, 1, self.windows, device=self.device).bool()
            gt_mask = torch.ones(B, 1, 1, self.windows, device=self.device).bool()

            if return_all_mask:
                return x0.float(), obs_mask, gt_mask
            else:
                return obs_mask, gt_mask
        else:
            return x0.float()


@dataclass
class CSBIParameters:
    num_samples: int = 20  # Sample count (reduced to save memory, was 100)
    interval: int = 100
    beta_max: float = 20.0
    beta_min: float = 0.001
    t0: float = 0.001
    T: float = 1.0
    train_bs_t: int = 2  # Time samples per batch (reduced to save memory, was 5)
    num_hutchinson_samp: int = 1
    use_corrector: bool = False
    backward_net: str = 'Transformerv5'
    forward_net: str = 'Transformerv5'
    sde_type: str = 'vp'  # 'vp' or 've'
    zero_out_last_layer: bool = True
    # Transformerv5 network config (reduced complexity to save memory)
    layers: int = 2
    nheads: int = 4  # Reduced head count (was 8)
    channels: int = 32  # Reduced channel count (was 64)
    diffusion_embedding_dim: int = 32  # Reduced (was 64)
    timeemb: int = 16  # Reduced (was 32)
    featureemb: int = 8  # Reduced (was 16)
    is_linear: bool = False
    npz_path: str = '../data_processing/data/prepped.npz'
    # X condition encoder config
    cond_dim: int = 16  # Compressed dimension for X features (original 32-dim -> 16-dim)
    cond_hidden_dim: int = 64  # XEncoder hidden layer dimension


@dataclass
class CSBICAMELS(ProbForecastExp, CSBIParameters):
    model_type: str = "CSBI_CAMELS"
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
            print(f"🎭 Masked basins (by ID): {masked_basins_list}")

        self.dataloader = CAMELSLoader(
            dataset=self.dataset,
            scaler=self.scaler,
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len,
            shuffle_train=shuffle,
            freq=self.dataset.freq,
            batch_size=59,  # CSBI needs smaller batch_size due to internal (B*L) reshape
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
        Initialize CSBI model

        CSBI characteristics:
        - Bidirectional networks: z_f (forward score), z_b (backward score)
        - SDE-based diffusion
        - Input format: (B, C, K, L) where C=1 or 2 (single/dual channel)

        CAMELS adaptation (X conditional encoding mode):
        - X features (32-dim) compressed via MLP to cond_dim (16-dim)
        - input_size = (cond_dim + 1, 365) = (17, 365)
        - Y (1-dim) as diffusion target
        """
        # Build opt configuration
        self.opt = argparse.Namespace()
        self.opt.device = self.device
        self.opt.interval = self.interval
        self.opt.beta_max = self.beta_max
        self.opt.beta_min = self.beta_min
        self.opt.t0 = self.t0
        self.opt.T = self.T
        self.opt.train_bs_t = self.train_bs_t
        self.opt.num_hutchinson_samp = self.num_hutchinson_samp
        self.opt.use_corrector = self.use_corrector
        self.opt.backward_net = self.backward_net
        self.opt.forward_net = self.forward_net
        self.opt.sde_type = self.sde_type
        self.opt.train_method = 'alternate'  # CSBI training method

        # X feature dimension (num_features includes Y, so X dim is num_features - 1)
        self.num_features = self.dataset.num_features  # 33 (includes X and Y)
        self.x_dim = self.num_features - 1  # 32 (X only)
        print(f"X dim: {self.x_dim}, cond_dim: {self.cond_dim}")

        # Create X feature encoder
        self.x_encoder = XEncoder(
            x_dim=self.x_dim,
            cond_dim=self.cond_dim,
            hidden_dim=self.cond_hidden_dim
        ).to(self.device)

        # Input size settings:
        # - input_size: network input dimension (cond_dim + 1, 365) - compressed X + Y
        # - data_dim: SDE computation dimension (1, 1, 365) - only add noise to Y
        self.opt.input_size = (self.cond_dim + 1, self.windows)  # Compressed X (16) + Y (1) = 17
        self.opt.data_dim = [1, 1, self.windows]  # SDE only operates on Y (1-dim)
        self.opt.problem_name = 'sinusoid'

        # Build timeline
        self.ts = torch.linspace(self.opt.t0, self.opt.T, self.opt.interval).to(self.device)

        # Build boundary distributions p (data) and q (prior)
        self.p = DataSampler(self.train_loader, self.windows, self.device)

        # Build prior distribution q (Gaussian) - only for Y features (1-dim)
        cov_coef = 1.0 if self.sde_type == 'vp' else self.opt.beta_max ** 2
        prior = td.MultivariateNormal(
            torch.zeros(self.windows, device=self.device),
            cov_coef * torch.eye(self.windows, device=self.device)
        )
        self.q = PriorSampler(prior, self.batch_size, self.device)

        # Build SDE dynamics
        self.dyn = sde.build(self.opt, self.p, self.q)

        # Build network configuration
        net_config = argparse.Namespace()
        net_config.input_size = self.opt.input_size
        net_config.layers = self.layers
        net_config.nheads = self.nheads
        net_config.channels = self.channels
        net_config.diffusion_embedding_dim = self.diffusion_embedding_dim
        net_config.timeemb = self.timeemb
        net_config.featureemb = self.featureemb
        net_config.is_linear = self.is_linear

        # Build bidirectional networks
        self.z_f = build_transformerv5(
            net_config,
            self.interval,
            self.zero_out_last_layer,
        ).to(self.device)

        self.z_b = build_transformerv5(
            net_config,
            self.interval,
            self.zero_out_last_layer,
        ).to(self.device)

        self.z_f.direction = 'forward'
        self.z_b.direction = 'backward'

        # CSBI mask design (calibration mode)
        # gt_mask = True: prediction target
        # observation_mask = False: condition (for calibration task, no observed condition)
        self.gt_mask = torch.ones(size=(self.windows, 1)).to(self.device).bool()
        self.observation_mask = ~self.gt_mask

    def _init_optimizer(self):
        """Initialize optimizer - CSBI has two networks + XEncoder"""
        self.model_optim = Adam(
            list(self.z_f.parameters()) + list(self.z_b.parameters()) + list(self.x_encoder.parameters()),
            lr=self.lr
        )

    def _setup_early_stopper(self):
        self.best_checkpoint_filepath = os.path.join(self.run_save_dir, "model.pth")
        self.early_stopper = CSBIEarlyStopping(
            self.patience, verbose=True, path=self.run_save_dir
        )

    def _load_best_model(self):
        self.z_f.load_state_dict(
            torch.load(os.path.join(self.run_save_dir, 'z_f.pth'), map_location=self.device)
        )
        self.z_b.load_state_dict(
            torch.load(os.path.join(self.run_save_dir, 'z_b.pth'), map_location=self.device)
        )
        # Load XEncoder
        x_encoder_path = os.path.join(self.run_save_dir, 'x_encoder.pth')
        if os.path.exists(x_encoder_path):
            self.x_encoder.load_state_dict(
                torch.load(x_encoder_path, map_location=self.device)
            )

    def _save_run_check_point(self, seed):
        """Save run checkpoint - CSBI version (includes XEncoder)"""
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)

        print(f"Saving run checkpoint to '{self.run_save_dir}'.")

        self.run_state = {
            "z_f": self.z_f.state_dict(),
            "z_b": self.z_b.state_dict(),
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
        """Train one epoch

        XEncoder conditional generation mode:
        - Extract X features (32-dim) and Y (1-dim)
        - X compressed via XEncoder to cond_dim (16-dim)
        - Network input: compressed X (16-dim) + Y (1-dim) = 17-dim
        - SDE only adds noise to Y
        """
        self.z_f.eval()
        self.z_b.train()
        self.x_encoder.train()  # XEncoder also needs to be trained

        policy_opt = activate_policy(self.z_b)
        compute_xs_label = sde.get_xs_label_computer(self.opt, self.ts)

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

                batch_x = batch_x.to(self.device).float()  # (B, T, 33) - contains X and Y
                batch_y = batch_y.to(self.device).float()  # (B, T, 1)
                B, T = batch_x.shape[:2]

                # Separate X and Y features
                x_raw = batch_x[:, :, :-1]  # (B, T, 32) - X features
                y_raw = batch_x[:, :, -1:]  # (B, T, 1) - Y features

                # Compress X features via XEncoder: (B, T, 32) -> (B, T, cond_dim=16)
                x_encoded = self.x_encoder(x_raw)  # (B, T, cond_dim)

                # Concatenate compressed X and Y: (B, T, cond_dim+1=17)
                combined = torch.cat([x_encoded, y_raw], dim=2)  # (B, T, 17)

                # Convert to CSBI format: (B, T, K) -> (B, 1, K, T)
                x0 = combined.permute(0, 2, 1).unsqueeze(1)  # (B, 1, 17, T)

                # Condition mask: compressed X (first cond_dim dims) is condition, Y (last 1 dim) is target
                # cond_mask = True: condition positions (keep clean)
                # cond_mask = False: target positions (add noise)
                K = self.cond_dim + 1  # 17
                cond_mask = torch.ones(B, 1, K, self.windows, device=self.device).bool()
                cond_mask[:, :, -1, :] = False  # Last dim (Y) is target, needs to be generated from noise

                # Sample time steps
                batch_t = self.train_bs_t
                samp_t_idx = torch.randint(self.opt.interval, (B, batch_t))
                ts = self.ts[samp_t_idx].detach()
                ts = ts.reshape(B * batch_t)

                # Compute noisy samples and labels (only add noise to Y features)
                y_only = x0[:, :, -1:, :]  # (B, 1, 1, T)
                xs_y, label_y, _ = compute_xs_label(x0=y_only, samp_t_idx=samp_t_idx, return_scale=True)

                # Build complete xs: compressed X stays clean, Y gets noise added
                x_encoded_csbi = x0[:, :, :-1, :]  # (B, 1, cond_dim, T) - compressed X
                x_encoded_expand = x_encoded_csbi.unsqueeze(1).repeat(1, batch_t, 1, 1, 1)
                x_encoded_expand = util.flatten_dim01(x_encoded_expand)  # (B*t, 1, cond_dim, T)
                xs_y = util.flatten_dim01(xs_y)  # (B*t, 1, 1, T)
                xs = torch.cat([x_encoded_expand, xs_y], dim=2)  # (B*t, 1, 17, T)

                # Expand other variables
                x0_expand = x0.unsqueeze(1).repeat(1, batch_t, 1, 1, 1)
                x0_expand = util.flatten_dim01(x0_expand)  # (B*t, 1, 17, T)
                cond_mask_expand = cond_mask.unsqueeze(1).repeat(1, batch_t, 1, 1, 1)
                cond_mask_expand = util.flatten_dim01(cond_mask_expand)  # (B*t, 1, 17, T)

                # Build input: condition positions use clean data, target positions use noisy data
                cond_obs = cond_mask_expand * x0_expand  # Compressed X (clean)
                noisy_target = (~cond_mask_expand) * xs   # Y (noisy)
                total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B*t, 2, 17, T)
                diff_input = (total_input, cond_mask_expand)

                # Prediction
                predicted = policy_opt(diff_input, ts)

                # Labels: only compute loss on Y features
                label_y = util.flatten_dim01(label_y.unsqueeze(1).repeat(1, batch_t, 1, 1, 1) if label_y.dim() == 4 else label_y)
                label_full = torch.zeros_like(predicted)
                label_full[:, :, -1:, :] = label_y.reshape(label_full[:, :, -1:, :].shape)

                # Compute loss (only at Y feature positions)
                target_mask = ~cond_mask_expand  # Y feature positions
                residual = (label_full - predicted) * target_mask
                num_eval = target_mask.sum()
                loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.z_b.parameters()) + list(self.x_encoder.parameters()), 1.0)
                self.model_optim.step()

                progress_bar.update(batch_x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )

        self.z_f.eval()
        self.z_b.eval()
        self.x_encoder.eval()
        return train_loss

    @torch.no_grad()
    def imputation(self, batch_x, num_samples):
        """Conditional sampling (using XEncoder)

        Args:
            batch_x: (B, T, 33) raw input features (X:32-dim + Y:1-dim)
            num_samples: number of samples

        Returns:
            y_samples: (B, num_samples, T) generated Y samples
        """
        self.x_encoder.eval()
        ts_reverse = torch.flip(self.ts, dims=[0])
        K, L = self.opt.input_size  # K=17, L=365
        B, T = batch_x.shape[:2]

        # Separate X and Y features
        x_raw = batch_x[:, :, :-1]  # (B, T, 32)
        y_raw = batch_x[:, :, -1:]  # (B, T, 1)

        # Compress X via XEncoder: (B, T, 32) -> (B, T, cond_dim=16)
        x_encoded = self.x_encoder(x_raw)  # (B, T, cond_dim)

        # Concatenate compressed X and Y: (B, T, 17)
        combined = torch.cat([x_encoded, y_raw], dim=2)

        # Convert to CSBI format: (B, T, K) -> (B, 1, K, T)
        x_cond = combined.permute(0, 2, 1).unsqueeze(1)  # (B, 1, 17, T)

        # Condition mask: compressed X (first cond_dim dims) is condition, Y (last 1 dim) is target
        cond_mask = torch.ones(B, 1, K, L, device=self.device).bool()
        cond_mask[:, :, -1, :] = False  # Y is target

        policy_net = freeze_policy(self.z_b)
        y_samples = torch.zeros(B, num_samples, L).to(self.device)

        for i in range(num_samples):
            # Initialize: condition positions use encoded X, target position (Y) starts from noise
            current_sample = x_cond.clone()
            noise = torch.randn(B, 1, 1, L, device=self.device)
            current_sample[:, :, -1:, :] = noise

            for idx, t in enumerate(ts_reverse):
                # Build conditional input: condition positions use encoded X, target positions use current sample
                cond_obs = cond_mask * x_cond  # Encoded X (clean)
                noisy_target = (~cond_mask) * current_sample  # Y (noisy/current)
                total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B, 2, 17, T)
                diff_input = (total_input, cond_mask)

                # Score prediction
                z = policy_net(diff_input, t)

                # Only update target position (Y)
                g = self.dyn.g(t)
                dt = self.dyn.dt
                dw = torch.randn(B, 1, 1, L, device=self.device) * np.sqrt(dt)

                # Update Y features
                y_update = current_sample[:, :, -1:, :]
                z_y = z[:, :, -1:, :]
                y_update = y_update + g * z_y * dt
                if t > 0:
                    y_update = y_update + g * dw

                # Keep encoded X unchanged, only update Y
                current_sample = x_cond.clone()
                current_sample[:, :, -1:, :] = y_update

            # Extract final Y prediction: (B, 1, 1, T) -> (B, T)
            y_samples[:, i] = current_sample[:, :, -1, :].squeeze(1).detach()

        return y_samples

    def _process_val_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        """Validation batch processing (using XEncoder)"""
        # imputation now directly accepts batch_x (B, T, 33) and returns y_samples (B, S, T)
        y_samples = self.imputation(batch_x, self.num_samples)  # (B, S, T)

        # Convert to validation format: (B, S, T) -> (B, T, 1, S)
        y_samples = y_samples.permute(0, 2, 1).unsqueeze(2)  # (B, T, 1, S)

        return y_samples, batch_y

    def run(self, seed=42) -> Dict[str, float]:

        self._setup_run(seed)
        self._check_run_exist(seed)

        self._run_print(f"run : {self.current_run} in seed: {seed}")

        z_f_params = sum(p.numel() for p in self.z_f.parameters())
        z_b_params = sum(p.numel() for p in self.z_b.parameters())
        total_params = z_f_params + z_b_params
        self._run_print(f"z_f parameters: {z_f_params}")
        self._run_print(f"z_b parameters: {z_b_params}")
        self._run_print(f"total parameters: {total_params}")


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
            self.early_stopper(np.mean(train_losses), model={'z_f': self.z_f, 'z_b': self.z_b, 'x_encoder': self.x_encoder})

            self._save_run_check_point(seed)


        self._load_best_model()
        best_test_result = self._test()

        self._save()

        return best_test_result

    def _predict(self, loader, desc="Predicting"):
        """Generate predictions on the specified DataLoader (using XEncoder)

        X features are compressed via XEncoder as condition to generate Y.
        """
        self.z_f.eval()
        self.z_b.eval()
        self.x_encoder.eval()

        all_preds = []
        all_truths = []

        with torch.no_grad():
            for batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, is_masked in tqdm(loader, desc=desc):
                batch_x = batch_x.to(self.device).float()  # (B, T, 33)
                batch_y = batch_y.to(self.device).float()  # (B, T, 1)

                # Average multiple samples for improved prediction stability
                # imputation now directly accepts batch_x (B, T, 33) and returns y_samples (B, S, T)
                y_samples = self.imputation(batch_x, num_samples=self.num_samples)  # (B, S, T)
                pred_mean = y_samples.mean(dim=1).unsqueeze(-1)  # (B, T, 1)

                all_preds.append(pred_mean.cpu().numpy())
                all_truths.append(batch_y.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_truths = np.concatenate(all_truths, axis=0)
        return all_preds, all_truths

    def _evaluate(self, dataloader):
        """Override parent's _evaluate method, using z_f/z_b instead of self.model"""
        self.z_f.eval()
        self.z_b.eval()
        self.metrics.reset()
        results = []

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

                # preds: (B, T, 1, S), truths: (B, T, 1)
                results.append(self.task_pool.apply_async(
                    update_metrics,
                    (preds.contiguous().cpu().detach(), truths.contiguous().cpu().detach(), self.metrics)
                ))

                progress_bar.update(batch_x.shape[0])

        for result in results:
            result.get()

        result = {name: float(metric.compute()) for name, metric in self.metrics.items()}
        return result

    def _test(self):
        """Override parent's _test method"""
        print("Testing .... ")
        test_result = self._evaluate(self.test_loader)
        self._run_print(f"test_results: {test_result}")
        return test_result

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
    fire.Fire(CSBICAMELS)
