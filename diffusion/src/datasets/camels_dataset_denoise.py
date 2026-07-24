"""
CAMELS Dataset with Denoising Training for Stage 2 Calibration

Key difference from camels_dataset.py:
- During training, adds noise to Y_history for ALL basins (not just masked ones)
- This prevents the model from learning "copy Y_history" shortcut
- Forces the model to learn true denoising/calibration capability

Design philosophy:
- At inference time, masked basins have LSTM predictions as Y_history (with errors)
- Training with noisy Y_history simulates this scenario
- The model learns to "calibrate" noisy input to recover true Y
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from torch_timeseries.core import TimeSeriesDataset


class CAMELSNpzDatasetDenoise(Dataset):
    """
    CAMELS Dataset with denoising training for Stage 2 calibration.

    Key feature: Adds configurable noise to Y_history during training,
    preventing the model from learning the "copy" shortcut.

    Noise types:
    - 'gaussian': Standard Gaussian noise scaled by noise_std
    - 'uniform': Uniform noise in [-noise_std, noise_std]
    - 'dropout': Randomly zero out portion of Y_history values
    - 'lstm_like': Noise calibrated to match typical LSTM prediction errors
    """

    def __init__(self, npz_path, split='train', window=168, pred_len=192,
                 masked_basins=None, mask_mode='noise',
                 denoise_training=True, noise_type='gaussian', noise_std=0.3,
                 dropout_rate=0.2):
        """
        Parameters:
        -----------
        npz_path : str
            .npz file path
        split : str
            'train', 'val', or 'test'
        window : int
            Input sequence length
        pred_len : int
            Prediction sequence length
        masked_basins : list or None
            Basins to mask (for CSDI-style masking, usually not used in Stage 2)
        mask_mode : str
            Mask fill mode: 'noise', 'zero', 'mean'
        denoise_training : bool
            Whether to add noise to Y_history during training (default: True)
            Only active when split='train'
        noise_type : str
            Type of noise to add: 'gaussian', 'uniform', 'dropout', 'lstm_like'
        noise_std : float
            Standard deviation of noise (for gaussian/uniform/lstm_like)
            Recommended range: 0.1 ~ 0.5 (data is standardized, so 0.3 = 30% of std)
        dropout_rate : float
            Dropout rate for 'dropout' noise type (fraction of values to zero out)
        """
        self.npz_path = npz_path
        self.split = split
        self.window = window
        self.pred_len = pred_len
        self.mask_mode = mask_mode

        # Denoising training parameters
        self.denoise_training = denoise_training and (split == 'train')
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.dropout_rate = dropout_rate

        # Load data
        data = np.load(npz_path, allow_pickle=True)

        # Select split
        if split == 'train':
            self.x = data['x_trn']
            self.y = data['y_obs_trn']
            self.times = data['times_trn']
            self.ids = data['ids_trn']
        elif split == 'val':
            self.x = data['x_val']
            self.y = data['y_obs_val']
            self.times = data['times_val']
            self.ids = data['ids_val']
        elif split == 'test':
            self.x = data['x_tst']
            self.y = data['y_obs_tst']
            self.times = data['times_tst']
            self.ids = data['ids_tst']
        else:
            raise ValueError(f"Unknown split: {split}")

        # Save scaling parameters
        self.x_mean = data['x_mean']
        self.x_std = data['x_std']
        self.y_mean = data['y_mean']
        self.y_std = data['y_std']
        self.basin_names = data['basin_names']

        # Build basin mappings
        self.basin_to_idx = {name: idx for idx, name in enumerate(self.basin_names)}
        self.idx_to_basin = {idx: name for idx, name in enumerate(self.basin_names)}

        # Handle masked_basins (CSDI-style, usually not used in Stage 2 denoise)
        self.masked_basin_ids = set()
        if masked_basins is not None:
            for item in masked_basins:
                if isinstance(item, (int, np.integer)):
                    if 0 <= item < len(self.basin_names):
                        self.masked_basin_ids.add(self.basin_names[item])
                else:
                    if item in self.basin_to_idx:
                        self.masked_basin_ids.add(item)

        # Sequence length
        self.seq_len = self.x.shape[1]

        # Calibration mode: window == pred_len == seq_len
        if window == pred_len == self.seq_len:
            print(f"Calibration mode: {self.seq_len} days input -> {self.seq_len} days output")
            self.n_original_samples = self.x.shape[0]
            self.n_samples = self.n_original_samples
        else:
            # Sliding window mode
            assert self.seq_len >= window + pred_len
            self.samples_per_seq = self.seq_len - window - pred_len + 1
            self.n_original_samples = self.x.shape[0]
            self.n_samples = self.n_original_samples * self.samples_per_seq

        print(f"Loaded {split} split:")
        print(f"  Samples: {self.n_samples}")
        print(f"  X shape: {self.x.shape}, Y shape: {self.y.shape}")
        if self.denoise_training:
            print(f"  Denoising training: ENABLED")
            print(f"    noise_type: {noise_type}")
            print(f"    noise_std: {noise_std}")
            if noise_type == 'dropout':
                print(f"    dropout_rate: {dropout_rate}")
        else:
            print(f"  Denoising training: DISABLED (inference mode)")

    def __len__(self):
        return self.n_samples

    def _add_noise(self, y_history):
        """
        Add noise to Y_history for denoising training.

        Parameters:
        -----------
        y_history : np.ndarray
            Y history values, shape (window, 1)

        Returns:
        --------
        np.ndarray : Noisy Y history, same shape
        """
        if self.noise_type == 'gaussian':
            # Standard Gaussian noise
            noise = np.random.randn(*y_history.shape) * self.noise_std
            return y_history + noise.astype(np.float32)

        elif self.noise_type == 'uniform':
            # Uniform noise in [-noise_std, noise_std]
            noise = np.random.uniform(-self.noise_std, self.noise_std, y_history.shape)
            return y_history + noise.astype(np.float32)

        elif self.noise_type == 'dropout':
            # Randomly zero out values (simulate missing data)
            mask = np.random.random(y_history.shape) > self.dropout_rate
            return (y_history * mask).astype(np.float32)

        elif self.noise_type == 'lstm_like':
            # LSTM-like errors: combination of bias and noise
            # Simulates typical LSTM prediction characteristics:
            # - Small systematic bias
            # - Noise proportional to signal magnitude
            bias = np.random.randn() * 0.1  # Small systematic bias
            noise = np.random.randn(*y_history.shape) * self.noise_std
            # Scale noise by absolute value (larger errors for extreme values)
            scale = 1.0 + 0.2 * np.abs(y_history)
            return (y_history + bias + noise * scale).astype(np.float32)

        else:
            # Unknown noise type, return original
            return y_history

    def __getitem__(self, idx):
        """
        Return a training sample with optional denoising.

        Key difference from original:
        - For ALL basins (not just masked), Y_history can be noised during training
        - This forces the model to learn calibration rather than copying
        """
        # Calibration mode
        if self.window == self.pred_len == self.seq_len:
            seq_idx = idx
            x_data = self.x[seq_idx, :, :]
            y_data = self.y[seq_idx, :, :]

            x_window = x_data
            y_history = y_data.copy()  # Make copy for potential modification
            y_target = y_data  # Target is always the true value
        else:
            # Sliding window mode
            seq_idx = idx // self.samples_per_seq
            time_offset = idx % self.samples_per_seq

            start_idx = time_offset
            mid_idx = start_idx + self.window
            end_idx = mid_idx + self.pred_len

            x_window = self.x[seq_idx, start_idx:mid_idx, :]
            y_history = self.y[seq_idx, start_idx:mid_idx, :].copy()
            y_target = self.y[seq_idx, mid_idx:end_idx, :]

        # Get basin info
        basin_id = self.ids[seq_idx, 0, 0]
        basin_idx = np.where(self.basin_names == basin_id)[0][0]

        # ========================================
        # Denoising training: Add noise to Y_history
        # ========================================
        if self.denoise_training:
            # Add noise to Y_history (for all basins during training)
            # This is the KEY difference from original dataset
            y_history = self._add_noise(y_history)

        # ========================================
        # CSDI-style masking (usually not used in Stage 2)
        # ========================================
        if basin_id in self.masked_basin_ids:
            if self.mask_mode == 'noise':
                y_history = np.random.randn(*y_history.shape).astype(np.float32)
            elif self.mask_mode == 'zero':
                y_history = np.zeros_like(y_history, dtype=np.float32)
            elif self.mask_mode == 'mean':
                y_history = np.zeros_like(y_history, dtype=np.float32)

        # ========================================
        # Construct batch_x and batch_y
        # ========================================
        batch_x_scaled = np.concatenate([x_window, y_history], axis=-1)
        batch_y_scaled = y_target

        # Inverse transform for origin_x, origin_y
        x_window_orig = x_window * (self.x_std + 1e-10) + self.x_mean

        if self.y_mean.ndim == 0 or len(self.y_mean) == 1:
            y_mean = self.y_mean.item() if self.y_mean.ndim == 0 else self.y_mean[0]
            y_std = self.y_std.item() if self.y_std.ndim == 0 else self.y_std[0]
        else:
            y_mean = self.y_mean[basin_idx]
            y_std = self.y_std[basin_idx]

        # origin_y uses TRUE y_history (for evaluation), not noised version
        y_history_orig = self.y[seq_idx, :self.window, :] if self.window == self.pred_len == self.seq_len \
            else self.y[seq_idx, start_idx:mid_idx, :]
        y_history_orig = y_history_orig * (y_std + 1e-10) + y_mean
        y_target_orig = y_target * (y_std + 1e-10) + y_mean

        origin_x = np.concatenate([x_window_orig, y_history_orig], axis=-1)
        origin_y = y_target_orig

        # Time encoding
        if self.window == self.pred_len == self.seq_len:
            times_x = self.times[seq_idx, :self.window, 0]
            times_y = self.times[seq_idx, :self.pred_len, 0]
        else:
            times_x = self.times[seq_idx, start_idx:mid_idx, 0]
            times_y = self.times[seq_idx, mid_idx:end_idx, 0]

        def datetime64_to_features(dt_array):
            import pandas as pd
            dt_series = pd.to_datetime(dt_array)
            features = np.zeros((len(dt_array), 4), dtype=np.float32)
            features[:, 0] = dt_series.month
            features[:, 1] = dt_series.day
            features[:, 2] = dt_series.dayofweek
            features[:, 3] = 0
            return features

        batch_x_date_enc = datetime64_to_features(times_x)
        batch_y_date_enc = datetime64_to_features(times_y)

        return (
            torch.FloatTensor(batch_x_scaled),
            torch.FloatTensor(batch_y_scaled),
            torch.FloatTensor(origin_x),
            torch.FloatTensor(origin_y),
            torch.FloatTensor(batch_x_date_enc),
            torch.FloatTensor(batch_y_date_enc),
        )


@dataclass
class CAMELSDenoise(TimeSeriesDataset):
    """CAMELS dataset class with denoising, compatible with torch_timeseries framework"""
    npz_path: str = '../data_processing/data/prepped.npz'
    num_features: int = 43
    freq: str = 'yd'

    def __post_init__(self):
        data = np.load(self.npz_path, allow_pickle=True)
        self.num_features = data['x_trn'].shape[2] + 1
        self.n_basins = int(data['n_segs'])
        del data


__all__ = ['CAMELSDenoise', 'CAMELSNpzDatasetDenoise']
