"""
CAMELS Dataset for NsDiff Framework
Adapts prepped.npz format to NsDiff's DataLoader interface

Design philosophy (TimeGrad/TSDiff style):
- Condition: [X, Y_history] as conditional input (43 dimensions)
- Target: Y as diffusion target (1 dimension)
- Clear separation of condition and target, with condition information injected via Cross-Attention

Note: Temporal features are already encoded into X features during preprocessing (month_sin, month_cos, doy_sin, doy_cos),
so no separate time encoding is generated here. Use time_embed=False to disable the model's temporal embedding layer.

Masking mechanism (CSDI style):
- Supports mask to specify basin Y_history
- Masked basins: Y_history is filled with Gaussian noise (diffusion model needs to impute)
- Returns y_history_mask to tell the model which values are real and which need imputation
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from torch_timeseries.core import TimeSeriesDataset


class CAMELSNpzDataset(Dataset):
    """
    Loads CAMELS data from .npz file, adapted to NsDiff DataLoader format

    Data format (TimeGrad/TSDiff style):
    Input (npz):
        x: (N_samples, seq_len, n_features)      # 365 steps, 42 features (meteorological drivers)
        y: (N_samples, seq_len, 1)               # 365 steps, 1-dim target (streamflow)
        times: (N_samples, seq_len, 1)           # timestamps

    Output (per sample):
        batch_x: (window, n_features+1)          # conditional input = [X, Y_history] (43 dims)
        batch_y: (pred_len, 1)                   # diffusion target = Y (1 dim)
        origin_x: (window, n_features+1)         # unnormalized condition
        origin_y: (pred_len, 1)                  # unnormalized target
        batch_x_date_enc: (window, 4)            # time encoding [month, day, weekday, hour]
        batch_y_date_enc: (pred_len, 4)          # time encoding [month, day, weekday, hour]
    """

    def __init__(self, npz_path, split='train', window=168, pred_len=192,
                 masked_basins=None, mask_mode='noise'):
        """
        Parameters:
        -----------
        npz_path : str
            Path to .npz file
        split : str
            'train', 'val', or 'test'
        window : int
            Input sequence length (history window)
        pred_len : int
            Prediction sequence length
        masked_basins : list or None
            List of basins whose Y_history should be masked, supports two formats:
            - Basin ID (string): ['01022500', '02069700']
            - Basin Index (integer): [0, 100, 530]
            If None, no masking is applied
        mask_mode : str
            Mask fill mode:
            - 'noise': Fill with standard Gaussian noise (CSDI style, recommended for diffusion models)
            - 'zero': Fill with 0 (SSSD style)
            - 'mean': Fill with 0 (mean is 0 after standardization)
        """
        self.npz_path = npz_path
        self.split = split
        self.window = window
        self.pred_len = pred_len
        self.mask_mode = mask_mode

        # Load data
        data = np.load(npz_path, allow_pickle=True)

        # Select data for the corresponding split
        if split == 'train':
            self.x = data['x_trn']          # (N, 365, 42)
            self.y = data['y_obs_trn']      # (N, 365, 1)
            self.times = data['times_trn']  # (N, 365, 1)
            self.ids = data['ids_trn']      # (N, 365, 1)
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

        # Save scaling parameters (for inverse standardization)
        self.x_mean = data['x_mean']
        self.x_std = data['x_std']
        self.y_mean = data['y_mean']  # (531,) per-basin
        self.y_std = data['y_std']    # (531,) per-basin
        self.basin_names = data['basin_names']

        # Build basin_id -> index mapping
        self.basin_to_idx = {name: idx for idx, name in enumerate(self.basin_names)}
        self.idx_to_basin = {idx: name for idx, name in enumerate(self.basin_names)}

        # Process masked_basins parameter, convert uniformly to basin_id set
        self.masked_basin_ids = set()
        if masked_basins is not None:
            for item in masked_basins:
                if isinstance(item, (int, np.integer)):
                    # Integer index -> convert to basin_id
                    if 0 <= item < len(self.basin_names):
                        self.masked_basin_ids.add(self.basin_names[item])
                    else:
                        print(f"Warning: basin index {item} out of range, ignored")
                else:
                    # String basin_id
                    if item in self.basin_to_idx:
                        self.masked_basin_ids.add(item)
                    else:
                        print(f"Warning: basin ID '{item}' not found, ignored")

            if self.masked_basin_ids:
                print(f"Mask mode: {mask_mode}")
                print(f"Masked basins ({len(self.masked_basin_ids)}): {sorted(list(self.masked_basin_ids))[:5]}...")

        # Check sequence length
        self.seq_len = self.x.shape[1]  # 365

        # Calibration mode: window == pred_len == seq_len (365 days input -> 365 days output, time-aligned)
        if window == pred_len == self.seq_len:
            print(f"Calibration mode: {self.seq_len} days input -> {self.seq_len} days output (time-aligned)")
            self.n_original_samples = self.x.shape[0]
            self.n_samples = self.n_original_samples  # No sliding window, one sample per sequence
        else:
            # Traditional forecasting mode: use sliding window
            print(f"Forecasting mode: {window} days input -> {pred_len} days output")
            assert self.seq_len >= window + pred_len, \
                f"seq_len ({self.seq_len}) < window ({window}) + pred_len ({pred_len})"
            self.samples_per_seq = self.seq_len - window - pred_len + 1
            self.n_original_samples = self.x.shape[0]
            self.n_samples = self.n_original_samples * self.samples_per_seq

        print(f"Loaded {split} split:")
        print(f"  Original samples: {self.n_original_samples}")
        print(f"  Effective samples: {self.n_samples}")
        print(f"  X shape: {self.x.shape}")
        print(f"  Y shape: {self.y.shape}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Return a training sample (TimeGrad/TSDiff style + CSDI mask mechanism)

        Mask mechanism description:
        - Masked basins: Y_history is internally replaced with noise/0
        - The last dimension of batch_x already contains the processed values
        - Training code needs no additional handling, just train normally
        - During evaluation, use postprocessing scripts to specify basins for analysis

        Returns:
        --------
        tuple of 6 tensors:
            batch_x: (window, 43)      # condition = [X(42), Y_history(1)], masked basin Y_history already replaced
            batch_y: (pred_len, 1)     # diffusion target = Y (true values, unaffected by mask)
            origin_x: (window, 43)     # unnormalized condition (contains true Y_history, for evaluation)
            origin_y: (pred_len, 1)    # unnormalized target
            batch_x_date_enc: (window, 4)  # [month, day, weekday, hour]
            batch_y_date_enc: (pred_len, 4)
        """
        # Calibration mode: use full sequence directly
        if self.window == self.pred_len == self.seq_len:
            seq_idx = idx

            # Extract entire sequence (already standardized)
            x_data = self.x[seq_idx, :, :]  # (365, 42) - X features
            y_data = self.y[seq_idx, :, :]  # (365, 1)  - Y target

            # Input and output cover the same time period (time-aligned calibration task)
            x_window = x_data   # (365, 42)
            y_history = y_data  # (365, 1)
            y_target = y_data   # (365, 1) - Calibration task: input and output Y are the same

        else:
            # Forecasting mode: use sliding window
            seq_idx = idx // self.samples_per_seq
            time_offset = idx % self.samples_per_seq

            # Extract time window
            start_idx = time_offset
            mid_idx = start_idx + self.window
            end_idx = mid_idx + self.pred_len

            # Extract X data (already standardized)
            x_window = self.x[seq_idx, start_idx:mid_idx, :]  # (window, 42)

            # Extract Y data (already standardized)
            y_history = self.y[seq_idx, start_idx:mid_idx, :]  # (window, 1)
            y_target = self.y[seq_idx, mid_idx:end_idx, :]     # (pred_len, 1)

        # Get the basin_id for the current sample
        basin_id = self.ids[seq_idx, 0, 0]
        basin_idx = np.where(self.basin_names == basin_id)[0][0]

        # ========================================
        # CSDI-style masking mechanism
        # ========================================
        # y_history_mask: 1 = valid observation, 0 = masked (needs imputation)
        y_history_mask = np.ones((self.window, 1), dtype=np.float32)

        if basin_id in self.masked_basin_ids:
            # This basin is masked, mark entire Y_history as missing
            y_history_mask[:] = 0.0

            # Fill Y_history according to mask_mode
            if self.mask_mode == 'noise':
                # CSDI style: fill with standard Gaussian noise
                y_history = np.random.randn(*y_history.shape).astype(np.float32)
            elif self.mask_mode == 'zero':
                # SSSD style: fill with 0
                y_history = np.zeros_like(y_history, dtype=np.float32)
            elif self.mask_mode == 'mean':
                # Fill with 0 (mean after standardization)
                y_history = np.zeros_like(y_history, dtype=np.float32)
            # else: keep original values (not recommended, only for debugging)

        # ========================================
        # TimeGrad/TSDiff style: clear separation of condition and target
        # ========================================

        # batch_x: conditional input = [X, Y_history] (43 dims)
        # Note: if basin is masked, y_history here has already been replaced
        batch_x_scaled = np.concatenate([x_window, y_history], axis=-1)  # (window, 43)

        # batch_y: diffusion target = Y (1 dim)
        # This is what the diffusion model needs to predict (unaffected by mask)
        batch_y_scaled = y_target  # (pred_len, 1)

        # Inverse standardization (origin_x, origin_y)
        # X inverse standardization
        x_window_orig = x_window * (self.x_std + 1e-10) + self.x_mean

        # Y inverse standardization
        # Compatible with two normalization approaches:
        # - Per-basin: y_mean.shape == (n_basins,) -> use y_mean[basin_idx]
        # - Global (all-basin): y_mean.shape == (1,) or scalar -> use y_mean[0] or y_mean
        if self.y_mean.ndim == 0 or len(self.y_mean) == 1:
            # Global normalization: only one mean/std value
            y_mean = self.y_mean.item() if self.y_mean.ndim == 0 else self.y_mean[0]
            y_std = self.y_std.item() if self.y_std.ndim == 0 else self.y_std[0]
        else:
            # Per-basin normalization: each basin has its own mean/std
            y_mean = self.y_mean[basin_idx]
            y_std = self.y_std[basin_idx]

        # For masked basins, origin_y still retains true values (for evaluation)
        y_history_orig = self.y[seq_idx, :self.window, :] if self.window == self.pred_len == self.seq_len \
            else self.y[seq_idx, start_idx:mid_idx, :]
        y_history_orig = y_history_orig * (y_std + 1e-10) + y_mean
        y_target_orig = y_target * (y_std + 1e-10) + y_mean

        origin_x = np.concatenate([x_window_orig, y_history_orig], axis=-1)  # (window, 43)
        origin_y = y_target_orig  # (pred_len, 1)

        # Time encoding: generate 4 dims [hour, weekday, day, month] (compatible with D3VAE and other models)
        # For daily-scale data: hour=0, weekday=day_of_year%7, day=day_of_month, month=month
        if self.window == self.pred_len == self.seq_len:
            times_x = self.times[seq_idx, :self.window, 0]
            times_y = self.times[seq_idx, :self.pred_len, 0]
        else:
            times_x = self.times[seq_idx, start_idx:mid_idx, 0]
            times_y = self.times[seq_idx, mid_idx:end_idx, 0]

        # Convert datetime64 to temporal features
        def datetime64_to_features(dt_array):
            """Convert datetime64 array to [month, day, weekday, hour] features
            Order matches torch_timeseries TemporalEmbedding:
            - x[:,:,0] -> month_embed (1-12)
            - x[:,:,1] -> day_embed (1-31)
            - x[:,:,2] -> weekday_embed (0-6)
            - x[:,:,3] -> hour_embed (0-23, fixed to 0 for daily-scale data)
            """
            import pandas as pd
            dt_series = pd.to_datetime(dt_array)
            features = np.zeros((len(dt_array), 4), dtype=np.float32)
            features[:, 0] = dt_series.month  # month: 1-12
            features[:, 1] = dt_series.day  # day of month: 1-31
            features[:, 2] = dt_series.dayofweek  # weekday: 0-6
            features[:, 3] = dt_series.hour  # hour: 0-23
            return features

        batch_x_date_enc = datetime64_to_features(times_x)  # (window, 4)
        batch_y_date_enc = datetime64_to_features(times_y)  # (pred_len, 4)

        # is_masked: flag indicating whether this is a masked basin (used to skip loss during training)
        is_masked = basin_id in self.masked_basin_ids

        # Convert to torch tensors (returns 7 values)
        return (
            torch.FloatTensor(batch_x_scaled),      # (window, 43) - condition [X, Y_history], masked basins already replaced
            torch.FloatTensor(batch_y_scaled),      # (pred_len, 1) - target Y (true values)
            torch.FloatTensor(origin_x),            # (window, 43) - original values (contains true Y_history)
            torch.FloatTensor(origin_y),            # (pred_len, 1)
            torch.FloatTensor(batch_x_date_enc),    # (window, 4) - [month, day, weekday, hour]
            torch.FloatTensor(batch_y_date_enc),    # (pred_len, 4) - [month, day, weekday, hour]
            torch.BoolTensor([is_masked]),          # (1,) - True=masked, False=normal
        )


@dataclass
class CAMELS(TimeSeriesDataset):
    """
    CAMELS dataset class, compatible with torch_timeseries framework
    """
    npz_path: str = '../data_processing/data/prepped.npz'
    num_features: int = 43  # 42 X features + 1 Y feature
    freq: str = 'yd'  # Daily frequency (year-day format)

    def __post_init__(self):
        """Load data to obtain metadata"""
        data = np.load(self.npz_path, allow_pickle=True)
        self.num_features = data['x_trn'].shape[2] + 1  # X features + Y
        self.n_basins = int(data['n_segs'])

        # Free memory
        del data


# For direct import
__all__ = ['CAMELS', 'CAMELSNpzDataset']
