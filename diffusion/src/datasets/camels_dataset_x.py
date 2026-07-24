"""
CAMELS Dataset for NsDiff_X Framework (X -> Y Mode)
Adapts prepped.npz format to the NsDiff DataLoader interface

Design philosophy (X -> Y mode):
- Condition: X as condition input (42 dims) - only meteorological drivers, no Y
- Target: Y as diffusion target (1 dim)
- Model predicts Y purely from X, suitable for ungauged basins scenario

Differences from camels_dataset.py (X+Y -> Y):
- batch_x: 42 dims (X only), instead of 43 dims (X + Y_history)
- No mask mechanism needed, since there is no dependency on Y_history

Note: Temporal features are already encoded into X features during preprocessing (month_sin, month_cos, doy_sin, doy_cos),
so time encoding is not generated separately here; use time_embed=False to disable the model's time embedding layer.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from torch_timeseries.core import TimeSeriesDataset


class CAMELSNpzDatasetX(Dataset):
    """
    Load CAMELS data from .npz file, adapted to NsDiff DataLoader format (X -> Y mode)

    Data format:
    Input (npz):
        x: (N_samples, seq_len, n_features)      # 365 steps, 42 features (meteorological drivers)
        y: (N_samples, seq_len, 1)               # 365 steps, 1-dim target (streamflow)
        times: (N_samples, seq_len, 1)           # timestamps

    Output (per sample):
        batch_x: (window, 42)              # condition input = X only (42 dims, no Y)
        batch_y: (pred_len, 1)             # diffusion target = Y (1 dim)
        origin_x: (window, 42)             # un-normalized condition
        origin_y: (pred_len, 1)            # un-normalized target
        batch_x_date_enc: (window, 4)      # time encoding [month, day, weekday, hour]
        batch_y_date_enc: (pred_len, 4)    # time encoding [month, day, weekday, hour]
        is_masked: (1,)                    # whether it is a masked basin (interface compatible)
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
            List of basins to mask (in X -> Y mode, marks which basins' Y does not contribute to loss)
        mask_mode : str
            Mask fill mode (not used in X -> Y mode, parameter kept for compatibility)
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

        # Save scaling parameters (for de-normalization)
        self.x_mean = data['x_mean']
        self.x_std = data['x_std']
        self.y_mean = data['y_mean']  # (531,) per-basin
        self.y_std = data['y_std']    # (531,) per-basin
        self.basin_names = data['basin_names']

        # Build basin_id -> index mapping
        self.basin_to_idx = {name: idx for idx, name in enumerate(self.basin_names)}
        self.idx_to_basin = {idx: name for idx, name in enumerate(self.basin_names)}

        # Process masked_basins parameter, uniformly convert to basin_id set
        # In X -> Y mode, masked basins mark which basins' Y does not contribute to loss
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
                print(f"X -> Y mode: masked basins ({len(self.masked_basin_ids)}): {sorted(list(self.masked_basin_ids))[:5]}...")

        # Check sequence length
        self.seq_len = self.x.shape[1]  # 365

        # Calibration mode: window == pred_len == seq_len (365 days input -> 365 days output, time-aligned)
        if window == pred_len == self.seq_len:
            print(f"X -> Y Calibration mode: {self.seq_len} days X -> {self.seq_len} days Y")
            self.n_original_samples = self.x.shape[0]
            self.n_samples = self.n_original_samples  # No sliding window, one sample per sequence
        else:
            # Traditional forecasting mode: use sliding window
            print(f"X -> Y Forecasting mode: {window} days X -> {pred_len} days Y")
            assert self.seq_len >= window + pred_len, \
                f"seq_len ({self.seq_len}) < window ({window}) + pred_len ({pred_len})"
            self.samples_per_seq = self.seq_len - window - pred_len + 1
            self.n_original_samples = self.x.shape[0]
            self.n_samples = self.n_original_samples * self.samples_per_seq

        print(f"Loaded {split} split (X -> Y mode):")
        print(f"  Original samples: {self.n_original_samples}")
        print(f"  Effective samples: {self.n_samples}")
        print(f"  X shape: {self.x.shape} (42 features, no Y)")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Return a training sample (X -> Y mode)

        X -> Y mode description:
        - batch_x only contains X (42 dims), no Y_history
        - Model purely predicts streamflow from meteorological drivers
        - is_masked flag is used to skip masked basins' loss during training

        Returns:
        --------
        tuple of 7 tensors:
            batch_x: (window, 42)          # condition = X only (no Y)
            batch_y: (pred_len, 1)         # diffusion target = Y
            origin_x: (window, 42)         # un-normalized X
            origin_y: (pred_len, 1)        # un-normalized Y
            batch_x_date_enc: (window, 4)  # [month, day, weekday, hour]
            batch_y_date_enc: (pred_len, 4)
            is_masked: (1,)                # 1 = masked basin, 0 = normal basin
        """
        # Calibration mode: use the full sequence directly
        if self.window == self.pred_len == self.seq_len:
            seq_idx = idx

            # Extract the entire sequence (already normalized)
            x_data = self.x[seq_idx, :, :]  # (365, 42) - X features
            y_data = self.y[seq_idx, :, :]  # (365, 1)  - Y target

            # Input and output cover the same time period (time-aligned calibration task)
            x_window = x_data   # (365, 42)
            y_target = y_data   # (365, 1)

        else:
            # Forecasting mode: use sliding window
            seq_idx = idx // self.samples_per_seq
            time_offset = idx % self.samples_per_seq

            # Extract time window
            start_idx = time_offset
            mid_idx = start_idx + self.window
            end_idx = mid_idx + self.pred_len

            # Extract X data (already normalized)
            x_window = self.x[seq_idx, start_idx:mid_idx, :]  # (window, 42)

            # Extract Y data (already normalized)
            y_target = self.y[seq_idx, mid_idx:end_idx, :]     # (pred_len, 1)

        # Get the basin_id for the current sample
        basin_id = self.ids[seq_idx, 0, 0]
        basin_idx = np.where(self.basin_names == basin_id)[0][0]

        # ========================================
        # X -> Y mode: batch_x only contains X (42 dims)
        # ========================================
        batch_x_scaled = x_window  # (window, 42) - X only, no Y

        # batch_y: diffusion target = Y (1 dim)
        batch_y_scaled = y_target  # (pred_len, 1)

        # De-normalization (origin_x, origin_y)
        # X de-normalization
        x_window_orig = x_window * (self.x_std + 1e-10) + self.x_mean

        # Y de-normalization
        if self.y_mean.ndim == 0 or len(self.y_mean) == 1:
            # Global normalization: only one mean/std value
            y_mean = self.y_mean.item() if self.y_mean.ndim == 0 else self.y_mean[0]
            y_std = self.y_std.item() if self.y_std.ndim == 0 else self.y_std[0]
        else:
            # Per-basin normalization: each basin has its own mean/std
            y_mean = self.y_mean[basin_idx]
            y_std = self.y_std[basin_idx]

        y_target_orig = y_target * (y_std + 1e-10) + y_mean

        origin_x = x_window_orig  # (window, 42)
        origin_y = y_target_orig  # (pred_len, 1)

        # Time encoding: generate 4 dims [hour, weekday, day, month]
        if self.window == self.pred_len == self.seq_len:
            times_x = self.times[seq_idx, :self.window, 0]
            times_y = self.times[seq_idx, :self.pred_len, 0]
        else:
            times_x = self.times[seq_idx, start_idx:mid_idx, 0]
            times_y = self.times[seq_idx, mid_idx:end_idx, 0]

        # Convert datetime64 to time features
        def datetime64_to_features(dt_array):
            """Convert datetime64 array to [month, day, weekday, hour] features"""
            import pandas as pd
            dt_series = pd.to_datetime(dt_array)
            features = np.zeros((len(dt_array), 4), dtype=np.float32)
            features[:, 0] = dt_series.month  # month: 1-12
            features[:, 1] = dt_series.day  # day of month: 1-31
            features[:, 2] = dt_series.dayofweek  # weekday: 0-6
            features[:, 3] = 0  # hour: fixed to 0 for daily-scale data
            return features

        batch_x_date_enc = datetime64_to_features(times_x)  # (window, 4)
        batch_y_date_enc = datetime64_to_features(times_y)  # (pred_len, 4)

        # is_masked: marks whether it is a masked basin (for skipping loss during training)
        is_masked = 1.0 if basin_id in self.masked_basin_ids else 0.0

        # Convert to torch tensor (return 7 values, compatible with original interface)
        return (
            torch.FloatTensor(batch_x_scaled),      # (window, 42) - X only
            torch.FloatTensor(batch_y_scaled),      # (pred_len, 1) - target Y
            torch.FloatTensor(origin_x),            # (window, 42) - original X
            torch.FloatTensor(origin_y),            # (pred_len, 1)
            torch.FloatTensor(batch_x_date_enc),    # (window, 4)
            torch.FloatTensor(batch_y_date_enc),    # (pred_len, 4)
            torch.FloatTensor([is_masked]),         # (1,) - mask flag
        )


@dataclass
class CAMELS_X(TimeSeriesDataset):
    """
    CAMELS dataset class (X -> Y mode), compatible with torch_timeseries framework

    Difference from CAMELS (43 dims): num_features = 42 (X only, no Y)
    """
    npz_path: str = '../data_processing/data/prepped.npz'
    num_features: int = 42  # 42 X features only (no Y)
    freq: str = 'yd'  # Daily frequency (year-day format)

    def __post_init__(self):
        """Load data to obtain meta information"""
        data = np.load(self.npz_path, allow_pickle=True)
        self.num_features = data['x_trn'].shape[2]  # X features only (42)
        self.n_basins = int(data['n_segs'])

        # Release memory
        del data


# For direct import
__all__ = ['CAMELS_X', 'CAMELSNpzDatasetX']
