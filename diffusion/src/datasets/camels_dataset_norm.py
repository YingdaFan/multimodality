"""
CAMELS Dataset for NsDiff Framework (Norm Version - Per-basin Normalized)

Key differences from camels_dataset.py:
- Input (Y_history): y_obs_* (LSTM predictions, per-basin normalized)
- Label (Y_target): y_raw_* (true observations, per-basin normalized)
- Supports loss masking: returns is_masked flag

Design philosophy:
- Learn the correction mapping from LSTM_predictions -> True_observations
- Masked basins are excluded from loss via the is_masked flag to prevent information leakage
- All basins participate in the forward pass (shared model parameters)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from torch_timeseries.core import TimeSeriesDataset


class CAMELSNpzDatasetNorm(Dataset):
    """
    Load CAMELS data from .npz file (Norm version)

    Key differences:
    - y_obs_*: used as input (Y_history) - LSTM predictions
    - y_raw_*: used as label (Y_target) - true observations
    - is_masked: marks whether the sample is a masked basin (for loss masking)

    Output (per sample):
        batch_x: (window, n_features+1)          # condition input = [X, Y_history] (43 dims)
        batch_y: (pred_len, 1)                   # label = Y_raw (true observations)
        origin_x: (window, n_features+1)         # un-normalized condition
        origin_y: (pred_len, 1)                  # un-normalized target
        batch_x_date_enc: (window, 4)            # time encoding [month, day, weekday, hour]
        batch_y_date_enc: (pred_len, 4)          # time encoding
        is_masked: bool                          # whether it is a masked basin (for loss masking)
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
            List of basins to exclude from loss, supports:
            - Basin ID (string): ['01022500', '02069700']
            - Basin Index (integer): [0, 100, 530]
            Note: these basins still participate in forward pass, just do not contribute to loss
        mask_mode : str
            Fill mode for input Y_history (for masked basins):
            - 'noise': fill with standard Gaussian noise
            - 'zero': fill with zeros
            - 'original': keep original LSTM predictions (default, no special handling)
        """
        self.npz_path = npz_path
        self.split = split
        self.window = window
        self.pred_len = pred_len
        self.mask_mode = mask_mode

        # Load data
        data = np.load(npz_path, allow_pickle=True)

        # Select data for the corresponding split
        # y_obs_*: LSTM predictions (input)
        # y_raw_*: true observations (label)
        if split == 'train':
            self.x = data['x_trn']               # (N, 365, 42)
            self.y_obs = data['y_obs_trn']       # (N, 365, 1) - LSTM predictions
            self.y_raw = data['y_raw_trn']       # (N, 365, 1) - true observations
            self.times = data['times_trn']       # (N, 365, 1)
            self.ids = data['ids_trn']           # (N, 365, 1)
        elif split == 'val':
            self.x = data['x_val']
            self.y_obs = data['y_obs_val']
            self.y_raw = data['y_raw_val']
            self.times = data['times_val']
            self.ids = data['ids_val']
        elif split == 'test':
            self.x = data['x_tst']
            self.y_obs = data['y_obs_tst']
            self.y_raw = data['y_raw_tst']
            self.times = data['times_tst']
            self.ids = data['ids_tst']
        else:
            raise ValueError(f"Unknown split: {split}")

        # Save scaling parameters (for de-normalization)
        self.x_mean = data['x_mean']
        self.x_std = data['x_std']
        self.y_mean = data['y_mean']  # (531,) per-basin - statistics of raw observations
        self.y_std = data['y_std']    # (531,) per-basin
        # VAE prediction statistics (Stage 1 LSTM uses these for normalization)
        self.y_mean_vae = data['y_mean_vae']  # (531,) per-basin
        self.y_std_vae = data['y_std_vae']    # (531,) per-basin
        self.basin_names = data['basin_names']

        # Build basin_id -> index mapping
        self.basin_to_idx = {name: idx for idx, name in enumerate(self.basin_names)}
        self.idx_to_basin = {idx: name for idx, name in enumerate(self.basin_names)}

        # Process masked_basins parameter, uniformly convert to basin_id set
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
                print(f"Loss Masking: {len(self.masked_basin_ids)} basins excluded from loss")
                print(f"Masked basins: {sorted(list(self.masked_basin_ids))[:5]}...")

        # Print Norm version specific info
        print(f"Norm Pipeline:")
        print(f"   - Input (Y_history): y_obs_* (LSTM predictions)")
        print(f"   - Label (Y_target): y_raw_* (True observations)")
        print(f"   - Learning: LSTM predictions -> True observations")

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
        print(f"  Y_obs shape: {self.y_obs.shape}")
        print(f"  Y_raw shape: {self.y_raw.shape}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Return a training sample (Norm version)

        Returns:
        --------
        tuple of 7 tensors:
            batch_x: (window, 43)      # condition = [X(42), Y_obs(1)], input is LSTM predictions
            batch_y: (pred_len, 1)     # label = Y_raw (true observations)
            origin_x: (window, 43)     # un-normalized condition
            origin_y: (pred_len, 1)    # un-normalized target
            batch_x_date_enc: (window, 4)  # [month, day, weekday, hour]
            batch_y_date_enc: (pred_len, 4)
            is_masked: bool tensor  # for loss masking
        """
        # Calibration mode: use the full sequence directly
        if self.window == self.pred_len == self.seq_len:
            seq_idx = idx

            # Extract the entire sequence (already normalized)
            x_data = self.x[seq_idx, :, :]         # (365, 42) - X features
            y_obs_data = self.y_obs[seq_idx, :, :] # (365, 1) - LSTM predictions (input)
            y_raw_data = self.y_raw[seq_idx, :, :] # (365, 1) - true observations (label)

            # Input and output cover the same time period (time-aligned calibration task)
            x_window = x_data        # (365, 42)
            y_history = y_obs_data   # (365, 1) - input: LSTM predictions
            y_target = y_raw_data    # (365, 1) - label: true observations

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

            # Extract Y data
            y_history = self.y_obs[seq_idx, start_idx:mid_idx, :]  # (window, 1) - input: LSTM predictions
            y_target = self.y_raw[seq_idx, mid_idx:end_idx, :]     # (pred_len, 1) - label: true observations

        # Get the basin_id for the current sample
        basin_id = self.ids[seq_idx, 0, 0]
        basin_idx = np.where(self.basin_names == basin_id)[0][0]

        # ========================================
        # Check if it is a masked basin
        # ========================================
        is_masked = basin_id in self.masked_basin_ids  # bool

        # Optional: process input for masked basins
        if is_masked and self.mask_mode != 'original':
            if self.mask_mode == 'noise':
                # Fill with standard Gaussian noise
                y_history = np.random.randn(*y_history.shape).astype(np.float32)
            elif self.mask_mode == 'zero':
                # Fill with zeros
                y_history = np.zeros_like(y_history, dtype=np.float32)

        # ========================================
        # Build condition input and target
        # ========================================

        # Get normalization statistics for the current basin (per-basin)
        y_mean_vae = self.y_mean_vae[basin_idx]
        y_std_vae = self.y_std_vae[basin_idx]
        y_mean = self.y_mean[basin_idx]
        y_std = self.y_std[basin_idx]

        # batch_x: condition input = [X, Y_obs] (43 dims)
        # y_history (y_obs) is already normalized
        batch_x_scaled = np.concatenate([x_window, y_history], axis=-1)  # (window, 43)

        # batch_y: label = Y_raw (1 dim, true observations)
        # y_raw is at original scale, normalize with y_mean/y_std (raw observation statistics)
        # postprocess uses y_mean_vae/y_std_vae for de-normalization (--use_vae_stats)
        batch_y_scaled = (y_target - y_mean) / (y_std + 1e-10)  # (pred_len, 1)

        # origin_x, origin_y: original values for visualization and evaluation
        # Stage 2 does not apply any normalization/de-normalization to X, X remains as-is
        # Y_obs de-normalization (input) - using y_mean_vae/y_std_vae
        if self.window == self.pred_len == self.seq_len:
            y_history_orig = self.y_obs[seq_idx, :self.window, :]
        else:
            y_history_orig = self.y_obs[seq_idx, start_idx:mid_idx, :]
        y_history_orig = y_history_orig * (y_std_vae + 1e-10) + y_mean_vae

        # Y_raw (label) - y_target is already at original scale, no de-normalization needed
        y_target_orig = y_target  # (pred_len, 1) already at original values

        origin_x = np.concatenate([x_window, y_history_orig], axis=-1)  # (window, 43) - X is not transformed
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

        # Convert to torch tensor (return 7 values, including is_masked)
        return (
            torch.FloatTensor(batch_x_scaled),      # (window, 43) - condition [X, Y_obs]
            torch.FloatTensor(batch_y_scaled),      # (pred_len, 1) - label Y_raw (true observations)
            torch.FloatTensor(origin_x),            # (window, 43) - original values
            torch.FloatTensor(origin_y),            # (pred_len, 1)
            torch.FloatTensor(batch_x_date_enc),    # (window, 4) - [month, day, weekday, hour]
            torch.FloatTensor(batch_y_date_enc),    # (pred_len, 4)
            torch.BoolTensor([is_masked]),          # (1,) - whether it is a masked basin
        )


@dataclass
class CAMELSNorm(TimeSeriesDataset):
    """
    CAMELS dataset class (Norm version), compatible with torch_timeseries framework
    """
    npz_path: str = '../data_processing/data/prepped.npz'
    num_features: int = 43  # 42 X features + 1 Y feature
    freq: str = 'yd'  # Daily frequency (year-day format)

    def __post_init__(self):
        """Load data to obtain meta information"""
        data = np.load(self.npz_path, allow_pickle=True)
        self.num_features = data['x_trn'].shape[2] + 1  # X features + Y
        self.n_basins = int(data['n_segs'])

        # Release memory
        del data


# For direct import
__all__ = ['CAMELSNorm', 'CAMELSNpzDatasetNorm']
