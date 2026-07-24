"""
CAMELS Dataset for RAW Pipeline (LOSS MASKING DESIGN)

Data flow:
1. y_obs_* = ALL basins use LSTM predictions (denormalized)
2. y_raw_* = True observations (unchanged, for evaluation)

Training design:
- ALL basins are included in the dataset (no filtering)
- Loss calculation: Only compute loss for non-masked basins
- Masked basins still do forward pass, but their loss is not backpropagated
- Returns is_masked flag for each sample

Rationale:
- Masked basins simulate ungauged stations with no historical observations
- Model learns correction: y_obs (LSTM) -> y_raw (true) on non-masked basins
- At inference, apply learned correction to all basins (including masked)
- All basins have predictions for evaluation
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from torch_timeseries.core import TimeSeriesDataset


class CAMELSNpzDatasetRaw(Dataset):
    """
    CAMELS Dataset for RAW Pipeline (LOSS MASKING DESIGN).

    Key design:
    - ALL basins included in dataset (no filtering)
    - Returns is_masked flag for loss masking in training/validation
    - Input: y_obs_* (LSTM predictions, globally normalized)
    - Label: y_raw (true observations, globally normalized)
    - Evaluation: y_raw (ground truth, original scale)
    """

    def __init__(self, npz_path, split='train', window=168, pred_len=192,
                 masked_basins=None, mask_mode='noise',
                 y_global_mean=None, y_global_std=None):
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
            Basins to mask in loss calculation (simulated ungauged stations)
            All basins are still included in the dataset for forward pass
        mask_mode : str
            Not used in new design (kept for compatibility)
        y_global_mean : float, optional
            Global Y mean for normalization
        y_global_std : float, optional
            Global Y std for normalization
        """
        self.npz_path = npz_path
        self.split = split
        self.window = window
        self.pred_len = pred_len
        self.mask_mode = mask_mode

        # Load data
        data = np.load(npz_path, allow_pickle=True)

        # Load arrays based on split
        if split == 'train':
            self.x = data['x_trn']
            self.y_obs = data['y_obs_trn']   # LSTM predictions (denormalized) for ALL basins
            self.y_raw = data['y_raw_trn']   # True observations (ground truth)
            self.times = data['times_trn']
            self.ids = data['ids_trn']
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

        # X scaling parameters
        self.x_mean = data['x_mean']
        self.x_std = data['x_std']

        # Basin info
        self.basin_names = data['basin_names']
        self.basin_to_idx = {name: idx for idx, name in enumerate(self.basin_names)}

        # ========================================
        # Basin-specific statistics (for conditioning)
        # ========================================
        self.y_mean_vae = data['y_mean_vae']  # (n_basins,) - per-basin Y mean
        self.y_std_vae = data['y_std_vae']    # (n_basins,) - per-basin Y std
        print(f"  Loaded basin statistics: y_mean_vae, y_std_vae for {len(self.y_mean_vae)} basins")

        # ========================================
        # Handle masked_basins (LOSS MASKING DESIGN)
        # ========================================
        self.masked_basin_ids = set()
        if masked_basins is not None:
            for item in masked_basins:
                if isinstance(item, (int, np.integer)):
                    if 0 <= item < len(self.basin_names):
                        self.masked_basin_ids.add(self.basin_names[item])
                else:
                    if item in self.basin_to_idx:
                        self.masked_basin_ids.add(item)

            print(f"  Masked basins ({len(self.masked_basin_ids)}): {sorted(list(self.masked_basin_ids))[:5]}...")

        # ========================================
        # Create is_masked array for each sample (NO FILTERING)
        # is_masked[i] = True if sample i belongs to a masked basin
        # ========================================
        n_samples_raw = len(self.ids)
        self.is_masked = np.zeros(n_samples_raw, dtype=bool)
        n_masked_samples = 0
        for i in range(n_samples_raw):
            basin_id = self.ids[i, 0, 0]
            if basin_id in self.masked_basin_ids:
                self.is_masked[i] = True
                n_masked_samples += 1

        if len(self.masked_basin_ids) > 0:
            print(f"  {split}: {n_masked_samples} masked samples (loss excluded), {n_samples_raw - n_masked_samples} non-masked samples")
        print(f"  Total samples: {n_samples_raw} (all basins included)")

        # ========================================
        # GLOBAL Y normalization
        # ========================================
        if y_global_mean is not None and y_global_std is not None:
            self.y_global_mean = y_global_mean
            self.y_global_std = y_global_std
        else:
            # Compute from y_raw_trn (true observations)
            y_raw_trn = data['y_raw_trn']
            y_flat = y_raw_trn.flatten()
            self.y_global_mean = float(np.nanmean(y_flat))
            self.y_global_std = float(np.nanstd(y_flat))

        print(f"  Global Y normalization: mean={self.y_global_mean:.4f}, std={self.y_global_std:.4f}")

        # Apply global normalization
        self.y_obs_norm = (self.y_obs - self.y_global_mean) / (self.y_global_std + 1e-10)
        self.y_raw_norm = (self.y_raw - self.y_global_mean) / (self.y_global_std + 1e-10)

        # ========================================
        # y_label = y_raw for all samples
        # ========================================
        # Loss masking is done in training loop, not here
        self.y_label_norm = self.y_raw_norm  # Direct reference, no copy needed

        # Sequence length
        self.seq_len = self.x.shape[1]

        # Calibration mode
        if window == pred_len == self.seq_len:
            print(f"  Calibration mode: {self.seq_len} days (RAW pipeline)")
            self.n_original_samples = self.x.shape[0]
            self.n_samples = self.n_original_samples
        else:
            assert self.seq_len >= window + pred_len
            self.samples_per_seq = self.seq_len - window - pred_len + 1
            self.n_original_samples = self.x.shape[0]
            self.n_samples = self.n_original_samples * self.samples_per_seq

        print(f"Loaded {split} split (RAW - LOSS MASKING DESIGN):")
        print(f"  Samples: {self.n_samples}")
        print(f"  X shape: {self.x.shape}")
        print(f"  Input: y_obs (LSTM predictions)")
        print(f"  Label: y_raw (true observations)")
        print(f"  Returns: is_masked flag for loss masking")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Return a sample for training.

        Returns:
        --------
        batch_x: [X, Y_obs] - input (LSTM predictions, globally normalized)
        batch_y: Y_raw - training label (true observations, globally normalized)
        origin_x: denormalized input
        origin_y: y_raw (ground truth, original scale for evaluation)
        batch_x_date_enc: time encoding for input
        batch_y_date_enc: time encoding for output
        is_masked: bool - True if this sample belongs to a masked basin (exclude from loss)
        """
        # Calibration mode
        if self.window == self.pred_len == self.seq_len:
            seq_idx = idx
            x_data = self.x[seq_idx, :, :]
            y_obs_data = self.y_obs_norm[seq_idx, :, :]     # Input (LSTM predictions)
            y_label_data = self.y_label_norm[seq_idx, :, :] # Label = y_raw
            y_raw_data = self.y_raw[seq_idx, :, :]          # Ground truth (original scale)

            x_window = x_data
            y_history = y_obs_data.copy()  # Input uses y_obs (LSTM predictions)
            y_target = y_label_data        # Label = y_raw
            y_target_raw = y_raw_data      # Ground truth for evaluation
        else:
            seq_idx = idx // self.samples_per_seq
            time_offset = idx % self.samples_per_seq

            start_idx = time_offset
            mid_idx = start_idx + self.window
            end_idx = mid_idx + self.pred_len

            x_window = self.x[seq_idx, start_idx:mid_idx, :]
            y_history = self.y_obs_norm[seq_idx, start_idx:mid_idx, :].copy()
            y_target = self.y_label_norm[seq_idx, mid_idx:end_idx, :]
            y_target_raw = self.y_raw[seq_idx, mid_idx:end_idx, :]

        # Get basin info
        basin_id = self.ids[seq_idx, 0, 0]
        basin_idx = self.basin_to_idx[basin_id]

        # Get basin-specific statistics and broadcast to (T, 1)
        y_mean_basin = np.full((self.window, 1), self.y_mean_vae[basin_idx], dtype=np.float32)
        y_std_basin = np.full((self.window, 1), self.y_std_vae[basin_idx], dtype=np.float32)

        # Construct batch_x and batch_y
        # batch_x: [X(42), y_mean_vae(1), y_std_vae(1), y_obs(1)] = 45 dims
        # y_obs always at the last position for easy extraction
        batch_x_scaled = np.concatenate([x_window, y_mean_basin, y_std_basin, y_history], axis=-1)  # (window, 45)
        batch_y_scaled = y_target  # (pred_len, 1) - globally normalized label

        # origin_x, origin_y: for visualization and evaluation
        # X is not normalized/denormalized, kept as-is
        # Y: use original raw values for origin_y (ground truth)
        if self.window == self.pred_len == self.seq_len:
            y_history_orig = self.y_raw[seq_idx, :self.window, :]
        else:
            y_history_orig = self.y_raw[seq_idx, start_idx:mid_idx, :]

        origin_x = np.concatenate([x_window, y_mean_basin, y_std_basin, y_history_orig], axis=-1)  # (window, 45)
        origin_y = y_target_raw  # Ground truth (original scale) for evaluation

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

        # Get is_masked flag for this sample
        is_masked = self.is_masked[seq_idx]

        return (
            torch.FloatTensor(batch_x_scaled),
            torch.FloatTensor(batch_y_scaled),
            torch.FloatTensor(origin_x),
            torch.FloatTensor(origin_y),
            torch.FloatTensor(batch_x_date_enc),
            torch.FloatTensor(batch_y_date_enc),
            torch.BoolTensor([is_masked]),  # Shape: (1,) for easy batch stacking
        )

    def get_global_stats(self):
        """Return global Y statistics for denormalization"""
        return self.y_global_mean, self.y_global_std


@dataclass
class CAMELSRaw(TimeSeriesDataset):
    """CAMELS dataset class for RAW pipeline"""
    npz_path: str = '../data_processing/data/prepped.npz'
    num_features: int = 45  # X(42) + y_obs(1) + y_mean_vae(1) + y_std_vae(1)
    freq: str = 'yd'

    def __post_init__(self):
        data = np.load(self.npz_path, allow_pickle=True)
        self.num_features = data['x_trn'].shape[2] + 3  # X + y_mean_vae + y_std_vae + y_obs
        self.n_basins = int(data['n_segs'])

        # Compute global Y stats from y_raw_trn (true observations)
        y_raw_trn = data['y_raw_trn']
        y_flat = y_raw_trn.flatten()
        self.y_global_mean = float(np.nanmean(y_flat))
        self.y_global_std = float(np.nanstd(y_flat))

        print(f"CAMELSRaw: Global Y mean={self.y_global_mean:.4f}, std={self.y_global_std:.4f}")

        del data


__all__ = ['CAMELSRaw', 'CAMELSNpzDatasetRaw']
