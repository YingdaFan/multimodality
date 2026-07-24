"""
Forecast Dataset — single-layer sliding window on continuous sequences.

Input npz format:
    x_trn:           (n_basins, n_times, n_feat)
    y_obs_trn:       (n_basins, n_times, 1)
    y_raw_trn:       (n_basins, n_times, 1)
    y_imputed_trn:   (n_basins, n_times, 1)   # optional: imputation prior
    times_trn:       (n_times,)               # 1-D shared time axis

Output per sample (8-tuple):
    batch_x:          (window, n_feat+1) or (window, n_feat+2) if y_imputed present
    batch_y:          (pred_len, 1)
    x_future:         (pred_len, n_feat) or (pred_len, n_feat+1) if y_imputed present
    origin_x:         (window, n_feat+1) or (window, n_feat+2)
    origin_y:         (pred_len, 1)
    batch_x_date_enc: (window, 4)
    batch_y_date_enc: (pred_len, 4)
    is_masked:        (1,)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from torch_timeseries.core import TimeSeriesDataset


def _datetime64_to_features(dt_array):
    """Convert datetime64 array to [month, day, weekday, hour] features."""
    dt_series = pd.to_datetime(dt_array)
    features = np.zeros((len(dt_array), 4), dtype=np.float32)
    features[:, 0] = dt_series.month       # 1-12
    features[:, 1] = dt_series.day         # 1-31
    features[:, 2] = dt_series.dayofweek   # 0-6
    features[:, 3] = dt_series.hour        # 0-23
    return features


class ForecastDataset(Dataset):
    """
    Single-layer sliding window dataset for forecasting.

    Index mapping:
        idx -> (basin_idx, time_offset)
        basin_idx  = idx // n_offsets
        time_offset = idx % n_offsets
    """

    def __init__(self, npz_path, split='train', window=168, pred_len=72,
                 masked_basins=None, mask_mode='noise', stride=1):
        self.window = window
        self.pred_len = pred_len
        self.mask_mode = mask_mode
        self.stride = stride

        data = np.load(npz_path, allow_pickle=True)

        split_key = {'train': 'trn', 'val': 'val', 'test': 'tst'}[split]

        self.x = data[f'x_{split_key}']            # (n_basins, n_times, n_feat)
        self.y = data[f'y_obs_{split_key}']         # (n_basins, n_times, 1)
        self.y_raw = data[f'y_raw_{split_key}']     # (n_basins, n_times, 1)
        self.times = data[f'times_{split_key}']     # (n_times,) 1-D

        # Y_imputed: imputation prior (optional)
        imputed_key = f'y_imputed_{split_key}'
        if imputed_key in data:
            self.y_imputed = data[imputed_key]      # (n_basins, n_times, 1)
            self.has_imputed = True
            print(f"  Y_imputed loaded: {self.y_imputed.shape}")
        else:
            self.has_imputed = False

        self.x_mean = data['x_mean']
        self.x_std = data['x_std']
        self.y_mean = data['y_mean']   # (n_basins,)
        self.y_std = data['y_std']     # (n_basins,)
        self.basin_names = data['basin_names']

        self.n_basins = self.x.shape[0]
        self.n_times = self.x.shape[1]
        max_offsets = self.n_times - window - pred_len + 1
        assert max_offsets > 0, (
            f"n_times ({self.n_times}) too short for "
            f"window ({window}) + pred_len ({pred_len})"
        )
        self.n_offsets = (max_offsets - 1) // stride + 1
        self.n_samples = self.n_basins * self.n_offsets

        # Masked basins
        self.masked_basin_ids = set()
        if masked_basins is not None:
            basin_to_idx = {name: i for i, name in enumerate(self.basin_names)}
            for item in masked_basins:
                if isinstance(item, (int, np.integer)):
                    if 0 <= item < self.n_basins:
                        self.masked_basin_ids.add(self.basin_names[item])
                else:
                    if item in basin_to_idx:
                        self.masked_basin_ids.add(item)

        print(f"ForecastDataset [{split}]:")
        print(f"  n_basins={self.n_basins}, n_times={self.n_times}, stride={self.stride}, "
              f"n_offsets={self.n_offsets}, total_samples={self.n_samples}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Same ordering as old pipeline: cycle basins first, then time offsets
        # So batch_size=n_basins gives all basins at the same time window
        time_offset = idx // self.n_basins
        basin_idx = idx % self.n_basins

        start = time_offset * self.stride
        mid = start + self.window
        end = mid + self.pred_len

        # Scaled data
        x_window = self.x[basin_idx, start:mid, :]       # (window, n_feat)
        x_future = self.x[basin_idx, mid:end, :]         # (pred_len, n_feat)
        y_history = self.y[basin_idx, start:mid, :]       # (window, 1)
        y_target = self.y[basin_idx, mid:end, :]          # (pred_len, 1)

        # Per-basin denorm params
        if self.y_mean.ndim == 0 or len(self.y_mean) == 1:
            y_mean = self.y_mean.item() if self.y_mean.ndim == 0 else self.y_mean[0]
            y_std_val = self.y_std.item() if self.y_std.ndim == 0 else self.y_std[0]
        else:
            y_mean = self.y_mean[basin_idx]
            y_std_val = self.y_std[basin_idx]

        # ---- Masking ----
        is_masked = False
        basin_id = self.basin_names[basin_idx]
        if basin_id in self.masked_basin_ids:
            is_masked = True
            if self.mask_mode == 'noise':
                y_history = np.random.randn(*y_history.shape).astype(np.float32)
            else:  # 'zero' or 'mean'
                y_history = np.zeros_like(y_history, dtype=np.float32)

        # ---- Condition = [X, (Y_imputed), Y_history] ----
        # Y_history stays last so batch_x[:, :, -1:] still works for g(x)
        if self.has_imputed:
            y_imp_hist = self.y_imputed[basin_idx, start:mid, :]   # (window, 1)
            y_imp_fut = self.y_imputed[basin_idx, mid:end, :]      # (pred_len, 1)
            batch_x_scaled = np.concatenate([x_window, y_imp_hist, y_history], axis=-1)
            x_future = np.concatenate([x_future, y_imp_fut], axis=-1)  # append to future
        else:
            batch_x_scaled = np.concatenate([x_window, y_history], axis=-1)
        batch_y_scaled = y_target

        # ---- Inverse standardise for origin_x / origin_y ----
        x_window_orig = x_window * (self.x_std + 1e-10) + self.x_mean
        y_history_orig = self.y[basin_idx, start:mid, :] * (y_std_val + 1e-10) + y_mean
        y_target_orig = y_target * (y_std_val + 1e-10) + y_mean

        if self.has_imputed:
            y_imp_hist_orig = y_imp_hist * (y_std_val + 1e-10) + y_mean
            origin_x = np.concatenate([x_window_orig, y_imp_hist_orig, y_history_orig], axis=-1)
        else:
            origin_x = np.concatenate([x_window_orig, y_history_orig], axis=-1)
        origin_y = y_target_orig

        # ---- Temporal encoding ----
        times_x = self.times[start:mid]
        times_y = self.times[mid:end]
        batch_x_date_enc = _datetime64_to_features(times_x)
        batch_y_date_enc = _datetime64_to_features(times_y)

        return (
            torch.FloatTensor(batch_x_scaled),
            torch.FloatTensor(batch_y_scaled),
            torch.FloatTensor(x_future),
            torch.FloatTensor(origin_x),
            torch.FloatTensor(origin_y),
            torch.FloatTensor(batch_x_date_enc),
            torch.FloatTensor(batch_y_date_enc),
            torch.BoolTensor([is_masked]),
        )


@dataclass
class ForecastMeta(TimeSeriesDataset):
    """Metadata class compatible with torch_timeseries framework."""
    npz_path: str = '../data_processing/data/prepped.npz'
    num_features: int = 43
    freq: str = 'yd'

    def __post_init__(self):
        data = np.load(self.npz_path, allow_pickle=True)
        has_imputed = 'y_imputed_trn' in data
        self.num_features = data['x_trn'].shape[2] + 1  # X + Y
        if has_imputed:
            self.num_features += 1  # + Y_imputed
        self.n_basins = int(data['n_segs'])
        del data


__all__ = ['ForecastMeta', 'ForecastDataset']
