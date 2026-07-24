"""
Custom DataLoader for CAMELS RAW Pipeline

Key differences from camels_loader.py:
- Uses CAMELSNpzDatasetRaw which works with y_raw_* (original scale)
- Passes global Y statistics to ensure consistent normalization across splits
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.datasets.camels_dataset_raw import CAMELSNpzDatasetRaw


class CAMELSLoaderRaw:
    """
    DataLoader for CAMELS RAW pipeline.

    Key design:
    - Uses y_raw_* (original scale) instead of y_obs_* (per-basin normalized)
    - Computes global Y mean/std from training set
    - Passes same global stats to val/test for consistent normalization
    """

    def __init__(
        self,
        dataset,
        scaler=None,
        window=168,
        horizon=1,
        steps=192,
        shuffle_train=False,
        freq='D',
        batch_size=None,
        num_worker=3,
        fast_test=False,
        fast_val=False,
        npz_path='../data_processing/data/prepped.npz',
        masked_basins=None,
        mask_mode='noise'
    ):
        """
        Parameters:
        -----------
        (Same as CAMELSLoader)
        """
        self.window = window
        self.horizon = horizon
        self.pred_len = steps
        self.num_worker = num_worker

        # Get number of segments
        data = np.load(npz_path, allow_pickle=True)
        self.n_segs = int(data['n_segs'])

        # Compute global Y stats from training data
        y_raw_trn = data['y_raw_trn']
        y_flat = y_raw_trn.flatten()
        self.y_global_mean = float(np.nanmean(y_flat))
        self.y_global_std = float(np.nanstd(y_flat))
        del data

        print(f"\n{'='*60}")
        print("Creating CAMELS RAW Pipeline DataLoaders")
        print(f"{'='*60}")
        print(f"Global Y normalization: mean={self.y_global_mean:.4f}, std={self.y_global_std:.4f}")

        # Set batch size
        if batch_size is None:
            print(f"Setting batch_size = n_segs = {self.n_segs}")
            batch_size = self.n_segs
        self.batch_size = batch_size

        # Create datasets with SAME global Y stats
        self.train_dataset = CAMELSNpzDatasetRaw(
            npz_path=npz_path,
            split='train',
            window=window,
            pred_len=steps,
            masked_basins=masked_basins,
            mask_mode=mask_mode,
            y_global_mean=self.y_global_mean,
            y_global_std=self.y_global_std
        )

        self.val_dataset = CAMELSNpzDatasetRaw(
            npz_path=npz_path,
            split='val',
            window=window,
            pred_len=steps,
            masked_basins=masked_basins,
            mask_mode=mask_mode,
            y_global_mean=self.y_global_mean,
            y_global_std=self.y_global_std
        )

        self.test_dataset = CAMELSNpzDatasetRaw(
            npz_path=npz_path,
            split='test',
            window=window,
            pred_len=steps,
            masked_basins=masked_basins,
            mask_mode=mask_mode,
            y_global_mean=self.y_global_mean,
            y_global_std=self.y_global_std
        )

        # Create DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_train,
            num_workers=num_worker,
            drop_last=False
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_worker,
            drop_last=False
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_worker,
            drop_last=False
        )

        print(f"\nCAMELS DataLoader (RAW) initialized:")
        print(f"  n_segs: {self.n_segs}")
        print(f"  batch_size: {self.batch_size}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Test batches: {len(self.test_loader)}")
        print(f"{'='*60}\n")

    def get_global_stats(self):
        """Return global Y statistics for denormalization"""
        return self.y_global_mean, self.y_global_std
