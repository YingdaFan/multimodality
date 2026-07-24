"""
Custom DataLoader for CAMELS Dataset (X → Y Mode)

Difference from camels_loader.py: uses CAMELSNpzDatasetX (42-dim X only)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.datasets.camels_dataset_x import CAMELSNpzDatasetX


class CAMELSLoaderX:
    """
    Create train/val/test DataLoaders for CAMELS data (X → Y mode).

    Differences from CAMELSLoader:
    - Uses CAMELSNpzDatasetX (batch_x = 42-dim X only)
    - Designed for ungauged basins scenarios
    """

    def __init__(
        self,
        dataset,  # CAMELS_X dataset object (for compatibility)
        scaler=None,  # scaler (for compatibility)
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
        (Same as CAMELSLoader, parameters kept consistent for interface compatibility)
        """
        self.window = window
        self.horizon = horizon
        self.pred_len = steps
        self.num_worker = num_worker

        data = np.load(npz_path, allow_pickle=True)
        self.n_segs = int(data['n_segs'])
        del data

        if batch_size is None:
            print(f"  [X→Y] Setting batch_size = n_segs = {self.n_segs}")
            batch_size = self.n_segs
        elif batch_size != self.n_segs:
            print(f"  [X→Y] Using custom batch_size = {batch_size} (n_segs = {self.n_segs})")
        self.batch_size = batch_size

        # Create PyTorch Dataset (X → Y version)
        self.train_dataset = CAMELSNpzDatasetX(
            npz_path=npz_path,
            split='train',
            window=window,
            pred_len=steps,
            masked_basins=masked_basins,
            mask_mode=mask_mode
        )

        self.val_dataset = CAMELSNpzDatasetX(
            npz_path=npz_path,
            split='val',
            window=window,
            pred_len=steps,
            masked_basins=masked_basins,
            mask_mode=mask_mode
        )

        self.test_dataset = CAMELSNpzDatasetX(
            npz_path=npz_path,
            split='test',
            window=window,
            pred_len=steps,
            masked_basins=masked_basins,
            mask_mode=mask_mode
        )

        # Create DataLoader
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

        print(f"CAMELS DataLoader (X → Y mode) initialized:")
        print(f"  n_segs (stations): {self.n_segs}")
        print(f"  batch_size: {self.batch_size}")
        print(f"  shuffle_train: {shuffle_train}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Test batches: {len(self.test_loader)}")
