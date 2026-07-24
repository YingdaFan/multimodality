"""
Custom DataLoader for CAMELS Dataset (Norm Version)

Differences from camels_loader.py:
- Uses CAMELSNpzDatasetNorm instead of CAMELSNpzDataset
- Returns 7 values (including is_masked) instead of 6
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.datasets.camels_dataset_norm import CAMELSNpzDatasetNorm


class CAMELSLoaderNorm:
    """
    Create train/val/test DataLoaders for CAMELS data (Norm version).

    Compatible with the ETTHLoader/ETTMLoader interface in the NsDiff framework.

    Key differences:
    - Input (Y_history): y_obs_* (LSTM predictions)
    - Label (Y_target): y_raw_* (true observations)
    - Additionally returns is_masked for loss masking
    """

    def __init__(
        self,
        dataset,  # CAMELS dataset object (for compatibility, not actually used)
        scaler=None,  # scaler (for compatibility, not used since data is pre-normalized)
        window=168,
        horizon=1,
        steps=192,
        shuffle_train=False,  # disabled by default to preserve station relationships
        freq='D',
        batch_size=None,  # None means auto-set to n_segs
        num_worker=3,
        fast_test=False,
        fast_val=False,
        npz_path='../data_processing/data/prepped.npz',
        masked_basins=None,  # list of basins to mask (indices or IDs)
        mask_mode='original'    # mask fill mode: 'original' (keep LSTM predictions), 'noise', 'zero'
    ):
        """
        Parameters:
        -----------
        dataset : TimeSeriesDataset
            Dataset object (for compatibility)
        scaler : Scaler
            Scaler (for compatibility, not actually used)
        window : int
            Input window length
        horizon : int
            Prediction start offset (typically 1)
        steps : int
            Prediction length (pred_len)
        shuffle_train : bool
            Whether to shuffle training set
        freq : str
            Time frequency
        batch_size : int
            Batch size
        num_worker : int
            Number of DataLoader workers
        fast_test : bool
            Fast test mode (reduces test set samples)
        fast_val : bool
            Fast validation mode (reduces validation set samples)
        npz_path : str
            Path to .npz data file
        masked_basins : list or None
            List of basins to exclude from loss, supports:
            - Basin ID (string): ['01022500', '02069700']
            - Basin Index (integer): [0, 100, 530]
            These basins still participate in forward pass, just don't contribute to loss
        mask_mode : str
            Processing mode for input Y_history (for masked basins):
            - 'original': keep original LSTM predictions (default, for correction learning)
            - 'noise': fill with standard Gaussian noise
            - 'zero': fill with 0
        """
        self.window = window
        self.horizon = horizon
        self.pred_len = steps
        self.num_worker = num_worker


        data = np.load(npz_path, allow_pickle=True)
        self.n_segs = int(data['n_segs'])
        del data


        if batch_size is None:
            print(f"  Setting batch_size = n_segs = {self.n_segs} (to preserve spatial relationships)")
            batch_size = self.n_segs
        elif batch_size != self.n_segs:
            print(f"  Using custom batch_size = {batch_size} (n_segs = {self.n_segs})")
        self.batch_size = batch_size



        # Create PyTorch Dataset (Norm version)
        self.train_dataset = CAMELSNpzDatasetNorm(
            npz_path=npz_path,
            split='train',
            window=window,
            pred_len=steps,
            masked_basins=masked_basins,
            mask_mode=mask_mode
        )

        self.val_dataset = CAMELSNpzDatasetNorm(
            npz_path=npz_path,
            split='val',
            window=window,
            pred_len=steps,
            masked_basins=masked_basins,
            mask_mode=mask_mode
        )

        self.test_dataset = CAMELSNpzDatasetNorm(
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

        print(f"CAMELS DataLoader (Norm) initialized:")
        print(f"  n_segs (stations): {self.n_segs}")
        print(f"  batch_size: {self.batch_size}")
        print(f"  shuffle_train: {shuffle_train}")
        print(f"  Train batches: {len(self.train_loader)} (each batch = all stations for one time window)")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Test batches: {len(self.test_loader)}")
        print(f"  Norm Pipeline:")
        print(f"    - Input: y_obs_* (LSTM predictions)")
        print(f"    - Label: y_raw_* (True observations)")
        print(f"    - Returns: 7 values (including is_masked)")
