"""
Custom DataLoader for CAMELS Dataset with Denoising Training

Key difference from camels_loader.py:
- Uses CAMELSNpzDatasetDenoise instead of CAMELSNpzDataset
- Supports denoising training parameters (noise_type, noise_std, etc.)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.datasets.camels_dataset_denoise import CAMELSNpzDatasetDenoise


class CAMELSLoaderDenoise:
    """
    DataLoader for CAMELS with denoising training capability.

    In Stage 2 calibration:
    - Training: Y_history is noised to prevent "copy" shortcut
    - Validation/Test: Y_history uses actual values (LSTM predictions for masked basins)
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
        mask_mode='noise',
        # Denoising parameters
        denoise_training=True,
        noise_type='gaussian',
        noise_std=0.3,
        dropout_rate=0.2
    ):
        """
        Parameters:
        -----------
        (Same as CAMELSLoader, plus denoising parameters)

        denoise_training : bool
            Whether to add noise to Y_history during training
        noise_type : str
            Type of noise: 'gaussian', 'uniform', 'dropout', 'lstm_like'
        noise_std : float
            Noise standard deviation (recommended: 0.1 ~ 0.5)
        dropout_rate : float
            Dropout rate for 'dropout' noise type
        """
        self.window = window
        self.horizon = horizon
        self.pred_len = steps
        self.num_worker = num_worker

        # Get number of segments
        data = np.load(npz_path, allow_pickle=True)
        self.n_segs = int(data['n_segs'])
        del data

        # Set batch size
        if batch_size is None:
            print(f"  Setting batch_size = n_segs = {self.n_segs}")
            batch_size = self.n_segs
        self.batch_size = batch_size

        # Create datasets with denoising
        print(f"\n{'='*50}")
        print("Creating CAMELS Datasets with Denoising Training")
        print(f"{'='*50}")

        # Training dataset: denoising enabled
        self.train_dataset = CAMELSNpzDatasetDenoise(
            npz_path=npz_path,
            split='train',
            window=window,
            pred_len=steps,
            masked_basins=masked_basins,
            mask_mode=mask_mode,
            denoise_training=denoise_training,  # Enable denoising
            noise_type=noise_type,
            noise_std=noise_std,
            dropout_rate=dropout_rate
        )

        # Validation dataset: denoising disabled (evaluate on clean data)
        self.val_dataset = CAMELSNpzDatasetDenoise(
            npz_path=npz_path,
            split='val',
            window=window,
            pred_len=steps,
            masked_basins=masked_basins,
            mask_mode=mask_mode,
            denoise_training=False,  # Disable for validation
            noise_type=noise_type,
            noise_std=noise_std,
            dropout_rate=dropout_rate
        )

        # Test dataset: denoising disabled
        self.test_dataset = CAMELSNpzDatasetDenoise(
            npz_path=npz_path,
            split='test',
            window=window,
            pred_len=steps,
            masked_basins=masked_basins,
            mask_mode=mask_mode,
            denoise_training=False,  # Disable for testing
            noise_type=noise_type,
            noise_std=noise_std,
            dropout_rate=dropout_rate
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

        print(f"\nCAMELS DataLoader (Denoise) initialized:")
        print(f"  n_segs: {self.n_segs}")
        print(f"  batch_size: {self.batch_size}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"  Test batches: {len(self.test_loader)}")
        print(f"{'='*50}\n")
