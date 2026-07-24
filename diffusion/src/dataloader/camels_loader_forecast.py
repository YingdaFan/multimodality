"""
DataLoader for CAMELSForecastDataset (single-layer sliding window).
"""

import numpy as np
from torch.utils.data import DataLoader
from src.datasets.camels_dataset_forecast import CAMELSForecastDataset


class CAMELSForecastLoader:
    """
    Create train/val/test DataLoaders for forecasting.

    Interface mirrors CAMELSLoader so NsDiff experiment code stays unchanged.
    """

    def __init__(
        self,
        dataset,
        scaler=None,
        window=168,
        horizon=1,
        steps=72,
        shuffle_train=False,
        freq='D',
        batch_size=None,
        num_worker=3,
        fast_test=False,
        fast_val=False,
        npz_path='../data_processing/data/prepped.npz',
        masked_basins=None,
        mask_mode='noise',
    ):
        self.window = window
        self.horizon = horizon
        self.pred_len = steps
        self.num_worker = num_worker

        data = np.load(npz_path, allow_pickle=True)
        self.n_segs = data['x_trn'].shape[0]
        del data

        if batch_size is None:
            print(f"  Setting batch_size = n_basins = {self.n_segs}")
            batch_size = self.n_segs
        self.batch_size = batch_size

        common_kw = dict(
            npz_path=npz_path,
            window=window,
            pred_len=steps,
            masked_basins=masked_basins,
            mask_mode=mask_mode,
        )

        self.train_dataset = CAMELSForecastDataset(split='train', **common_kw)
        self.val_dataset = CAMELSForecastDataset(split='val', **common_kw)
        self.test_dataset = CAMELSForecastDataset(split='test', **common_kw)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=shuffle_train, num_workers=num_worker, drop_last=False,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=num_worker, drop_last=False,
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=num_worker, drop_last=False,
        )

        print(f"CAMELSForecastLoader initialized:")
        print(f"  n_basins: {self.n_segs}, batch_size: {self.batch_size}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val   batches: {len(self.val_loader)}")
        print(f"  Test  batches: {len(self.test_loader)}")
