"""
DataLoader for ForecastDataset (single-layer sliding window).
"""

import os
import numpy as np
from torch.utils.data import DataLoader
from src.datasets.forecast_dataset import ForecastDataset
from src.utils.meta import write_meta


class ForecastLoader:
    """
    Create train/val/test DataLoaders for forecasting.

    Create train/val/test DataLoaders from npz data.
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
        stride=24,
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
            stride=stride,
        )

        self.train_dataset = ForecastDataset(split='train', **common_kw)
        self.val_dataset = ForecastDataset(split='val', **common_kw)
        self.test_dataset = ForecastDataset(split='test', **common_kw)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=shuffle_train, num_workers=num_worker, drop_last=False,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=num_worker, drop_last=False,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=num_worker, drop_last=False,
            pin_memory=True,
        )

        print(f"ForecastLoader initialized:")
        print(f"  n_basins: {self.n_segs}, batch_size: {self.batch_size}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val   batches: {len(self.val_loader)}")
        print(f"  Test  batches: {len(self.test_loader)}")

        # Sidecar: 把本次训练实际用到的核心参数写到 output/pred/meta.json
        # postprocess 会优先读这里，避免 shell flag 与训练值脱钩
        _proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _pred_dir = os.path.join(_proj_root, 'output', 'pred')
        write_meta(
            _pred_dir,
            stride=int(stride),
            window=int(window),
            pred_len=int(steps),
            horizon=int(horizon),
            n_basins=int(self.n_segs),
            batch_size=int(self.batch_size),
        )
        print(f"  Sidecar meta.json -> {os.path.join(_pred_dir, 'meta.json')}")
