
from dataclasses import dataclass, field
from typing import List, Dict
import os
import torch
import torch.nn.functional as F
from torch_timeseries.model import DLinear
from src.experiments.prob_forecast import ProbForecastExp, update_metrics
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from torch.optim import *
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
import torch.multiprocessing as mp
import numpy as np

from src.datasets.forecast_dataset import ForecastMeta
from src.dataloader.forecast_loader import ForecastLoader


class DLinearWithDropout(torch.nn.Module):
    """Wrap DLinear with a dropout layer for MC Dropout sampling."""

    def __init__(self, seq_len, pred_len, enc_in, dropout=0.1):
        super().__init__()
        self.dlinear = DLinear(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        out = self.dlinear(x)        # (B, O, N)
        out = self.dropout(out)
        return out


def enable_dropout(model):
    """Enable dropout layers during inference for MC Dropout."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@dataclass
class DLinearParameters:
    dropout: float = 0.1
    num_samples: int = 100
    npz_path: str = '../data_processing/data/prepped.npz'


@dataclass
class DLinearForecast(ProbForecastExp, DLinearParameters):
    model_type: str = "DLinear_forecast"
    dataset_type: str = "CAMELS"

    def _init_dataset(self):
        self.dataset = ForecastMeta(npz_path=self.npz_path)

    def _init_data_loader(self, shuffle=True, fast_test=True, fast_val=True):
        self._init_dataset()
        from torch_timeseries.scaler import StandardScaler
        self.scaler = StandardScaler()
        self.dataloader = ForecastLoader(
            dataset=self.dataset,
            scaler=self.scaler,
            window=self.windows,
            horizon=self.horizon,
            steps=self.pred_len,
            shuffle_train=shuffle,
            freq=self.dataset.freq,
            batch_size=None,
            num_worker=self.num_worker,
            fast_test=fast_test,
            fast_val=fast_val,
            npz_path=self.npz_path,
            masked_basins=None,
        )
        self.train_loader, self.val_loader, self.test_loader = (
            self.dataloader.train_loader,
            self.dataloader.val_loader,
            self.dataloader.test_loader,
        )

    def _init_model(self):
        N = self.dataset.num_features  # 43
        self.model = DLinearWithDropout(
            seq_len=self.windows,
            pred_len=self.pred_len,
            enc_in=N,
            dropout=self.dropout,
        ).to(self.device)

    def _train(self):
        self.model.train()
        with torch.enable_grad():
            train_loss = []
            for i, (
                batch_x, batch_y, x_future, origin_x, origin_y,
                batch_x_date_enc, batch_y_date_enc, is_masked,
            ) in enumerate(self.train_loader):
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()

                self.model_optim.zero_grad()
                loss = self._process_train_batch(batch_x, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                train_loss.append(loss.item())
                self.model_optim.step()

        self.model.eval()
        return train_loss

    def _process_train_batch(self, batch_x, batch_y):
        """
        batch_x: (B, T, 43)
        batch_y: (B, O, 1) - streamflow target
        DLinear is encode-only: input history, output all features, extract Y.
        """
        pred = self.model(batch_x)  # (B, O, 43)
        pred_y = pred[:, :, -1:]    # (B, O, 1) last feature = streamflow
        loss = F.mse_loss(pred_y, batch_y)
        return loss

    def _process_val_batch(self, batch_x, batch_y, x_future, batch_x_date_enc, batch_y_date_enc):
        """
        MC Dropout inference.
        Output: preds (B, O, 1, S), batch_y_target (B, O, 1)
        """
        self.model.eval()
        enable_dropout(self.model)

        samples = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                pred = self.model(batch_x)  # (B, O, 43)
                pred_y = pred[:, :, -1:]    # (B, O, 1)
                samples.append(pred_y)

        outs = torch.stack(samples, dim=-1)  # (B, O, 1, S)
        batch_y_target = batch_y[:, -self.pred_len:, :].to(self.device)
        return outs, batch_y_target

    def run(self, seed=42) -> Dict[str, float]:
        self._setup_run(seed)
        self._check_run_exist(seed)
        self._run_print(f"run : {self.current_run} in seed: {seed}")

        parameter_tables, model_parameters_num = count_parameters(self.model)
        self._run_print(f"parameter_tables: {parameter_tables}")
        self._run_print(f"model parameters: {model_parameters_num}")

        while self.current_epoch < self.epochs:
            epoch_start_time = time.time()
            if self.early_stopper.early_stop is True:
                self._run_print(
                    f"val loss no decreased for patience={self.patience} epochs, early stopping ...."
                )
                break

            reproducible(seed + self.current_epoch)
            train_losses = self._train()
            self._run_print(
                "Epoch: {} cost time: {}s".format(
                    self.current_epoch + 1, time.time() - epoch_start_time
                )
            )
            self._run_print(f"Training loss : {np.mean(train_losses)}")

            self.current_epoch += 1
            val_loss = np.mean(train_losses)
            self.early_stopper(val_loss, model=self.model)
            self._save_run_check_point(seed)

        self._load_best_model()
        best_test_result = self._test_and_save()
        return best_test_result

    def _test_and_save(self):
        """Test evaluation + save predictions."""
        print("Testing and saving predictions...")
        self.model.eval()
        self.metrics.reset()

        all_preds = []
        all_truths = []
        metric_results = []

        with torch.no_grad():
            for batch_x, batch_y, x_future, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, *rest in self.test_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                x_future = x_future.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                preds, truths = self._process_val_batch(
                    batch_x, batch_y, x_future, batch_x_date_enc, batch_y_date_enc
                )
                metric_results.append(self.task_pool.apply_async(
                    update_metrics,
                    (preds.contiguous().cpu().detach(),
                     truths.contiguous().cpu().detach(),
                     self.metrics)
                ))
                pred_mean = preds.mean(dim=-1)
                all_preds.append(pred_mean.cpu().numpy())
                all_truths.append(truths.cpu().numpy())

        for r in metric_results:
            r.get()
        test_result = {name: float(metric.compute()) for name, metric in self.metrics.items()}
        self._run_print(f"test_results: {test_result}")

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        pred_dir = os.path.join(project_root, 'output', 'pred')
        os.makedirs(pred_dir, exist_ok=True)
        all_preds = np.concatenate(all_preds, axis=0)
        pred_path = os.path.join(pred_dir, 'tst.npy')
        np.save(pred_path, all_preds)
        print(f"Predictions saved: {pred_path}, shape: {all_preds.shape}")
        return test_result


if __name__ == "__main__":
    import fire
    fire.Fire(DLinearForecast)
