"""
TiDE forecast experiment — 把 TiDE 接入 ForecastLoader 流水线。

模型本体: src/models/TiDE.py
关键特点:
  • 显式使用 X_future（视为 futr_exog 协变量）
  • 全 MLP 架构，速度快，参数量小
  • 与 FutureTST 同样需要 batch_x + x_future

数据映射 (ForecastLoader 8-tuple → TiDE forward 输入)
  batch_x  : (B, window,   F+1)  → y_hist    = batch_x[:, :, -1:]            (B, window, 1)
                                    X_hist    = batch_x[:, :, :-1]            (B, window, F)
  x_future : (B, pred_len, F)    → futr_exog = cat([X_hist, x_future], dim=1) (B, window+pred_len, F)
  batch_y  : (B, pred_len, 1)    → MSE target

注：F = dataset.num_features - 1（has_imputed 情况下 num_features 已自动 +1，无需特判）
"""

from dataclasses import dataclass
from typing import Dict
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import *

from src.experiments.prob_forecast import ProbForecastExp, update_metrics
from src.datasets.forecast_dataset import ForecastMeta
from src.dataloader.forecast_loader import ForecastLoader
from src.models.TiDE import TiDE
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible


def enable_dropout(model):
    """Enable dropout layers during inference for MC Dropout."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@dataclass
class TiDEParameters:
    hidden_size: int = 512
    decoder_output_dim: int = 32
    temporal_decoder_dim: int = 128
    dropout: float = 0.3
    num_encoder_layers: int = 1
    num_decoder_layers: int = 1
    temporal_width: int = 4
    num_samples: int = 100
    npz_path: str = '../data_processing/data/prepped.npz'


@dataclass
class TiDEForecast(ProbForecastExp, TiDEParameters):
    model_type: str = "TiDE_forecast"
    dataset_type: str = "CAMELS"

    # ───────── Dataset / DataLoader ─────────

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
        self.train_loader = self.dataloader.train_loader
        self.val_loader = self.dataloader.val_loader
        self.test_loader = self.dataloader.test_loader

    # ───────── Model ─────────

    def _init_model(self):
        # F_exo = X + (Y_imp 如果 has_imputed) 的总通道数
        # batch_x 形状 (B, window, F_exo+1)，最后一列是 Y_history
        F_exo = self.dataset.num_features - 1
        self.model = TiDE(
            input_size=self.windows,
            h=self.pred_len,
            hist_exog_size=0,
            futr_exog_size=F_exo,
            stat_exog_size=0,
            hidden_size=self.hidden_size,
            decoder_output_dim=self.decoder_output_dim,
            temporal_decoder_dim=self.temporal_decoder_dim,
            dropout=self.dropout,
            layernorm=True,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            temporal_width=self.temporal_width,
            n_outputs=1,
        ).to(self.device)

    # ───────── 输入构造 ─────────

    @staticmethod
    def _build_inputs(batch_x: torch.Tensor, x_future: torch.Tensor):
        """
        batch_x  : (B, window,   F_exo+1)  last col = Y_history
        x_future : (B, pred_len, F_exo)
        →
        y_hist    : (B, window, 1)
        futr_exog : (B, window+pred_len, F_exo)
        """
        assert batch_x.shape[-1] - 1 == x_future.shape[-1], \
            (f"feature mismatch: batch_x has {batch_x.shape[-1]} cols (expect F_exo+1), "
             f"x_future has {x_future.shape[-1]} cols (expect F_exo).")
        Y_hist = batch_x[:, :, -1:]                                       # (B, L, 1)
        X_hist = batch_x[:, :, :-1]                                       # (B, L, F)
        futr_exog = torch.cat([X_hist, x_future], dim=1)                  # (B, L+h, F)
        return Y_hist, futr_exog

    # ───────── Training / Validation step ─────────

    def _process_train_batch(self, batch_x, batch_y, x_future):
        Y_hist, futr_exog = self._build_inputs(batch_x, x_future)
        pred = self.model(y_hist=Y_hist, futr_exog=futr_exog)             # (B, pred_len, 1)
        return F.mse_loss(pred, batch_y)

    def _process_val_batch(self, batch_x, batch_y, x_future, batch_x_date_enc, batch_y_date_enc):
        self.model.eval()
        enable_dropout(self.model)
        Y_hist, futr_exog = self._build_inputs(batch_x, x_future)
        samples = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                samples.append(self.model(y_hist=Y_hist, futr_exog=futr_exog))
        outs = torch.stack(samples, dim=-1)                                # (B, pred_len, 1, S)
        batch_y_target = batch_y[:, -self.pred_len:, :].to(self.device)
        return outs, batch_y_target

    # ───────── Train loop ─────────

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
                x_future = x_future.to(self.device).float()

                # Sample-level NaN filtering: drop basins with any NaN, keep the rest.
                nan_mask = (torch.isnan(batch_x).any(dim=(1, 2)) |
                            torch.isnan(batch_y).any(dim=(1, 2)) |
                            torch.isnan(x_future).any(dim=(1, 2)))
                if nan_mask.all():
                    continue
                if nan_mask.any():
                    valid = ~nan_mask
                    batch_x = batch_x[valid]
                    batch_y = batch_y[valid]
                    x_future = x_future[valid]

                self.model_optim.zero_grad()
                loss = self._process_train_batch(batch_x, batch_y, x_future)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                train_loss.append(loss.item())
                self.model_optim.step()

        self.model.eval()
        return train_loss

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

    # ───────── Test / save ─────────

    def _test_and_save(self):
        print("Testing and saving predictions...")
        self.model.eval()
        self.metrics.reset()

        all_preds, metric_results = [], []
        with torch.no_grad():
            for batch_x, batch_y, x_future, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc, *rest in self.test_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                x_future = x_future.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                # Sample-level NaN filtering with reassembly.
                nan_mask = (torch.isnan(batch_x).any(dim=(1, 2)) |
                            torch.isnan(batch_y).any(dim=(1, 2)) |
                            torch.isnan(x_future).any(dim=(1, 2)))
                full_size = batch_x.shape[0]
                if nan_mask.all():
                    nan_pred = np.full((full_size, self.pred_len, 1), np.nan, dtype=np.float32)
                    all_preds.append(nan_pred)
                    continue
                has_nan = nan_mask.any()
                if has_nan:
                    valid = ~nan_mask
                    batch_x = batch_x[valid]
                    batch_y = batch_y[valid]
                    x_future = x_future[valid]
                    batch_x_date_enc = batch_x_date_enc[valid]
                    batch_y_date_enc = batch_y_date_enc[valid]

                preds, truths = self._process_val_batch(
                    batch_x, batch_y, x_future, batch_x_date_enc, batch_y_date_enc
                )
                metric_results.append(self.task_pool.apply_async(
                    update_metrics,
                    (preds.contiguous().cpu().detach(),
                     truths.contiguous().cpu().detach(),
                     self.metrics)
                ))
                pred_mean = preds.mean(dim=-1).cpu().numpy()                # (valid, pred_len, 1)

                if has_nan:
                    full_pred = np.full((full_size, self.pred_len, 1), np.nan, dtype=np.float32)
                    valid_np = valid.cpu().numpy()
                    full_pred[valid_np] = pred_mean
                    pred_mean = full_pred

                all_preds.append(pred_mean)

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
    fire.Fire(TiDEForecast)
