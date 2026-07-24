
from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os
import torch
from dataclasses import dataclass, asdict, field
from src.models.CSDI import CSDI_Forecasting
from src.experiments.prob_forecast import ProbForecastExp, update_metrics
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from torch.optim import *
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
import torch.multiprocessing as mp
import numpy as np
import torch
import concurrent.futures

from src.datasets.forecast_dataset import ForecastMeta
from src.dataloader.forecast_loader import ForecastLoader


@dataclass
class CSDIParameters:
    layers: int = 4
    channels: int = 64
    nheads: int = 8
    diffusion_embedding_dim: int = 128
    beta_start: float = 0.0001
    beta_end: float = 0.5
    num_steps: int = 50
    schedule: str = "quad"
    is_linear: bool = True
    is_unconditional: int = 0
    timeemb: int = 128
    featureemb: int = 16
    num_samples: int = 100
    target_strategy: str = "test"
    num_sample_features: int = 64
    npz_path: str = '../data_processing/data/prepped.npz'


@dataclass
class CSDIForecast(ProbForecastExp, CSDIParameters):
    model_type: str = "CSDI_forecast"
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
        n_x = N - 1  # 42 meteorological features

        configs = {
            "diffusion": {
                "layers": self.layers,
                "channels": self.channels,
                "nheads": self.nheads,
                "diffusion_embedding_dim": self.diffusion_embedding_dim,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "num_steps": self.num_steps,
                "schedule": self.schedule,
                "is_linear": self.is_linear,
            },
            "model": {
                "is_unconditional": self.is_unconditional,
                "timeemb": self.timeemb,
                "featureemb": self.featureemb,
                "target_strategy": self.target_strategy,
                "num_sample_features": min(self.num_sample_features, N),
            }
        }
        self.model = CSDI_Forecasting(
            config=configs,
            device=self.device,
            target_dim=N,
        ).to(self.device)

        # gt_mask: True = given/conditioning, False = to predict
        # History: all features given; Future: X given, Y to predict
        gt_future = torch.zeros(self.pred_len, N)
        gt_future[:, :n_x] = 1.0
        self.gt_mask = torch.cat([
            torch.ones(self.windows, N),
            gt_future,
        ]).to(self.device).float()

        # observation_mask: True where data exists (everywhere for our dataset)
        self.observation_mask = torch.ones(
            self.windows + self.pred_len, N
        ).to(self.device).float()

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
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                self.model_optim.zero_grad()
                noise, pred_noise = self._process_train_batch(
                    batch_x, batch_y, x_future, batch_x_date_enc, batch_y_date_enc
                )
                loss = self.loss_func(pred_noise, noise)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                train_loss.append(loss.item())
                self.model_optim.step()

        self.model.eval()
        return train_loss

    def _process_train_batch(self, batch_x, batch_y, x_future, batch_x_date_enc, batch_y_date_enc):
        """
        Training: full sequence with actual future Y values.
        - batch_x:  (B, T, 43) = [X(42), Y_history(1)]
        - batch_y:  (B, O, 1)  = future streamflow target
        - x_future: (B, O, 42) = known future meteorological forcing
        """
        B = batch_x.size(0)
        future_full = torch.cat([x_future, batch_y], dim=-1)  # (B, O, 43)
        observed_data = torch.cat([batch_x, future_full], dim=1)  # (B, T+O, 43)

        batch_input = {
            "observed_data": observed_data,
            "observed_mask": self.observation_mask.unsqueeze(0).expand(B, -1, -1),
            "timepoints": torch.cat([batch_x_date_enc, batch_y_date_enc], dim=1)[:, :, 0],
            "gt_mask": self.gt_mask.unsqueeze(0).expand(B, -1, -1),
        }
        noise, pred_noise = self.model(batch_input, is_train=self.num_samples)
        return noise, pred_noise

    def _process_val_batch(self, batch_x, batch_y, x_future, batch_x_date_enc, batch_y_date_enc):
        """
        Inference: future X known, future Y zeros. Generate probabilistic samples.
        Output: preds (B, O, 1, S), batch_y_target (B, O, 1)
        """
        B = batch_x.size(0)
        future_full = torch.cat([
            x_future,
            torch.zeros(B, self.pred_len, 1, device=self.device),
        ], dim=-1)  # (B, O, 43)
        observed_data = torch.cat([batch_x, future_full], dim=1)  # (B, T+O, 43)

        batch_input = {
            "observed_data": observed_data,
            "observed_mask": self.observation_mask.unsqueeze(0).expand(B, -1, -1),
            "timepoints": torch.cat([batch_x_date_enc, batch_y_date_enc], dim=1)[:, :, 0],
            "gt_mask": self.gt_mask.unsqueeze(0).expand(B, -1, -1),
        }

        samples, _, _, _, _ = self.model.evaluate(batch_input, self.num_samples, 1)
        # samples: (B, S, K=N, L=T+O), internally data is (B, K, L)
        # Extract future Y: last feature (index -1), last pred_len timesteps
        y_samples = samples[:, :, -1:, -self.pred_len:]  # (B, S, 1, O)
        y_samples = y_samples.permute(0, 3, 2, 1)  # (B, O, 1, S)

        assert (y_samples.shape[1], y_samples.shape[2], y_samples.shape[3]) == (
            self.pred_len, 1, self.num_samples), \
            f"Expected ({self.pred_len}, 1, {self.num_samples}), got {y_samples.shape[1:]}"

        batch_y_target = batch_y[:, -self.pred_len:, :].to(self.device)
        return y_samples, batch_y_target

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
    fire.Fire(CSDIForecast)
