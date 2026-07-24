
from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os
import torch
from dataclasses import dataclass, asdict, field
from src.models.DiffusionTS import Diffusion_TS
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
class DiffusionTSParameters:
    num_samples: int = 100
    n_layer_enc: int = 3
    n_layer_dec: int = 6
    d_model: int = 64
    timesteps: int = 100
    sampling_timesteps: int = 100
    loss_type: str = 'l1'
    beta_schedule: str = 'cosine'
    n_heads: int = 4
    mlp_hidden_times: int = 4
    eta: float = 0.
    attn_pd: float = 0.
    resid_pd: float = 0.
    kernel_size: int = 1
    padding_size: int = 0
    use_ff: bool = True
    reg_weight: float = None
    npz_path: str = '../data_processing/data/prepped.npz'


@dataclass
class DiffusionTSForecast(ProbForecastExp, DiffusionTSParameters):
    model_type: str = "DiffusionTS_forecast"
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
        n_x = N - 1  # 42

        self.model = Diffusion_TS(
            seq_length=self.windows + self.pred_len,
            feature_size=N,
            n_layer_enc=self.n_layer_enc,
            n_layer_dec=self.n_layer_dec,
            d_model=self.d_model,
            timesteps=self.timesteps,
            sampling_timesteps=self.sampling_timesteps,
            loss_type='l2',
            beta_schedule=self.beta_schedule,
            n_heads=self.n_heads,
            mlp_hidden_times=self.mlp_hidden_times,
            eta=self.eta,
            attn_pd=self.attn_pd,
            resid_pd=self.resid_pd,
            kernel_size=self.kernel_size,
            padding_size=self.padding_size,
            use_ff=self.use_ff,
            reg_weight=self.reg_weight,
        ).to(self.device)

        # gt_mask: True = known/given, False = to predict
        gt_future = torch.zeros(self.pred_len, N)
        gt_future[:, :n_x] = 1.0  # future X known
        self.gt_mask = torch.cat([
            torch.ones(self.windows, N),
            gt_future,
        ]).to(self.device).bool()

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

                self.model_optim.zero_grad()
                loss = self._process_train_batch(batch_x, batch_y, x_future)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                train_loss.append(loss.item())
                self.model_optim.step()

        self.model.eval()
        return train_loss

    def _process_train_batch(self, batch_x, batch_y, x_future):
        """
        Training: DiffusionTS takes full sequence and computes loss internally.
        - batch_x:  (B, T, 43)
        - batch_y:  (B, O, 1)
        - x_future: (B, O, 42)
        """
        future_full = torch.cat([x_future, batch_y], dim=-1)  # (B, O, 43)
        data = torch.cat([batch_x, future_full], dim=1)  # (B, T+O, 43)
        loss = self.model(data, target=data)
        return loss

    def _process_val_batch(self, batch_x, batch_y, x_future, batch_x_date_enc, batch_y_date_enc):
        """
        Inference: conditional infilling with known history + future X.
        Output: preds (B, O, 1, S), batch_y_target (B, O, 1)
        """
        B = batch_x.size(0)
        future_full = torch.cat([
            x_future,
            torch.zeros(B, self.pred_len, 1, device=self.device),
        ], dim=-1)  # (B, O, 43)
        x = torch.cat([batch_x, future_full], dim=1)  # (B, T+O, 43)
        t_m = self.gt_mask.unsqueeze(0).expand(B, -1, -1)

        coef = 1e-1
        stepsize = 5e-2
        model_kwargs = {'coef': coef, 'learning_rate': stepsize}
        minisample = 10

        samples = []
        for _ in range(self.num_samples // minisample):
            repeat_x = x.repeat(minisample, 1, 1)
            repeat_t_m = t_m.repeat(minisample, 1, 1)

            with torch.no_grad():
                sample = self.model.fast_sample_infill(
                    shape=repeat_x.shape,
                    target=repeat_x * repeat_t_m,
                    partial_mask=repeat_t_m,
                    model_kwargs=model_kwargs,
                    sampling_timesteps=self.sampling_timesteps,
                )
            # sample: (B*mini, T+O, 43) → extract future Y
            sample_y = sample[:, -self.pred_len:, -1:]  # (B*mini, O, 1)
            sample_y = sample_y.reshape(B, minisample, self.pred_len, 1)
            samples.append(sample_y)

        samples = torch.cat(samples, dim=1)  # (B, S, O, 1)
        outs = samples.permute(0, 2, 3, 1)  # (B, O, 1, S)

        assert (outs.shape[1], outs.shape[2], outs.shape[3]) == (
            self.pred_len, 1, self.num_samples), \
            f"Expected ({self.pred_len}, 1, {self.num_samples}), got {outs.shape[1:]}"

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
    fire.Fire(DiffusionTSForecast)
