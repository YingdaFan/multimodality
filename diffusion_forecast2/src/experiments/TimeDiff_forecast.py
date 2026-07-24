
from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os
import torch
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.TimeDiff import TimeDiff
from src.experiments.prob_forecast import ProbForecastExp, update_metrics
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from torch.optim import *
from tqdm import tqdm
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
import torch.multiprocessing as mp
from torch_timeseries.utils.parse_type import parse_type
import numpy as np
import torch.distributed as dist
import torch
from tqdm import tqdm
import concurrent.futures
from types import SimpleNamespace

from src.datasets.forecast_dataset import ForecastMeta
from src.dataloader.forecast_loader import ForecastLoader


@dataclass
class TimeDiffParameters:
    num_samples: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.5
    num_steps: int = 100
    vis_ar_part: int = 0
    vis_MTS_analysis: int = 1
    schedule: str = "quad"
    use_window_normalization: bool = True
    t0: float = 1e-4
    T: float = 1
    nfe: int = 100
    dim_LSTM: int = 64
    UNet_Type: str = 'CNN'
    D3PM_kernel_size: int = 5
    use_freq_enhance: int = 0
    type_sampler: str = 'dpm'
    parameterization: str = 'x_start'
    ddpm_inp_embed: int = 256
    ddpm_dim_diff_steps: int = 256
    ddpm_channels_conv: int = 256
    ddpm_channels_fusion_I: int = 256
    ddpm_layers_inp: int = 5
    ddpm_layers_I: int = 5
    ddpm_layers_II: int = 5
    cond_ddpm_num_layers: int = 5
    cond_ddpm_channels_conv: int = 64
    ablation_study_case: str = "none"
    weight_pred_loss: float = 0.0
    ablation_study_F_type: str = "CNN"
    ablation_study_masking_type: str = "none"
    ablation_study_masking_tau: float = 0.9
    ot_ode: bool = True
    npz_path: str = '../data_processing/data/prepped.npz'


@dataclass
class TimeDiffForecast(ProbForecastExp, TimeDiffParameters):
    model_type: str = "TimeDiff_forecast"
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
        self.label_len = self.pred_len // 2
        # n_x_features: number of known future covariates (meteorological forcing)
        n_x_features = self.dataset.num_features - 1  # all except target Y

        args_dict = {
            "seq_len": self.windows,
            "device": self.device,
            "pred_len": self.pred_len,
            "label_len": self.label_len,
            "features": 'MS',
            "vis_ar_part": self.vis_ar_part,
            "vis_MTS_analysis": self.vis_MTS_analysis,
            "num_vars": self.dataset.num_features,  # 43: all features for conditioning
            "freq": self.dataset.freq,
            "interval": self.num_steps,
            "beta-max": self.beta_end,
            "use_window_normalization": self.use_window_normalization,
            "t0": self.t0,
            "T": self.T,
            "nfe": self.nfe,
            "dim_LSTM": self.dim_LSTM,
            "diff_steps": self.num_steps,
            "UNet_Type": self.UNet_Type,
            "D3PM_kernel_size": self.D3PM_kernel_size,
            "use_freq_enhance": self.use_freq_enhance,
            "type_sampler": self.type_sampler,
            "parameterization": self.parameterization,
            "ddpm_inp_embed": self.ddpm_inp_embed,
            "ddpm_dim_diff_steps": self.ddpm_dim_diff_steps,
            "ddpm_channels_conv": self.ddpm_channels_conv,
            "ddpm_channels_fusion_I": self.ddpm_channels_fusion_I,
            "ddpm_layers_inp": self.ddpm_layers_inp,
            "ddpm_layers_I": self.ddpm_layers_I,
            "ddpm_layers_II": self.ddpm_layers_II,
            "cond_ddpm_num_layers": self.cond_ddpm_num_layers,
            "cond_ddpm_channels_conv": self.cond_ddpm_channels_conv,
            "ablation_study_case": self.ablation_study_case,
            "weight_pred_loss": self.weight_pred_loss,
            "ablation_study_F_type": self.ablation_study_F_type,
            "ablation_study_masking_type": self.ablation_study_masking_type,
            "ablation_study_masking_tau": self.ablation_study_masking_tau,
            "ot-ode": self.ot_ode,
            "n_x_features": n_x_features,
        }

        self.args = SimpleNamespace(**args_dict)
        self.model = TimeDiff(self.args).to(self.device)

    def _train(self):
        self.model.train()

        with torch.enable_grad():
            train_loss = []
            for i, (
                batch_x,
                batch_y,
                x_future,
                origin_x,
                origin_y,
                batch_x_date_enc,
                batch_y_date_enc,
                is_masked,
            ) in enumerate(self.train_loader):
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                x_future = x_future.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                self.model_optim.zero_grad()
                loss = self._process_train_batch(
                    batch_x, batch_y, x_future, batch_x_date_enc, batch_y_date_enc
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)

                train_loss.append(loss.item())
                self.model_optim.step()

        self.model.eval()
        return train_loss

    def _process_train_batch(self, batch_x, batch_y, x_future, batch_x_date_enc, batch_y_date_enc):
        """
        Training: construct x_dec with known future X + target Y, then call TimeDiff.

        - batch_x:  (B, T, 43) = [X(42), Y_history(1)]
        - batch_y:  (B, O, 1)  = future streamflow target
        - x_future: (B, O, 42) = known future meteorological forcing
        """
        # Construct decoder input: [future_X, target_Y] → (B, O, 43)
        x_dec = torch.cat([x_future, batch_y], dim=-1)

        loss = self.model.train_forward(batch_x, batch_x_date_enc, x_dec, batch_y_date_enc)
        return loss

    def _process_val_batch(self, batch_x, batch_y, x_future, batch_x_date_enc, batch_y_date_enc):
        """
        Inference: future X known, Y unknown (zeros). Generate probabilistic samples.

        Output: preds (B, O, 1, S), batch_y_target (B, O, 1)
        """
        # Construct decoder input: [future_X, zeros_for_Y] → (B, O, 43)
        x_dec = torch.cat([
            x_future,
            torch.zeros([batch_x.size(0), self.pred_len, 1], device=self.device)
        ], dim=-1)

        outs, _, _, _, _ = self.model(
            batch_x, batch_x_date_enc, x_dec, batch_y_date_enc,
            None, None, None, self.num_samples
        )
        # outs: (B, S, O, 1) from TimeDiff with MS mode
        outs = outs.permute(0, 2, 3, 1)  # (B, O, 1, S)
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
    fire.Fire(TimeDiffForecast)
