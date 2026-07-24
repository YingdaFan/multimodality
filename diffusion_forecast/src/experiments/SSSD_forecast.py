
from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os
import torch
import torch.nn.functional as F
from dataclasses import dataclass, asdict, field
from src.models.SSSD import SSSDSAImputer, calc_diffusion_hyperparams
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


def std_normal(size, device):
    """Generate standard Gaussian variable on the specified device."""
    return torch.normal(0, 1, size=size, device=device)


@dataclass
class SSSDParameters:
    beta_start: float = 0.0001
    beta_end: float = 0.02
    num_steps: int = 200
    num_samples: int = 100
    d_model: int = 128
    n_layers: int = 6
    pool: List[int] = field(default_factory=lambda: [2, 2])
    expand: int = 2
    ff: int = 2
    glu: bool = True
    unet: bool = True
    dropout: float = 0.02
    diffusion_step_embed_dim_in: int = 128
    diffusion_step_embed_dim_mid: int = 512
    diffusion_step_embed_dim_out: int = 512
    label_embed_dim: int = 128
    label_embed_classes: int = 71
    bidirectional: bool = True
    s4_lmax: int = 1000
    s4_d_state: int = 64
    s4_dropout: float = 0.00
    only_generate_missing: int = 1
    s4_bidirectional: bool = True
    npz_path: str = '../data_processing/data/prepped.npz'


@dataclass
class SSSDForecast(ProbForecastExp, SSSDParameters):
    model_type: str = "SSSD_forecast"
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

        self.model = SSSDSAImputer(
            d_model=self.d_model,
            n_layers=self.n_layers,
            pool=self.pool,
            expand=self.expand,
            ff=self.ff,
            glu=self.glu,
            unet=self.unet,
            dropout=self.dropout,
            in_channels=N,
            out_channels=N,
            diffusion_step_embed_dim_in=self.diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid=self.diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out=self.diffusion_step_embed_dim_out,
            label_embed_dim=self.label_embed_dim,
            label_embed_classes=self.label_embed_classes,
            bidirectional=self.bidirectional,
            s4_lmax=self.s4_lmax,
            s4_d_state=self.s4_d_state,
            s4_dropout=self.s4_dropout,
            s4_bidirectional=self.s4_bidirectional,
        ).to(self.device)

        self.diffu_params = calc_diffusion_hyperparams(self.num_steps, self.beta_start, self.beta_end)

        # gt_mask: True = known/given, False = to predict
        # History all known, future X known, future Y to predict
        gt_future = torch.zeros(self.pred_len, N)
        gt_future[:, :n_x] = 1.0
        self.gt_mask = torch.cat([
            torch.ones(self.windows, N),
            gt_future,
        ]).to(self.device).float()

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
        Training: forward diffusion on masked region, predict noise.
        - batch_x:  (B, T, 43)
        - batch_y:  (B, O, 1)
        - x_future: (B, O, 42)
        """
        B = batch_x.size(0)
        future_full = torch.cat([x_future, batch_y], dim=-1)  # (B, O, 43)
        # Full sequence: (B, T+O, 43) → transpose to (B, N, L)
        audio = torch.cat([batch_x, future_full], dim=1).transpose(1, 2)  # (B, 43, T+O)
        cond = audio.clone()
        mask = self.gt_mask.unsqueeze(0).expand(B, -1, -1).transpose(1, 2)  # (B, N, L), 1=known
        loss_mask = (1.0 - mask)  # (B, N, L), 1=to_predict

        T_steps = self.num_steps
        Alpha_bar = self.diffu_params["Alpha_bar"].to(self.device)

        diffusion_steps = torch.randint(T_steps, size=(B, 1, 1), device=self.device).long()
        z = std_normal(audio.shape, self.device)
        if self.only_generate_missing == 1:
            z = audio * mask + z * (1 - mask)
        transformed_X = (
            torch.sqrt(Alpha_bar[diffusion_steps]) * audio
            + torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z
        )
        epsilon_theta = self.model(
            (transformed_X, cond, mask, diffusion_steps.view(B, 1))
        )

        loss = F.mse_loss(epsilon_theta[loss_mask.bool()], z[loss_mask.bool()])
        return loss

    def _process_val_batch(self, batch_x, batch_y, x_future, batch_x_date_enc, batch_y_date_enc):
        """
        Inference: reverse diffusion with masking.
        Output: preds (B, O, 1, S), batch_y_target (B, O, 1)
        """
        B = batch_x.size(0)
        future_full = torch.cat([
            x_future,
            torch.zeros(B, self.pred_len, 1, device=self.device),
        ], dim=-1)
        full_seq = torch.cat([batch_x, future_full], dim=1).transpose(1, 2)  # (B, N, L)
        cond = full_seq.clone()
        mask = self.gt_mask.unsqueeze(0).expand(B, -1, -1).transpose(1, 2)  # (B, N, L)

        T_steps = self.num_steps
        Alpha = self.diffu_params["Alpha"].to(self.device)
        Alpha_bar = self.diffu_params["Alpha_bar"].to(self.device)
        Sigma = self.diffu_params["Sigma"].to(self.device)

        N_feat, L_len = full_seq.shape[1], full_seq.shape[2]
        size = (B * self.num_samples, N_feat, L_len)

        # Repeat for parallel sampling
        cond_rep = cond.repeat(self.num_samples, 1, 1)
        mask_rep = mask.repeat(self.num_samples, 1, 1)

        x = std_normal(size, self.device)

        with torch.no_grad():
            for t in range(T_steps - 1, -1, -1):
                if self.only_generate_missing == 1:
                    x = x * (1 - mask_rep) + cond_rep * mask_rep
                diffusion_steps = (t * torch.ones((size[0], 1), device=self.device))
                epsilon_theta = self.model((x, cond_rep, mask_rep, diffusion_steps))
                x = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
                if t > 0:
                    x = x + Sigma[t] * std_normal(size, self.device)

        # x: (B*S, N, L) → reshape and extract future Y
        x = x.reshape(B, self.num_samples, N_feat, L_len)
        # Extract Y (last feature) from future portion
        y_samples = x[:, :, -1:, -self.pred_len:]  # (B, S, 1, O)
        outs = y_samples.permute(0, 3, 2, 1)  # (B, O, 1, S)

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
    fire.Fire(SSSDForecast)
