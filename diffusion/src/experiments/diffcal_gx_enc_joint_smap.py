"""
Joint LSTM+Diffusion fine-tuning with the SMAP branch inside the graph.

Extends diffcal_gx_enc_joint.py:
1. The stage-1 model is LSTMWithSMAP (pixel-set encoder + LSTM), loaded from
   the stage-1 checkpoint and fine-tuned end-to-end (small LR).
2. On every forward pass the encoder produces the live embedding, which is
   (a) concatenated into the LSTM input to compute the prior and
   (b) concatenated into the calibrator's condition. Condition completeness
   therefore holds by construction; the augment bridge is not used.
3. Requires JOINT_SMAP=1 so the dataset returns per-sample indices, which
   the SMAPProvider maps to (basin, window start) for pixel lookup.

Condition layout per timestep: [X(51) | emb(32) | y_mean_vae | y_std_vae |
y_obs] — the y-related tail stays in the last three slots, as every
downstream consumer assumes.
"""
import os
import sys
import torch
import numpy as np
from dataclasses import dataclass
from torch.amp import autocast
from tqdm import tqdm

# the prior checkpoint is a pickled whole model (LSTMWithSMAP); unpickling
# imports its classes from `smap_encoder`/`model`, so the stage-1 dir must be
# on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PRIOR_DIR = os.path.join(os.path.dirname(_PROJECT_ROOT), 'lstm')
sys.path.insert(0, _PRIOR_DIR)
from smap_data import SMAPProvider

from src.experiments.diffcal_gx_enc_joint import DiffCalJoint
from src.experiments.prob_forecast import update_metrics


@dataclass
class DiffCalJointSmap(DiffCalJoint):
    model_type: str = "diffusion_gx_enc_joint_smap"
    smap_packed_path: str = '../data_processing/data/smap_packed.npz'

    # --- Model init: widened condition + LSTMWithSMAP + provider ---

    @property
    def condition_dim(self):
        # base features + live embedding: the ONE override that widens the
        # condition for every model and placeholder tensor; the embedding
        # width comes from the loaded prior
        return self.dataset.num_features + self.prior_model.d_smap

    def _init_model(self):
        # load the prior first: condition_dim (consulted while building the
        # diffusion models below) depends on the checkpoint's embedding width
        loaded = torch.load(self.prior_weights_path, map_location=self.device, weights_only=False)
        n_x = self.dataset.num_features - 3   # width of X (unwidened)
        if getattr(loaded, 'input_dim', n_x) != n_x:
            raise ValueError(f"prior input_dim={loaded.input_dim} != dataset X width {n_x}")
        self.prior_model = loaded.to(self.device)

        super(DiffCalJoint, self)._init_model()

        prepped = np.load(self.npz_path, allow_pickle=True)
        self.smap_provider = SMAPProvider(self.smap_packed_path)
        for split in ['trn', 'val', 'tst']:
            self.smap_provider.register_split(
                split, prepped[f'ids_{split}'], prepped[f'times_{split}'])
        if getattr(self.prior_model.smap_encoder, 'use_attrs', False):
            # the checkpoint decides whether the encoder is attr-conditioned
            self.smap_provider.set_static_attrs(prepped['basin_names'], prepped['static_enc'])
            print('[JointSmap] attribute-conditioned encoder ON (from checkpoint)')

        self._loader_split = {
            id(self.train_loader): 'trn',
            id(self.val_loader): 'val',
            id(self.test_loader): 'tst',
        }
        n_params = sum(p.numel() for p in self.prior_model.parameters())
        print(f"[JointSmap] prior: {type(loaded).__name__} from {self.prior_weights_path}, params={n_params}; "
              f"condition width {self.dataset.num_features}+{self.prior_model.d_smap}")

    # --- Core: live embedding into both the prior and the condition ---

    def _replace_y_obs_with_prior(self, batch_x, idx=None, split='trn'):
        x_features = batch_x[:, :, :-3]        # (B, T, 51)
        y_mean_vae = batch_x[:, :, -3:-2]
        y_std_vae = batch_x[:, :, -2:-1]

        smap_args = self.smap_provider.batch(split, idx, self.device)
        sm, xy, seg = smap_args[:3]
        attrs = smap_args[3] if len(smap_args) > 3 else None

        emb = self.prior_model.smap_encoder(
            sm, xy, seg, n_samples=x_features.shape[0], attrs=attrs)  # (B, T, 32)
        prior_out = self.prior_model.lstm(torch.cat([x_features, emb], dim=-1))

        y_denorm = prior_out * (y_std_vae + 1e-10) + y_mean_vae
        y_obs_global = (y_denorm - self.y_global_mean) / (self.y_global_std + 1e-10)

        return torch.cat([x_features, emb, y_mean_vae, y_std_vae, y_obs_global], dim=-1)

    # --- Training loop (8-tuple batches carry the sample index) ---

    def _train(self):
        self.model.train()
        self.cond_pred_model.train()
        self.prior_model.train()

        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            train_loss = []
            for i, (
                batch_x, batch_y, origin_x, origin_y,
                batch_x_date_enc, batch_y_date_enc, is_masked, idx,
            ) in enumerate(self.train_loader):
                origin_y = origin_y.to(self.device).float()
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                is_masked = is_masked.to(self.device).squeeze(-1)

                with autocast('cuda', dtype=torch.bfloat16):
                    batch_x = self._replace_y_obs_with_prior(batch_x, idx, 'trn')
                    loss = self._process_train_batch(
                        batch_x, batch_y, batch_x_date_enc, batch_y_date_enc, is_masked
                    )

                if torch.isnan(loss) or torch.isinf(loss):
                    self.model_optim.zero_grad()
                    progress_bar.update(batch_x.size(0))
                    continue

                self.grad_scaler.scale(loss).backward()
                progress_bar.update(batch_x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    diff_lr=self.model_optim.param_groups[0]["lr"],
                    prior_lr=self.model_optim.param_groups[2]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )
                self.grad_scaler.step(self.model_optim)
                self.grad_scaler.update()
                self.model_optim.zero_grad()

        self.model.eval()
        self.cond_pred_model.eval()
        self.prior_model.eval()
        return train_loss

    # --- Prediction with live embedding ---

    def _predict(self, loader, desc="Predicting"):
        self.model.eval()
        self.cond_pred_model.eval()
        self.prior_model.eval()
        split = self._loader_split.get(id(loader), 'tst')

        all_preds = []
        all_truths = []
        self.metrics.reset()
        metric_results = []

        with torch.no_grad():
            for (batch_x, batch_y, origin_x, origin_y,
                 batch_x_date_enc, batch_y_date_enc, is_masked, idx) in tqdm(loader, desc=desc):
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                batch_x = self._replace_y_obs_with_prior(batch_x, idx, split)

                preds, truths = self._process_val_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )

                preds_denorm = preds * self.y_global_std + self.y_global_mean
                metrics_truths = origin_y.unsqueeze(-1) if origin_y.dim() == 2 else origin_y
                metric_results.append(self.task_pool.apply_async(
                    update_metrics,
                    (preds_denorm.contiguous().cpu().detach(),
                     metrics_truths.contiguous().cpu().detach(),
                     self.metrics)
                ))

                pred_mean = preds.mean(dim=-1)
                pred_mean_denorm = pred_mean * self.y_global_std + self.y_global_mean
                all_preds.append(pred_mean_denorm.cpu().numpy())
                all_truths.append(origin_y.cpu().numpy())

        for r in metric_results:
            r.get()
        prob_metrics = {name: float(metric.compute()) for name, metric in self.metrics.items()}
        print(f"[{desc}] Probabilistic metrics (denormalized): {prob_metrics}")

        all_preds = np.concatenate(all_preds, axis=0)
        all_truths = np.concatenate(all_truths, axis=0)
        return all_preds, all_truths


if __name__ == "__main__":
    from src.utils.parse_args import parse_masked_basin_ids
    import fire

    globals()['_MASKED_BASIN_IDS'] = parse_masked_basin_ids()

    fire.Fire(DiffCalJointSmap)
