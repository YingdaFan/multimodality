"""
Training / validation utilities tailored for the CoupledFMDiff model.

This is a self-contained parallel of `lstm/torch_utils.py::train_torch` with
the differences required by FM + Diffusion training:

  - Loss is computed via `model.compute_loss(x, y_true)` instead of
    `loss_fn(y_true, model(x))` — diffusion / flow-matching never produces a
    direct y_pred during training.

  - The total loss is the sum of three components: score + velocity + ODE
    consistency. We log them separately each epoch so degenerate behaviour
    (e.g. ODE term dominating, or score blowing up) is visible.

  - `lambda_ode` is ramped linearly from 0 → `lambda_ode_max` over the first
    `lambda_ode_ramp_frac` of training steps. This avoids the ODE consistency
    loss dominating before the two networks have learned anything useful.

  - Optional periodic `sample-based` evaluation: every `sample_eval_every`
    epochs we run the (slow) reverse-time ODE sample on a single val batch
    and report RMSE in the y_obs space. This is the only metric directly
    comparable to LSTM's RMSE.

NaN handling lives inside `model.compute_loss` (see CoupledFMDiff docstring).

Predict utility lives in `predict.py` (this file does not handle inference).
"""

import os
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.utils.data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_loader(x, y, batch_size, shuffle, pin_memory=True):
    if x is None:
        return None
    pairs = []
    for i in range(len(x)):
        pairs.append([torch.from_numpy(x[i]).float(),
                      torch.from_numpy(y[i]).float()])
    return torch.utils.data.DataLoader(
        pairs, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory
    )


def _lambda_ode_ramp(step, total_steps, max_val, ramp_frac):
    """Linear ramp 0 → max over the first ramp_frac of training steps."""
    if ramp_frac <= 0:
        return max_val
    ramp_steps = max(1, int(ramp_frac * total_steps))
    if step < ramp_steps:
        return max_val * (step / ramp_steps)
    return max_val


def _rmse_masked_np(y_pred, y_true):
    """Numpy RMSE that mirrors lstm/torch_utils.py::rmse_masked semantics."""
    valid = ~np.isnan(y_true)
    n = valid.sum()
    if n == 0:
        return float('nan')
    err = np.where(valid, y_pred - y_true, 0.0)
    return float(np.sqrt((err ** 2).sum() / n))


# ---------------------------------------------------------------------------
# Per-epoch loops with component logging
# ---------------------------------------------------------------------------
def train_loop_fmdiff(dataloader, model, optimizer, device,
                     lambda_ode_value, grad_clip=3.0):
    """
    One epoch of training.
    Returns dict with mean values of total/score/vel/ode loss.
    """
    agg = {'total': 0.0, 'score': 0.0, 'vel': 0.0, 'ode': 0.0, 'n': 0}
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        # All-NaN guard (matches lstm/torch_utils.py::train_loop)
        if torch.isnan(y).all():
            continue
        optimizer.zero_grad()
        loss, comps = model.compute_loss(
            x, y, lambda_ode=lambda_ode_value, return_components=True
        )
        if not torch.isfinite(loss):
            continue
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        agg['total'] += comps['total']
        agg['score'] += comps['score']
        agg['vel'] += comps['vel']
        agg['ode'] += comps['ode']
        agg['n'] += 1
    n = max(agg['n'], 1)
    return {k: agg[k] / n for k in ['total', 'score', 'vel', 'ode']}


@torch.no_grad()
def val_loop_fmdiff(dataloader, model, device, lambda_ode_value):
    """One epoch of validation. Returns mean component losses."""
    agg = {'total': 0.0, 'score': 0.0, 'vel': 0.0, 'ode': 0.0, 'n': 0}
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        if torch.isnan(y).all():
            continue
        _, comps = model.compute_loss(
            x, y, lambda_ode=lambda_ode_value, return_components=True
        )
        if not math.isfinite(comps['total']):
            continue
        agg['total'] += comps['total']
        agg['score'] += comps['score']
        agg['vel'] += comps['vel']
        agg['ode'] += comps['ode']
        agg['n'] += 1
    n = max(agg['n'], 1)
    return {k: agg[k] / n for k in ['total', 'score', 'vel', 'ode']}


@torch.no_grad()
def sample_rmse_fmdiff(dataloader, model, device, max_batches=1):
    """
    Optional slow eval: sample y_pred via reverse-ODE and compute RMSE in
    y_obs space. Provides a metric directly comparable to an LSTM's RMSE.

    Set max_batches=1 by default — sampling is expensive, one batch is
    usually enough for a quick gut-check during training.
    """
    rmses = []
    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break
        x = x.to(device)
        # forward() = sample
        y_pred = model(x).detach().cpu().numpy()
        y_true = y.numpy()
        rmses.append(_rmse_masked_np(y_pred, y_true))
    if not rmses:
        return float('nan')
    return float(np.nanmean(rmses))


# ---------------------------------------------------------------------------
# Top-level trainer
# ---------------------------------------------------------------------------
def train_torch_fmdiff(model,
                      optimizer,
                      x_train, y_train,
                      batch_size,
                      max_epochs,
                      early_stopping_patience=False,
                      x_val=None, y_val=None,
                      x_tst=None, y_tst=None,
                      shuffle=True,
                      weights_file=None,
                      log_file=None,
                      device='cpu',
                      grad_clip=3.0,
                      lambda_ode_max=1.0,
                      lambda_ode_ramp_frac=0.2,
                      sample_eval_every=0):
    """
    Train a CoupledFMDiff model. API mirrors lstm/torch_utils.py::train_torch
    but uses model.compute_loss instead of an external loss function.

    Args:
        model:                  CoupledFMDiff instance
        optimizer:              torch optimizer (pre-built)
        x_train, y_train:       numpy arrays, shape (N, L, F) and (N, L, 1)
        batch_size:             int
        max_epochs:             int
        early_stopping_patience: int or False
        x_val/y_val/x_tst/y_tst: optional eval splits
        shuffle:                shuffle training batches (recommended True;
                                early windows of streamflow data are often
                                all-NaN so unshuffled training wastes early
                                steps)
        weights_file:           where to save best-val checkpoint
        log_file:               where to save per-epoch csv log
        device:                 'cpu' or 'cuda:N'
        grad_clip:              gradient norm clip (set None to disable)
        lambda_ode_max:         max value of ODE consistency weight
        lambda_ode_ramp_frac:   ramp linearly 0 -> max over first frac of
                                total epochs (default 0.2 -> first 20%)
        sample_eval_every:      run sample-based RMSE every N epochs on val
                                (0 = disabled). EXPENSIVE — keep small N or 0.

    Returns:
        the trained model.
    """
    print(f"Training CoupledFMDiff on {device}", flush=True)
    print("start training...", flush=True)

    if not early_stopping_patience:
        early_stopping_patience = max_epochs

    # Build loaders
    train_loader = _build_loader(x_train, y_train, batch_size, shuffle=shuffle)
    val_loader = _build_loader(x_val, y_val, batch_size, shuffle=False) if x_val is not None else None
    tst_loader = _build_loader(x_tst, y_tst, batch_size, shuffle=False) if x_tst is not None else None

    model.to(device)

    log_cols = [
        'epoch',
        'train_total', 'train_score', 'train_vel', 'train_ode',
        'val_total', 'val_score', 'val_vel', 'val_ode',
        'tst_total', 'tst_score', 'tst_vel', 'tst_ode',
        'val_sample_rmse',
        'lambda_ode', 'train_time', 'val_time',
    ]
    train_log = pd.DataFrame(columns=log_cols)

    epochs_since_best = 0
    best_val = float('inf')

    train_times = []
    val_times = []

    for epoch in range(max_epochs):
        lam = _lambda_ode_ramp(epoch, max_epochs, lambda_ode_max, lambda_ode_ramp_frac)

        t1 = time.time()
        model.train()
        train_metrics = train_loop_fmdiff(
            train_loader, model, optimizer, device, lam, grad_clip
        )
        train_time = time.time() - t1
        train_times.append(train_time)

        print(
            f"Epoch {epoch+1}: "
            f"total={train_metrics['total']:.4f} "
            f"score={train_metrics['score']:.4f} "
            f"vel={train_metrics['vel']:.4f} "
            f"ode={train_metrics['ode']:.4f} "
            f"lam={lam:.3f} time={train_time:.1f}s",
            flush=True,
        )

        row = {
            'epoch': epoch,
            'train_total': train_metrics['total'],
            'train_score': train_metrics['score'],
            'train_vel': train_metrics['vel'],
            'train_ode': train_metrics['ode'],
            'val_total': np.nan, 'val_score': np.nan, 'val_vel': np.nan, 'val_ode': np.nan,
            'tst_total': np.nan, 'tst_score': np.nan, 'tst_vel': np.nan, 'tst_ode': np.nan,
            'val_sample_rmse': np.nan,
            'lambda_ode': lam,
            'train_time': train_time,
            'val_time': np.nan,
        }

        # Validation
        if val_loader is not None:
            model.eval()
            t2 = time.time()
            vm = val_loop_fmdiff(val_loader, model, device, lam)
            val_time = time.time() - t2
            val_times.append(val_time)
            row.update({
                'val_total': vm['total'], 'val_score': vm['score'],
                'val_vel': vm['vel'], 'val_ode': vm['ode'],
                'val_time': val_time,
            })
            print(
                f"  [val]  total={vm['total']:.4f} "
                f"score={vm['score']:.4f} "
                f"vel={vm['vel']:.4f} "
                f"ode={vm['ode']:.4f} "
                f"time={val_time:.1f}s",
                flush=True,
            )

            # Early stopping on val total
            if vm['total'] < best_val:
                best_val = vm['total']
                epochs_since_best = 0
                if weights_file is not None:
                    torch.save(model.state_dict(), weights_file)
            else:
                epochs_since_best += 1

            # Optional sample-based RMSE (slow)
            if sample_eval_every and (epoch + 1) % sample_eval_every == 0:
                rm = sample_rmse_fmdiff(val_loader, model, device, max_batches=1)
                row['val_sample_rmse'] = rm
                print(f"  [val sample RMSE] {rm:.4f}", flush=True)

            if epochs_since_best > early_stopping_patience:
                print(f"Early Stopping at Epoch {epoch}", flush=True)
                # Append final row before breaking
                train_log = pd.concat(
                    [train_log, pd.DataFrame([row], columns=log_cols)],
                    ignore_index=True
                )
                break

        # Test loop (informational only)
        if tst_loader is not None:
            model.eval()
            tm = val_loop_fmdiff(tst_loader, model, device, lam)
            row.update({
                'tst_total': tm['total'], 'tst_score': tm['score'],
                'tst_vel': tm['vel'], 'tst_ode': tm['ode'],
            })
            print(
                f"  [tst]  total={tm['total']:.4f} "
                f"score={tm['score']:.4f} "
                f"vel={tm['vel']:.4f} "
                f"ode={tm['ode']:.4f}",
                flush=True,
            )

        train_log = pd.concat(
            [train_log, pd.DataFrame([row], columns=log_cols)],
            ignore_index=True
        )

        # Periodic flush so log is visible during long runs
        if log_file is not None and (epoch + 1) % 5 == 0:
            train_log.to_csv(log_file, index=False)

    # Final save
    if log_file is not None:
        train_log.to_csv(log_file, index=False)
    if val_loader is None and weights_file is not None:
        torch.save(model.state_dict(), weights_file)

    if train_times:
        print(f"Average Training Time: {np.mean(train_times):.4f}s/epoch", flush=True)
    if val_times:
        print(f"Average Validation Time: {np.mean(val_times):.4f}s/epoch", flush=True)

    return model
