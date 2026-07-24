"""
Inference utilities for the CoupledFMDiff model.

Self-contained parallel of `lstm/predict.py`. Same output format
(.npy in shape `(n_samples, seq_len, 1)`) so anything downstream that
reads those files (e.g. fill_prepped_npz_raw.py, postprocess scripts)
works unchanged.

Differences vs LSTM's predict path:

  - Forward pass is a reverse-time ODE sample, not a single feed-forward.
    We accept `num_steps` and `method` knobs so the user can trade
    accuracy for inference speed.

  - Optional `n_samples` enables ensemble inference: average over multiple
    stochastic samples to reduce variance, or stack to compute
    uncertainty (CRPS, percentile bands, etc.)

  - No `unscale_output` here; the wrapper outputs predictions in
    normalized y space, exactly like LSTM. Denormalization happens
    downstream (postprocess_perseg_aligntime.py, etc.).
"""

from pathlib import Path
import numpy as np
from numpy.lib.npyio import NpzFile
import torch
import torch.utils.data


def _get_data_if_file(d):
    """Load .npz lazily if a path is passed; otherwise pass-through."""
    if isinstance(d, NpzFile) or isinstance(d, dict):
        return d
    return np.load(d, allow_pickle=True)


# ---------------------------------------------------------------------------
# Batched inference: invoke model(x) which internally samples via reverse ODE
# ---------------------------------------------------------------------------
def predict_torch_fmdiff(x_data, model, batch_size,
                       num_steps=None, method=None,
                       n_samples=1, device=None):
    """
    Args:
        x_data:     numpy array (N, L, F)
        model:      CoupledFMDiff in eval mode
        batch_size: int
        num_steps:  override model.sample_steps (None = use model default)
        method:     'vel' | 'score' | 'avg' (None = use model default)
        n_samples:  if >1, draw n_samples and return their MEAN (variance-reduced).
                    Each draw is a fresh reverse-ODE rollout.
        device:     cpu / cuda; if None, auto-detect.

    Returns:
        torch.Tensor of shape (N, L, 1) — predictions in normalized y space.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    # Allow runtime overrides without permanently changing model attrs
    saved_steps = model.sample_steps
    saved_method = model.sample_method
    if num_steps is not None:
        model.sample_steps = int(num_steps)
    if method is not None:
        model.sample_method = method

    data = [torch.from_numpy(x_data[i]).float() for i in range(len(x_data))]
    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    try:
        all_preds = []
        with torch.no_grad():
            for x in loader:
                x = x.to(device)
                if n_samples == 1:
                    y_pred = model(x).detach().cpu()
                else:
                    accum = None
                    for _ in range(n_samples):
                        y = model(x).detach().cpu()
                        accum = y if accum is None else (accum + y)
                    y_pred = accum / n_samples
                all_preds.append(y_pred)
        return torch.cat(all_preds, dim=0)
    finally:
        # restore
        model.sample_steps = saved_steps
        model.sample_method = saved_method


# ---------------------------------------------------------------------------
# Mid-level: predict and (optionally) save .npy in LSTM-compatible format
# ---------------------------------------------------------------------------
def predict_fmdiff(model, x_data, pred_ids, pred_dates,
                  y_stds, y_means, y_vars,
                  keep_last_portion=1, outfile=None,
                  num_steps=None, method=None, n_samples=1,
                  batch_size=None, device=None,
                  pad_mask=None):
    """
    Drop-in replacement for `lstm/predict.py::predict()` adapted for FMDiff.

    Inputs and outputs match the LSTM version where possible:
      x_data:        (N, L, F) numpy
      pred_ids:      (N, L, 1) basin id strings
      pred_dates:    (N, L, 1) datetime64 timestamps
      y_stds, y_means: per-basin scaling parameters (NOT used here; kept in
                       the signature for API compatibility — denormalization
                       is performed downstream by postprocess scripts)
      y_vars:        list of target variable names (typically ['Q_camelsh_obs_norm'])
      keep_last_portion: float in (0,1] OR int (number of timesteps to retain
                         from the END of each window). Same semantics as LSTM.
      outfile:       if given, save predictions as `<outfile>.npy`
      num_steps:     override model.sample_steps
      method:        override model.sample_method
      n_samples:     >=1; if >1 returns the MEAN over multiple ODE rollouts
      batch_size:    if None, infer from unique-ids count (matches LSTM)
      device:        cpu / cuda; auto if None
      pad_mask:      ignored (FMDiff has no graph dim) — kept for API parity.

    Returns:
        numpy array (N, kept_seq, 1) of predictions in normalized y space.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")

    if batch_size is None:
        batch_size = len(np.unique(pred_ids))

    y_pred = predict_torch_fmdiff(
        x_data, model, batch_size=batch_size,
        num_steps=num_steps, method=method,
        n_samples=n_samples, device=device,
    )  # torch.Tensor (N, L, 1)

    if keep_last_portion > 1:
        frac_seq_len = int(keep_last_portion)
    else:
        frac_seq_len = round(pred_ids.shape[1] * keep_last_portion)

    y_pred = y_pred[:, -frac_seq_len:, ...]
    y_pred_np = y_pred.numpy()

    if outfile:
        outpath = Path(outfile).with_suffix('.npy')
        np.save(outpath, y_pred_np)
        print(f"FMDiff predictions saved to {outpath}")

    print(f"y_pred shape: {y_pred_np.shape}")
    return y_pred_np


# ---------------------------------------------------------------------------
# Top-level: read partition from prepped.npz and invoke predict_fmdiff
# ---------------------------------------------------------------------------
def predict_fmdiff_from_io_data(
        model,
        io_data,
        partition,
        outfile=None,
        trn_offset=1.0,
        tst_val_offset=1.0,
        num_steps=None,
        method=None,
        n_samples=1,
        batch_size=None,
        device=None,
):
    """
    Top-level inference entry analogous to
    `lstm/predict.py::predict_from_io_data()`.

    io_data:   path to `prepped.npz` OR an already-loaded NpzFile / dict
    partition: one of 'trn' / 'val' / 'tst'
    outfile:   path; suffix `.npy` is appended if missing
    trn_offset / tst_val_offset:  inherited from preprocess
                                  (controls keep_last_portion)
    """
    io = _get_data_if_file(io_data)

    if partition == 'trn':
        keep_portion = trn_offset
    elif partition in ('val', 'tst'):
        keep_portion = tst_val_offset
    else:
        raise ValueError(f"partition must be 'trn'/'val'/'tst', got {partition!r}")

    return predict_fmdiff(
        model=model,
        x_data=io[f'x_{partition}'],
        pred_ids=io[f'ids_{partition}'],
        pred_dates=io[f'times_{partition}'],
        y_stds=io['y_std'],
        y_means=io['y_mean'],
        y_vars=io['y_obs_vars'],
        keep_last_portion=keep_portion,
        outfile=outfile,
        num_steps=num_steps,
        method=method,
        n_samples=n_samples,
        batch_size=batch_size,
        device=device,
        pad_mask=io.get(f'padded_{partition}', None),
    )
