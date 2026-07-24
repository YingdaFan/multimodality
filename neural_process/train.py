"""
Training and evaluation script for Neural Process baselines (ANP / GNP).

Loads the same prepped.npz used by ZeroDiff, trains an NP model with
leave-one-out episodes on observed basins, and saves predictions in the
identical format expected by postprocess_perseg_aligntime_raw.py.

Usage:
    python train.py --model_type anp --npz_path ../data_processing/data/prepped.npz \
                    --masked_basins 01022500 02069700 --epochs 200
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from models import (ConditionalNeuralProcess, NeuralProcess,
                     TransformerNeuralProcess,
                     AttentiveNeuralProcess, GraphNeuralProcess)


# ============================================================
# Data loading
# ============================================================

def load_split(npz_path, split, y_global_mean=None, y_global_std=None):
    """
    Load one split from the NPZ and reshape to (n_windows, n_basins, ...).

    Returns a dict with numpy arrays (kept on CPU; moved to GPU per-batch).
    """
    data = np.load(npz_path, allow_pickle=True)
    n_basins = int(data['n_segs'])
    basin_names = data['basin_names']

    suffix = {'train': 'trn', 'val': 'val', 'test': 'tst'}[split]
    x = data[f'x_{suffix}']          # (n_samples, seq_len, x_dim)
    y_raw = data[f'y_raw_{suffix}']   # (n_samples, seq_len, 1)

    seq_len, x_dim = x.shape[1], x.shape[2]
    n_windows = x.shape[0] // n_basins
    assert x.shape[0] == n_windows * n_basins, "n_samples must be divisible by n_basins"

    x = x.reshape(n_windows, n_basins, seq_len, x_dim)
    y_raw = y_raw.reshape(n_windows, n_basins, seq_len, 1)

    # Global Y stats (compute from training split only)
    if y_global_mean is None or y_global_std is None:
        y_trn = data['y_raw_trn'].flatten()
        y_global_mean = float(np.nanmean(y_trn))
        y_global_std = float(np.nanstd(y_trn))

    y_norm = (y_raw - y_global_mean) / (y_global_std + 1e-10)

    dist_matrix = data['dist_matrix'] if 'dist_matrix' in data.files else None

    return {
        'x': x.astype(np.float32),
        'y_raw': y_raw.astype(np.float32),
        'y_norm': y_norm.astype(np.float32),
        'basin_names': basin_names,
        'n_basins': n_basins,
        'n_windows': n_windows,
        'seq_len': seq_len,
        'x_dim': x_dim,
        'y_global_mean': y_global_mean,
        'y_global_std': y_global_std,
        'dist_matrix': dist_matrix.astype(np.float32) if dist_matrix is not None else None,
    }


def get_basin_indices(basin_names, masked_basins):
    name2idx = {str(name): idx for idx, name in enumerate(basin_names)}
    masked = [name2idx[b] for b in masked_basins if b in name2idx]
    non_masked = sorted(set(range(len(basin_names))) - set(masked))
    return np.array(masked, dtype=np.int64), np.array(non_masked, dtype=np.int64)


# ============================================================
# Training helpers
# ============================================================

def _fill_nan(y):
    """Replace NaN with 0 (neutral in normalised space). Return (filled, valid_mask)."""
    nan = torch.isnan(y)
    y_filled = y.clone()
    y_filled[nan] = 0.0
    return y_filled, ~nan


def train_epoch(model, data, non_masked_idx, device, optimizer,
                context_ratio=0.8, beta_kl=1.0, dist_matrix_np=None, is_gnp=False,
                max_context=64, max_target=32):
    """
    One training epoch.  To fit in GPU memory, we sub-sample at most
    `max_context` context locations and `max_target` target locations
    per episode (standard practice in NP training).
    """
    model.train()
    losses = []
    n_obs = len(non_masked_idx)
    x_all, y_all = data['x'], data['y_norm']

    for win in np.random.permutation(data['n_windows']):
        # Random context / target split among observed basins
        perm = np.random.permutation(n_obs)
        n_ctx = min(max(int(n_obs * context_ratio), 1), max_context)
        n_tgt = min(n_obs - n_ctx, max_target)
        if n_tgt <= 0:
            continue
        ctx_basins = non_masked_idx[perm[:n_ctx]]
        tgt_basins = non_masked_idx[perm[n_ctx:n_ctx + n_tgt]]
        if len(tgt_basins) == 0:
            continue

        ctx_x = torch.from_numpy(x_all[win, ctx_basins]).to(device)
        ctx_y = torch.from_numpy(y_all[win, ctx_basins]).to(device)
        tgt_x = torch.from_numpy(x_all[win, tgt_basins]).to(device)
        tgt_y = torch.from_numpy(y_all[win, tgt_basins]).to(device)

        ctx_y, _ = _fill_nan(ctx_y)
        tgt_y_filled, tgt_valid = _fill_nan(tgt_y)

        # Forward
        kw = {}
        if is_gnp and dist_matrix_np is not None:
            kw['dist_matrix'] = torch.from_numpy(dist_matrix_np).to(device)
            kw['ctx_indices'] = torch.from_numpy(ctx_basins).to(device)
            kw['tgt_indices'] = torch.from_numpy(tgt_basins).to(device)

        y_pred, kl = model(ctx_x, ctx_y, tgt_x, tgt_y_filled, **kw)

        # MSE on valid (non-NaN) time steps only
        if tgt_valid.any():
            mse = ((y_pred - tgt_y_filled) ** 2 * tgt_valid.float()).sum() / tgt_valid.float().sum()
        else:
            mse = torch.tensor(0.0, device=device)

        loss = mse + beta_kl * kl
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses)) if losses else 0.0


def _subsample_context(non_masked_idx, max_ctx, rng=None):
    """Subsample context indices if there are too many (for TNP-D consistency)."""
    if max_ctx is not None and len(non_masked_idx) > max_ctx:
        if rng is not None:
            chosen = rng.choice(len(non_masked_idx), max_ctx, replace=False)
        else:
            chosen = np.random.choice(len(non_masked_idx), max_ctx, replace=False)
        return non_masked_idx[np.sort(chosen)]
    return non_masked_idx


@torch.no_grad()
def evaluate(model, data, masked_idx, non_masked_idx, device,
             dist_matrix_np=None, is_gnp=False, max_context_infer=None,
             n_eval_samples=5):
    """
    MSE on masked basins (normalised space).

    For models sensitive to context size (TNP-D), we average over
    `n_eval_samples` random context subsamples for robustness.
    """
    model.eval()
    total_se, total_n = 0.0, 0
    x_all, y_all = data['x'], data['y_norm']
    rng = np.random.RandomState(0)  # deterministic eval

    n_repeats = n_eval_samples if max_context_infer is not None else 1

    for win in range(data['n_windows']):
        tgt_x = torch.from_numpy(x_all[win, masked_idx]).to(device)
        tgt_y = torch.from_numpy(y_all[win, masked_idx]).to(device)
        _, tgt_valid = _fill_nan(tgt_y)
        if not tgt_valid.any():
            continue

        win_se = 0.0
        for _ in range(n_repeats):
            ctx_idx = _subsample_context(non_masked_idx, max_context_infer, rng)
            ctx_x = torch.from_numpy(x_all[win, ctx_idx]).to(device)
            ctx_y = torch.from_numpy(y_all[win, ctx_idx]).to(device)
            ctx_y, _ = _fill_nan(ctx_y)

            kw = {}
            if is_gnp and dist_matrix_np is not None:
                kw['dist_matrix'] = torch.from_numpy(dist_matrix_np).to(device)
                kw['ctx_indices'] = torch.from_numpy(ctx_idx).to(device)
                kw['tgt_indices'] = torch.from_numpy(masked_idx).to(device)

            y_pred, _ = model(ctx_x, ctx_y, tgt_x, **kw)
            win_se += ((y_pred - tgt_y) ** 2 * tgt_valid.float()).sum().item()

        total_se += win_se / n_repeats
        total_n += tgt_valid.float().sum().item()

    return total_se / max(total_n, 1)


# ============================================================
# Prediction (full output for postprocessing)
# ============================================================

@torch.no_grad()
def predict_all(model, data, masked_idx, non_masked_idx, device,
                dist_matrix_np=None, is_gnp=False, max_context_infer=None,
                n_pred_samples=10, crossval_k=5):
    """
    Produce predictions for ALL basins in time-major order.

    - Masked basins:     context = all non-masked → NP prediction.
    - Non-masked basins: K-fold cross-prediction (leave-out) so they also
      carry realistic NP errors — critical for downstream diffusion calibration.

    Returns: (n_samples, seq_len, 1) in original scale.
    """
    model.eval()
    x_all, y_norm_all, y_raw_all = data['x'], data['y_norm'], data['y_raw']
    y_mean, y_std = data['y_global_mean'], data['y_global_std']
    n_win, n_bas, seq_len = data['n_windows'], data['n_basins'], data['seq_len']
    rng = np.random.RandomState(42)

    out = np.zeros_like(y_raw_all)  # will be filled entirely with predictions
    n_repeats = n_pred_samples if max_context_infer is not None else 1

    # --- Assign non-masked basins to K folds for cross-prediction ---
    n_obs = len(non_masked_idx)
    fold_ids = np.arange(n_obs) % crossval_k  # deterministic assignment

    for win in range(n_win):
        # (A) Predict MASKED basins: context = all non-masked
        tgt_x = torch.from_numpy(x_all[win, masked_idx]).to(device)
        accum = np.zeros((len(masked_idx), seq_len, 1), dtype=np.float32)

        for _ in range(n_repeats):
            ctx_idx = _subsample_context(non_masked_idx, max_context_infer, rng)
            ctx_x = torch.from_numpy(x_all[win, ctx_idx]).to(device)
            ctx_y = torch.from_numpy(y_norm_all[win, ctx_idx]).to(device)
            ctx_y, _ = _fill_nan(ctx_y)

            kw = {}
            if is_gnp and dist_matrix_np is not None:
                kw['dist_matrix'] = torch.from_numpy(dist_matrix_np).to(device)
                kw['ctx_indices'] = torch.from_numpy(ctx_idx).to(device)
                kw['tgt_indices'] = torch.from_numpy(masked_idx).to(device)

            y_pred, _ = model(ctx_x, ctx_y, tgt_x, **kw)
            accum += y_pred.cpu().numpy()

        accum /= n_repeats
        out[win, masked_idx] = accum * y_std + y_mean

        # (B) Cross-predict NON-MASKED basins (K-fold leave-out)
        for k in range(crossval_k):
            tgt_local = non_masked_idx[fold_ids == k]     # predict these
            ctx_local = non_masked_idx[fold_ids != k]     # using these as context

            if max_context_infer is not None and len(ctx_local) > max_context_infer:
                ctx_local = ctx_local[rng.choice(len(ctx_local), max_context_infer, replace=False)]

            ctx_x = torch.from_numpy(x_all[win, ctx_local]).to(device)
            ctx_y = torch.from_numpy(y_norm_all[win, ctx_local]).to(device)
            ctx_y, _ = _fill_nan(ctx_y)
            tgt_x_k = torch.from_numpy(x_all[win, tgt_local]).to(device)

            kw = {}
            if is_gnp and dist_matrix_np is not None:
                kw['dist_matrix'] = torch.from_numpy(dist_matrix_np).to(device)
                kw['ctx_indices'] = torch.from_numpy(ctx_local).to(device)
                kw['tgt_indices'] = torch.from_numpy(tgt_local).to(device)

            y_pred_k, _ = model(ctx_x, ctx_y, tgt_x_k, **kw)
            out[win, tgt_local] = y_pred_k.cpu().numpy() * y_std + y_mean

    return out.reshape(-1, seq_len, 1)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Neural Process baseline for zero-shot reconstruction')
    parser.add_argument('--model_type', type=str, default='anp',
                        choices=['cnp', 'np', 'tnpd', 'anp', 'gnp'])
    parser.add_argument('--npz_path', type=str, default='../data_processing/data/prepped.npz')
    parser.add_argument('--masked_basins', nargs='+', required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--enc_layers', type=int, default=2)
    parser.add_argument('--dec_layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta_kl', type=float, default=0.1)
    parser.add_argument('--context_ratio', type=float, default=0.8)
    parser.add_argument('--gnn_layers', type=int, default=2)
    parser.add_argument('--k_neighbors', type=int, default=10)
    parser.add_argument('--max_context', type=int, default=64,
                        help='Max context locations per training episode (GPU memory)')
    parser.add_argument('--max_target', type=int, default=32,
                        help='Max target locations per training episode (GPU memory)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ---- Load data ----
    print('Loading training data ...')
    trn = load_split(args.npz_path, 'train')
    print(f'  {trn["n_windows"]} windows x {trn["n_basins"]} basins x {trn["seq_len"]} steps x {trn["x_dim"]} features')
    print(f'  Y global: mean={trn["y_global_mean"]:.4f}, std={trn["y_global_std"]:.4f}')

    print('Loading validation data ...')
    val = load_split(args.npz_path, 'val',
                     y_global_mean=trn['y_global_mean'],
                     y_global_std=trn['y_global_std'])
    print(f'  {val["n_windows"]} windows x {val["n_basins"]} basins')

    masked_idx, non_masked_idx = get_basin_indices(trn['basin_names'], args.masked_basins)
    print(f'Masked basins: {len(masked_idx)}, Observed basins: {len(non_masked_idx)}')

    # ---- Model ----
    x_dim = trn['x_dim']
    is_gnp = args.model_type == 'gnp'
    common = dict(x_dim=x_dim, y_dim=1, hidden_dim=args.hidden_dim,
                  latent_dim=args.latent_dim, n_heads=args.n_heads,
                  enc_layers=args.enc_layers, dec_layers=args.dec_layers)

    model_cls = {
        'cnp':  ConditionalNeuralProcess,
        'np':   NeuralProcess,
        'tnpd': TransformerNeuralProcess,
        'anp':  AttentiveNeuralProcess,
        'gnp':  GraphNeuralProcess,
    }[args.model_type]

    if args.model_type == 'gnp':
        model = model_cls(**common, gnn_layers=args.gnn_layers,
                          k_neighbors=args.k_neighbors).to(device)
    else:
        model = model_cls(**common).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {args.model_type.upper()}  |  Parameters: {n_params:,}')

    optimizer = Adam(model.parameters(), lr=args.lr)
    dist_np = trn.get('dist_matrix', None)

    # For TNP-D: subsample context during inference to match training distribution
    max_ctx_infer = args.max_context if args.model_type == 'tnpd' else None

    # ---- Output dir ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    pred_dir = os.path.join(output_dir, 'pred')
    os.makedirs(pred_dir, exist_ok=True)

    # ---- Training ----
    best_val, patience_ctr, best_state = float('inf'), 0, None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, trn, non_masked_idx, device, optimizer,
                                 context_ratio=args.context_ratio, beta_kl=args.beta_kl,
                                 dist_matrix_np=dist_np, is_gnp=is_gnp,
                                 max_context=args.max_context, max_target=args.max_target)
        val_loss = evaluate(model, val, masked_idx, non_masked_idx, device,
                            dist_matrix_np=dist_np, is_gnp=is_gnp,
                            max_context_infer=max_ctx_infer)
        elapsed = time.time() - t0
        print(f'Epoch {epoch:3d}/{args.epochs} | trn {train_loss:.6f} | val {val_loss:.6f} | {elapsed:.1f}s')

        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(output_dir, 'best_model.pth'))
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f'Early stopping at epoch {epoch}')
                break

    # ---- Predict ----
    model.load_state_dict(best_state)
    print('Generating predictions ...')
    preds = predict_all(model, trn, masked_idx, non_masked_idx, device,
                        dist_matrix_np=dist_np, is_gnp=is_gnp,
                        max_context_infer=max_ctx_infer)

    # Clip negative (physical constraint for streamflow-like variables)
    preds = np.maximum(preds, 0.0)

    path = os.path.join(pred_dir, 'trn.npy')
    np.save(path, preds)
    print(f'Saved: {path}  shape={preds.shape}  range=[{preds.min():.2f}, {preds.max():.2f}]')


if __name__ == '__main__':
    main()
