"""
MIDM Calibration: Stage 2 of LSTM + MIDM pipeline.

Reads y_obs_* (LSTM predictions, RAW scale) and y_raw_* (truth, RAW scale)
from prepped.npz. Normalizes with y_mean_vae/y_std_vae internally.

Training: noise y_true, condition on y_lstm (clean), predict noise.
Inference: SDEdit — start from noised y_lstm, denoise ~1/3 steps.

Usage:
    python train_cal.py --npz_path ../data_processing/data/prepped.npz \
                        --masked_basins 01022500 02069700 --epochs 200
"""

import argparse, os, time
import numpy as np
import torch
from torch.optim import Adam
from models import MIDM, DiffusionSchedule


# ============================================================
# Data loading — reads RAW y_obs (LSTM) + y_raw (truth),
# normalizes with y_mean_vae/y_std_vae
# ============================================================

def load_split(npz_path, split, y_mean_vae=None, y_std_vae=None):
    data = np.load(npz_path, allow_pickle=True)
    n_basins = int(data['n_segs'])
    basin_names = data['basin_names']
    suffix = {'train': 'trn', 'val': 'val', 'test': 'tst'}[split]

    y_obs_raw = data[f'y_obs_{suffix}']   # LSTM predictions, RAW scale
    y_raw = data[f'y_raw_{suffix}']       # Ground truth, RAW scale
    x = data[f'x_{suffix}']              # Exogenous features

    seq_len = y_raw.shape[1]
    d_x = x.shape[2]
    n_windows = y_raw.shape[0] // n_basins
    assert y_raw.shape[0] == n_windows * n_basins

    y_obs_raw = y_obs_raw.reshape(n_windows, n_basins, seq_len, 1)
    y_raw = y_raw.reshape(n_windows, n_basins, seq_len, 1)
    x = x.reshape(n_windows, n_basins, seq_len, d_x)

    # Per-basin normalization with VAE-predicted stats
    if y_mean_vae is None or y_std_vae is None:
        if 'y_mean_vae' in data:
            y_mean_vae = data['y_mean_vae']  # (n_basins,)
            y_std_vae = data['y_std_vae']    # (n_basins,)
        else:
            y_mean_vae = data['y_mean']
            y_std_vae = data['y_std']
    y_std_vae = np.maximum(y_std_vae, 1e-6)

    # Normalize to per-basin space
    ym = y_mean_vae.reshape(1, -1, 1, 1)   # (1, n_bas, 1, 1)
    ys = y_std_vae.reshape(1, -1, 1, 1)
    y_lstm_norm = (y_obs_raw - ym) / ys     # LSTM predictions, normalized
    y_true_norm = (y_raw - ym) / ys         # Ground truth, normalized

    return {
        'y_lstm': y_lstm_norm.astype(np.float32),   # (n_win, n_bas, seq, 1)
        'y_true': y_true_norm.astype(np.float32),
        'y_raw': y_raw.astype(np.float32),
        'x_features': x.astype(np.float32),
        'basin_names': basin_names, 'n_basins': n_basins,
        'n_windows': n_windows, 'seq_len': seq_len, 'd_x': d_x,
        'y_mean_vae': y_mean_vae.astype(np.float32),
        'y_std_vae': y_std_vae.astype(np.float32),
    }


def get_basin_indices(basin_names, masked_basins):
    name2idx = {str(name): idx for idx, name in enumerate(basin_names)}
    masked = [name2idx[b] for b in masked_basins if b in name2idx]
    non_masked = sorted(set(range(len(basin_names))) - set(masked))
    return np.array(masked, dtype=np.int64), np.array(non_masked, dtype=np.int64)


# ============================================================
# Training — noise y_true, condition on y_lstm (clean)
# ============================================================

def train_epoch(model, schedule, data, masked_idx, non_masked_idx, device,
                optimizer, n_repeats=8):
    """
    Calibration training: noise y_true, condition on y_lstm.

    Denoiser input per basin:
      [y_noisy_true, y_lstm_clean, obs_flag, X]

    Loss on observed basins only (masked basins' y_true not used).
    """
    model.train()
    losses = []
    y_true_all = data['y_true']       # (n_win, n_bas, seq, 1)
    y_lstm_all = data['y_lstm']       # (n_win, n_bas, seq, 1)
    x_all = data['x_features']

    for win in np.random.permutation(data['n_windows']):
        for _ in range(n_repeats):
            # Load all basins
            y_true = torch.from_numpy(y_true_all[win, :, :, 0]).float().to(device).unsqueeze(0)
            y_lstm = torch.from_numpy(y_lstm_all[win, :, :, 0]).float().to(device).unsqueeze(0)
            x_win = torch.from_numpy(x_all[win]).float().to(device).unsqueeze(0)

            # Handle NaN
            nan_mask = torch.isnan(y_true)
            y_true_clean = y_true.clone(); y_true_clean[nan_mask] = 0.0
            valid = (~nan_mask).float()
            y_lstm_clean = y_lstm.clone(); y_lstm_clean[torch.isnan(y_lstm)] = 0.0
            x_nan = torch.isnan(x_win); x_win = x_win.clone(); x_win[x_nan] = 0.0

            # obs_mask: 1 for observed basins, 0 for K-fold masked
            obs_mask = torch.zeros_like(y_true_clean)
            obs_mask[:, non_masked_idx, :] = 1.0
            obs_mask = obs_mask * valid

            # Forward diffusion: noise y_true (NOT y_lstm)
            t = torch.randint(0, schedule.n_steps, (1,), device=device)
            B, _, L = y_true_clean.shape
            noise = model.cov.sample_noise((B, L), device).permute(0, 2, 1)
            ab = schedule.alpha_bars[t].view(-1, 1, 1)
            y_noisy = torch.sqrt(ab) * y_true_clean + torch.sqrt(1 - ab) * noise

            # Denoiser input: [y_noisy_true, y_lstm_clean, obs_flag, X]
            # y_lstm replaces y_prev_c — it's ALWAYS clean, NEVER noised
            noise_pred = model.predict_noise(y_noisy, y_lstm_clean, obs_mask, t, x_win)

            # Loss on observed basins only
            loss_mask = obs_mask
            noise_loss = (loss_mask * (noise_pred - noise) ** 2).sum() / \
                         loss_mask.sum().clamp(min=1)

            # x0 auxiliary loss
            x0_pred = (y_noisy - torch.sqrt(1 - ab) * noise_pred) / torch.sqrt(ab)
            x0_loss = (loss_mask * (x0_pred - y_true_clean) ** 2).sum() / \
                      loss_mask.sum().clamp(min=1)

            loss = noise_loss + 0.5 * x0_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

    return float(np.mean(losses))


# ============================================================
# SDEdit Sampling — start from noised y_lstm, not pure noise
# ============================================================

@torch.no_grad()
def sample_sdedit(model, schedule, y_lstm_full, y_true_full, obs_mask,
                  miss_idx, x_features=None, t_start_frac=0.33):
    """
    SDEdit sampling: start from noised y_lstm, denoise ~1/3 steps.

    Unlike full DDIM (start from conditional noise at t=T):
    - Start from noised y_lstm at t_start ≈ T/3
    - y_lstm is already close to y_true, diffusion only corrects residual
    - At each step, force observed basins to true values

    Args:
        y_lstm_full: (B, N, L) LSTM predictions, normalized, ALL basins
        y_true_full: (B, N, L) ground truth, normalized, ALL basins
        obs_mask:    (B, N, L) 1=observed, 0=masked
        miss_idx:    indices of masked basins
        x_features:  (B, N, L, D_x) optional
        t_start_frac: fraction of total steps to start from (default 0.33)
    Returns:
        (B, n_miss, L) imputed Y for masked basins, normalized
    """
    model.eval()
    device = y_lstm_full.device
    B, N, L = y_lstm_full.shape
    total_T = schedule.n_steps

    # Start from intermediate timestep
    t_start = min(total_T - 1, int(total_T * t_start_frac))

    # Initialize: noised y_lstm (not pure noise!)
    ab_start = schedule.alpha_bars[t_start]
    noise_init = model.cov.sample_noise((B, L), device).permute(0, 2, 1)
    current = torch.sqrt(ab_start) * y_lstm_full + \
              torch.sqrt(1 - ab_start) * noise_init

    # Force observed basins to true values
    current = obs_mask * y_true_full + (1 - obs_mask) * current

    # Reverse denoising from t_start down to 0
    for t_val in range(t_start, -1, -1):
        t_batch = torch.full((B,), t_val, device=device, dtype=torch.long)

        # Predict noise — condition on y_lstm (clean, never noised)
        eps = model.predict_noise(current, y_lstm_full, obs_mask, t_batch, x_features)

        # Reverse step
        ab_t = schedule.alpha_bars[t_val]
        abh_t = 1.0 - schedule.betas[t_val]  # alpha_hat_t
        coeff1 = 1.0 / (abh_t ** 0.5)
        coeff2 = schedule.betas[t_val] / ((1.0 - ab_t) ** 0.5)
        current = coeff1 * (current - coeff2 * eps)

        # Add noise (except at t=0)
        if t_val > 0:
            ab_prev = schedule.alpha_bars[t_val - 1]
            sigma = ((1.0 - ab_prev) / (1.0 - ab_t) * schedule.betas[t_val]) ** 0.5
            noise_step = model.cov.sample_noise((B, L), device).permute(0, 2, 1)
            current = current + sigma * noise_step

        # Clamp for stability
        current = current.clamp(-10, 10)

        # Force observed basins to true values at each step
        current = obs_mask * y_true_full + (1 - obs_mask) * current

    return current[:, miss_idx, :]


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, schedule, data, masked_idx, non_masked_idx, device,
             n_samples=3, t_start_frac=0.33):
    """MSE on masked basins (normalized space), using SDEdit."""
    model.eval()
    total_se, total_n = 0.0, 0
    y_true_all = data['y_true']
    y_lstm_all = data['y_lstm']
    x_all = data['x_features']

    for win in range(data['n_windows']):
        y_true = torch.from_numpy(y_true_all[win, :, :, 0]).float().to(device).unsqueeze(0)
        y_lstm = torch.from_numpy(y_lstm_all[win, :, :, 0]).float().to(device).unsqueeze(0)
        x_win = torch.from_numpy(x_all[win]).float().to(device).unsqueeze(0)

        nan_true = torch.isnan(y_true)
        y_true_clean = y_true.clone(); y_true_clean[nan_true] = 0.0
        y_lstm_clean = y_lstm.clone(); y_lstm_clean[torch.isnan(y_lstm)] = 0.0
        x_win[torch.isnan(x_win)] = 0.0
        tgt_valid = (~nan_true[:, masked_idx, :]).float()

        obs_mask = torch.zeros_like(y_true_clean)
        obs_mask[:, non_masked_idx, :] = 1.0
        obs_mask = obs_mask * (~nan_true).float()

        accum = torch.zeros(1, len(masked_idx), data['seq_len'], device=device)
        for _ in range(n_samples):
            pred = sample_sdedit(model, schedule, y_lstm_clean, y_true_clean,
                                 obs_mask, masked_idx, x_win, t_start_frac)
            accum += pred
        accum /= n_samples

        total_se += ((accum - y_true_clean[:, masked_idx, :]) ** 2 * tgt_valid).sum().item()
        total_n += tgt_valid.sum().item()

    return total_se / max(total_n, 1)


# ============================================================
# Prediction
# ============================================================

@torch.no_grad()
def predict_all(model, schedule, data, masked_idx, non_masked_idx, device,
                n_pred_samples=10, t_start_frac=0.33):
    """
    Produce predictions for ALL basins in RAW scale.

    Masked basins: SDEdit from y_lstm.
    Non-masked basins: ground truth passthrough.
    """
    model.eval()
    y_true_all = data['y_true']
    y_lstm_all = data['y_lstm']
    x_all = data['x_features']
    y_raw_all = data['y_raw']
    y_mean_vae = data['y_mean_vae']
    y_std_vae = data['y_std_vae']
    n_win, seq_len = data['n_windows'], data['seq_len']

    out = y_raw_all.copy()  # (n_win, n_bas, seq, 1)

    for win in range(n_win):
        print(f'  Window {win + 1}/{n_win}', end='\r')

        y_true = torch.from_numpy(y_true_all[win, :, :, 0]).float().to(device).unsqueeze(0)
        y_lstm = torch.from_numpy(y_lstm_all[win, :, :, 0]).float().to(device).unsqueeze(0)
        x_win = torch.from_numpy(x_all[win]).float().to(device).unsqueeze(0)

        y_true_clean = y_true.clone(); y_true_clean[torch.isnan(y_true)] = 0.0
        y_lstm_clean = y_lstm.clone(); y_lstm_clean[torch.isnan(y_lstm)] = 0.0
        x_win[torch.isnan(x_win)] = 0.0

        obs_mask = torch.zeros_like(y_true_clean)
        obs_mask[:, non_masked_idx, :] = 1.0
        obs_mask = obs_mask * (~torch.isnan(y_true)).float()

        accum = torch.zeros(1, len(masked_idx), seq_len, device=device)
        for _ in range(n_pred_samples):
            pred = sample_sdedit(model, schedule, y_lstm_clean, y_true_clean,
                                 obs_mask, masked_idx, x_win, t_start_frac)
            accum += pred
        accum /= n_pred_samples

        # Denormalize per-basin: pred_raw = pred_norm * y_std_vae + y_mean_vae
        pred_np = accum.squeeze(0).cpu().numpy()  # (n_miss, seq_len)
        pred_raw = pred_np * y_std_vae[masked_idx, None] + \
                   y_mean_vae[masked_idx, None]
        out[win, masked_idx, :, 0] = pred_raw

    print()
    return out.reshape(-1, seq_len, 1)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='MIDM Calibration (Stage 2)')
    parser.add_argument('--npz_path', default='../data_processing/data/prepped.npz')
    parser.add_argument('--masked_basins', nargs='+', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_diffusion_steps', type=int, default=50)
    parser.add_argument('--cov_rank', type=int, default=8)
    parser.add_argument('--n_repeats', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_pred_samples', type=int, default=10)
    parser.add_argument('--t_start_frac', type=float, default=0.33,
                        help='SDEdit start fraction (0.33 = start from 1/3 of steps)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print('Loading data ...')
    trn = load_split(args.npz_path, 'train')
    val = load_split(args.npz_path, 'val',
                     trn['y_mean_vae'], trn['y_std_vae'])
    masked_idx, non_masked_idx = get_basin_indices(trn['basin_names'], args.masked_basins)
    d_x = trn['d_x']
    print(f'  {trn["n_windows"]}win x {trn["n_basins"]}bas x {trn["seq_len"]}steps x {d_x} X features')
    print(f'  Masked: {len(masked_idx)}, Observed: {len(non_masked_idx)}')
    print(f'  SDEdit t_start_frac: {args.t_start_frac}')

    model = MIDM(
        n_vars=trn['n_basins'], max_seq_len=trn['seq_len'],
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        n_diffusion_steps=args.n_diffusion_steps, cov_rank=args.cov_rank,
        d_x=d_x, dropout=args.dropout,
    ).to(device)
    schedule = DiffusionSchedule(n_steps=args.n_diffusion_steps, device=device)
    print(f'  Params: {sum(p.numel() for p in model.parameters()):,}  (d_x={d_x})')

    optimizer = Adam(model.parameters(), lr=args.lr)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(os.path.join(output_dir, 'pred'), exist_ok=True)

    best_val, patience_ctr, best_state = float('inf'), 0, None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        trn_loss = train_epoch(model, schedule, trn, masked_idx, non_masked_idx,
                               device, optimizer, n_repeats=args.n_repeats)
        val_loss = evaluate(model, schedule, val, masked_idx, non_masked_idx,
                            device, n_samples=3, t_start_frac=args.t_start_frac)
        print(f'Epoch {epoch:3d}/{args.epochs} | trn {trn_loss:.6f} | '
              f'val {val_loss:.6f} | {time.time()-t0:.1f}s')
        if val_loss < best_val:
            best_val, patience_ctr = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(output_dir, 'best_model.pth'))
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f'Early stopping at epoch {epoch}'); break

    model.load_state_dict(best_state); model.to(device)
    print('Generating predictions (SDEdit) ...')
    preds = np.maximum(predict_all(
        model, schedule, trn, masked_idx, non_masked_idx, device,
        args.n_pred_samples, args.t_start_frac), 0.0)
    path = os.path.join(output_dir, 'pred', 'trn.npy')
    np.save(path, preds)
    print(f'Saved: {path}  shape={preds.shape}  range=[{preds.min():.2f}, {preds.max():.2f}]')


if __name__ == '__main__':
    main()
