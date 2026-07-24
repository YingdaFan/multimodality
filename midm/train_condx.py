"""
Training for MIDM with position-invariant denoiser + X conditioning.

No pseudo-masking. Like LSTM, ALL basins participate in training:
  - Observed basins: have Y, obs_flag=1, contribute to loss
  - Masked basins: Y=0, obs_flag=0, X participates via spatial attention
  - Shared weights transfer learning from observed → masked basins

Usage:
    python train_condx.py --npz_path ../data_processing/data/prepped.npz \
                          --masked_basins 01022500 02069700 --epochs 200
"""

import argparse, os, time
import numpy as np
import torch
from torch.optim import Adam
from models import MIDM, DiffusionSchedule, sample_ddim


def load_split(npz_path, split, y_basin_mean=None, y_basin_std=None):
    data = np.load(npz_path, allow_pickle=True)
    n_basins = int(data['n_segs'])
    basin_names = data['basin_names']
    suffix = {'train': 'trn', 'val': 'val', 'test': 'tst'}[split]
    y_raw = data[f'y_raw_{suffix}']
    x = data[f'x_{suffix}']
    seq_len = y_raw.shape[1]
    d_x = x.shape[2]
    n_windows = y_raw.shape[0] // n_basins
    assert y_raw.shape[0] == n_windows * n_basins
    y_raw = y_raw.reshape(n_windows, n_basins, seq_len, 1)
    x = x.reshape(n_windows, n_basins, seq_len, d_x)

    # Per-basin normalization: each basin gets its own mean/std
    if y_basin_mean is None or y_basin_std is None:
        y_trn = data['y_raw_trn'].reshape(-1, n_basins, seq_len, 1)
        y_basin_mean = np.nanmean(y_trn, axis=(0, 2, 3))  # (n_basins,)
        y_basin_std = np.nanstd(y_trn, axis=(0, 2, 3))    # (n_basins,)
        y_basin_std = np.maximum(y_basin_std, 1e-6)        # avoid div by 0

    # Normalize: (n_win, n_bas, seq, 1) with per-basin stats (1, n_bas, 1, 1)
    y_norm = (y_raw - y_basin_mean[None, :, None, None]) / \
             y_basin_std[None, :, None, None]

    return {
        'y_raw': y_raw.astype(np.float32),
        'y_norm': y_norm.astype(np.float32),
        'x_features': x.astype(np.float32),
        'basin_names': basin_names, 'n_basins': n_basins,
        'n_windows': n_windows, 'seq_len': seq_len, 'd_x': d_x,
        'y_basin_mean': y_basin_mean.astype(np.float32),  # (n_basins,)
        'y_basin_std': y_basin_std.astype(np.float32),    # (n_basins,)
    }


def get_basin_indices(basin_names, masked_basins):
    name2idx = {str(name): idx for idx, name in enumerate(basin_names)}
    masked = [name2idx[b] for b in masked_basins if b in name2idx]
    non_masked = sorted(set(range(len(basin_names))) - set(masked))
    return np.array(masked, dtype=np.int64), np.array(non_masked, dtype=np.int64)


def train_epoch(model, schedule, data, masked_idx, non_masked_idx, device,
                optimizer, n_repeats=8, mask_ratio_range=(0.1, 0.3)):
    """
    One training epoch — hybrid: pseudo-masking + all observed basins contribute.

    ALL basins participate in the forward pass:
      - K-fold masked basins: obs_flag=0, Y=0, X via spatial attention, no loss
      - Pseudo-masked observed basins: obs_flag=0, Y known → loss (weight 2.0)
      - Remaining observed basins: obs_flag=1, Y known → loss (weight 1.0)

    This combines v3's pseudo-masking (model practices obs_flag=0 predictions)
    with v5's full training signal (all observed basins contribute to loss).
    """
    model.train()
    losses = []
    y_all, x_all = data['y_norm'], data['x_features']
    n_obs = len(non_masked_idx)

    for win in np.random.permutation(data['n_windows']):
        for _ in range(n_repeats):
            y_win = torch.from_numpy(y_all[win, :, :, 0]).float().to(device).unsqueeze(0)
            x_win = torch.from_numpy(x_all[win]).float().to(device).unsqueeze(0)

            nan_mask = torch.isnan(y_win)
            y_win = y_win.clone(); y_win[nan_mask] = 0.0
            valid = (~nan_mask).float()
            x_nan = torch.isnan(x_win); x_win = x_win.clone(); x_win[x_nan] = 0.0

            # Pseudo-mask some observed basins
            mask_ratio = np.random.uniform(*mask_ratio_range)
            n_pseudo_mask = max(1, int(n_obs * mask_ratio))
            perm = np.random.permutation(n_obs)
            pseudo_miss_global = non_masked_idx[perm[:n_pseudo_mask]]
            pseudo_obs_global = non_masked_idx[perm[n_pseudo_mask:]]

            # obs_mask: 1 for pseudo-observed, 0 for pseudo-masked + K-fold masked
            obs_mask = torch.zeros_like(y_win)
            obs_mask[:, pseudo_obs_global, :] = 1.0
            obs_mask = obs_mask * valid

            y_win[:, masked_idx, :] = 0.0

            # Forward diffusion
            t = torch.randint(0, schedule.n_steps, (1,), device=device)
            B, _, L = y_win.shape
            noise = model.cov.sample_noise((B, L), device).permute(0, 2, 1)
            ab = schedule.alpha_bars[t].view(-1, 1, 1)
            y_noisy = torch.sqrt(ab) * y_win + torch.sqrt(1 - ab) * noise

            if t.item() > 0:
                ab_prev = schedule.alpha_bars[t.item() - 1].view(1, 1, 1)
            else:
                ab_prev = torch.ones(1, 1, 1, device=device)
            noise_prev = model.cov.sample_noise((B, L), device).permute(0, 2, 1)
            y_prev_c = (torch.sqrt(ab_prev) * y_win +
                        torch.sqrt(1 - ab_prev) * noise_prev) * obs_mask

            noise_pred = model.predict_noise(y_noisy, y_prev_c, obs_mask, t, x_win)
            loss_mask = torch.zeros_like(y_win)
            loss_mask[:, pseudo_obs_global, :] = 1.0
            loss_mask[:, pseudo_miss_global, :] = 2.0
            loss_mask = loss_mask * valid

            noise_loss = (loss_mask * (noise_pred - noise) ** 2).sum() / \
                         loss_mask.sum().clamp(min=1)

            x0_pred = (y_noisy - torch.sqrt(1 - ab) * noise_pred) / torch.sqrt(ab)
            x0_loss = (loss_mask * (x0_pred - y_win) ** 2).sum() / \
                      loss_mask.sum().clamp(min=1)

            loss = noise_loss + 0.5 * x0_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, schedule, data, masked_idx, non_masked_idx, device,
             n_sample_steps=15):
    model.eval()
    total_se, total_n = 0.0, 0
    y_all, x_all = data['y_norm'], data['x_features']
    obs_idx_t = torch.from_numpy(non_masked_idx).long().to(device)
    miss_idx_t = torch.from_numpy(masked_idx).long().to(device)

    for win in range(data['n_windows']):
        y_obs = torch.from_numpy(
            y_all[win, non_masked_idx, :, 0]).float().to(device).unsqueeze(0)
        y_tgt = torch.from_numpy(
            y_all[win, masked_idx, :, 0]).float().to(device).unsqueeze(0)
        tgt_nan = torch.isnan(y_tgt)
        y_tgt_clean = y_tgt.clone(); y_tgt_clean[tgt_nan] = 0.0
        tgt_valid = (~tgt_nan).float()

        x_win = torch.from_numpy(x_all[win]).float().to(device).unsqueeze(0)
        x_nan = torch.isnan(x_win); x_win[x_nan] = 0.0

        y_pred = sample_ddim(model, schedule, y_obs, obs_idx_t, miss_idx_t,
                             n_steps=n_sample_steps, x_features=x_win)
        total_se += ((y_pred - y_tgt_clean) ** 2 * tgt_valid).sum().item()
        total_n += tgt_valid.sum().item()

    return total_se / max(total_n, 1)


@torch.no_grad()
def predict_all(model, schedule, data, masked_idx, non_masked_idx, device,
                n_pred_samples=10, n_sample_steps=50):
    model.eval()
    y_all, x_all = data['y_norm'], data['x_features']
    y_raw_all = data['y_raw']
    y_basin_mean = data['y_basin_mean']  # (n_basins,)
    y_basin_std = data['y_basin_std']    # (n_basins,)
    n_win, seq_len = data['n_windows'], data['seq_len']
    obs_idx_t = torch.from_numpy(non_masked_idx).long().to(device)
    miss_idx_t = torch.from_numpy(masked_idx).long().to(device)
    out = y_raw_all.copy()

    for win in range(n_win):
        print(f'  Window {win + 1}/{n_win}', end='\r')
        y_obs = torch.from_numpy(
            y_all[win, non_masked_idx, :, 0]).float().to(device).unsqueeze(0)
        x_win = torch.from_numpy(x_all[win]).float().to(device).unsqueeze(0)
        x_nan = torch.isnan(x_win); x_win[x_nan] = 0.0

        accum = torch.zeros(1, len(masked_idx), seq_len, device=device)
        for _ in range(n_pred_samples):
            accum += sample_ddim(model, schedule, y_obs, obs_idx_t, miss_idx_t,
                                 n_steps=n_sample_steps, x_features=x_win)
        accum /= n_pred_samples
        # Per-basin denormalization: (n_miss, seq_len) * std[miss] + mean[miss]
        pred_np = accum.squeeze(0).cpu().numpy()              # (n_miss, seq_len)
        pred_raw = pred_np * y_basin_std[masked_idx, None] + \
                   y_basin_mean[masked_idx, None]
        out[win, masked_idx, :, 0] = pred_raw
    print()
    return out.reshape(-1, seq_len, 1)


def main():
    parser = argparse.ArgumentParser(description='MIDM + X conditioning')
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
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_pred_samples', type=int, default=10)
    parser.add_argument('--n_pred_steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print('Loading data ...')
    trn = load_split(args.npz_path, 'train')
    val = load_split(args.npz_path, 'val',
                     trn['y_basin_mean'], trn['y_basin_std'])
    masked_idx, non_masked_idx = get_basin_indices(trn['basin_names'], args.masked_basins)
    d_x = trn['d_x']
    print(f'  {trn["n_windows"]}win x {trn["n_basins"]}bas x {trn["seq_len"]}steps x {d_x} X features')
    print(f'  Masked: {len(masked_idx)}, Observed: {len(non_masked_idx)}')

    model = MIDM(
        n_vars=trn['n_basins'], max_seq_len=trn['seq_len'],
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        n_diffusion_steps=args.n_diffusion_steps, cov_rank=args.cov_rank,
        d_x=d_x, dropout=args.dropout,
    ).to(device)
    # Convert denoiser to BF16 for memory efficiency; keep covariance in FP32
    model.denoiser = model.denoiser.to(torch.bfloat16)
    schedule = DiffusionSchedule(n_steps=args.n_diffusion_steps, device=device)
    print(f'  Params: {sum(p.numel() for p in model.parameters()):,}  (d_x={d_x}, BF16 denoiser)')

    optimizer = Adam(model.parameters(), lr=args.lr)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(os.path.join(output_dir, 'pred'), exist_ok=True)

    best_val, patience_ctr, best_state = float('inf'), 0, None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        trn_loss = train_epoch(model, schedule, trn, masked_idx, non_masked_idx,
                               device, optimizer, n_repeats=args.n_repeats)
        val_loss = evaluate(model, schedule, val, masked_idx, non_masked_idx,
                            device, n_sample_steps=15)
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
    print('Generating predictions ...')
    preds = np.maximum(predict_all(
        model, schedule, trn, masked_idx, non_masked_idx, device,
        args.n_pred_samples, args.n_pred_steps), 0.0)
    path = os.path.join(output_dir, 'pred', 'trn.npy')
    np.save(path, preds)
    print(f'Saved: {path}  shape={preds.shape}  range=[{preds.min():.2f}, {preds.max():.2f}]')


if __name__ == '__main__':
    main()
