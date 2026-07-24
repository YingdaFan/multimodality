"""
Bridge script: fill prepped.npz with NP predictions + NP-derived statistics.

Replaces:
  - y_obs_trn  ←  NP predictions (original scale)
  - y_mean_vae ←  per-basin mean of NP predictions
  - y_std_vae  ←  per-basin std  of NP predictions

After running this, the existing diffusion stage 2 can be called directly
(no code changes needed — the NPZ is the interface).

Usage:
    python fill_npz.py --pred_path output/pred/trn.npy \
                       --npz_path ../data_processing/data/prepped.npz
"""

import argparse
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, default='output/pred/trn.npy',
                        help='NP predictions (n_samples, seq_len, 1), original scale')
    parser.add_argument('--npz_path', type=str, default='../data_processing/data/prepped.npz')
    args = parser.parse_args()

    # --- Load ---
    preds = np.load(args.pred_path)  # (n_samples, seq_len, 1)
    data = dict(np.load(args.npz_path, allow_pickle=True))

    n_basins = int(data['n_segs'])
    n_samples = preds.shape[0]
    n_windows = n_samples // n_basins
    seq_len = preds.shape[1]

    print(f'Predictions: {preds.shape}  ({n_windows} windows x {n_basins} basins)')
    print(f'Value range: [{preds.min():.4f}, {preds.max():.4f}]')

    # --- Compute per-basin mean / std from NP predictions ---
    # Data is time-major: sample[t * n_basins + b] = basin b at window t
    preds_by_basin = preds.reshape(n_windows, n_basins, seq_len, -1)  # (win, bas, seq, 1)

    y_mean_np = np.zeros(n_basins, dtype=np.float32)
    y_std_np = np.zeros(n_basins, dtype=np.float32)

    for b in range(n_basins):
        vals = preds_by_basin[:, b, :, 0].flatten()  # all predictions for basin b
        valid = vals[~np.isnan(vals)]
        if len(valid) > 0:
            y_mean_np[b] = np.mean(valid)
            y_std_np[b] = max(np.std(valid), 1e-6)  # avoid zero std
        else:
            y_mean_np[b] = 0.0
            y_std_np[b] = 1.0

    print(f'Per-basin stats: mean=[{y_mean_np.min():.4f}, {y_mean_np.max():.4f}], '
          f'std=[{y_std_np.min():.4f}, {y_std_np.max():.4f}]')

    # --- Fill NPZ ---
    data['y_obs_trn'] = preds.astype(np.float32)
    data['y_mean_vae'] = y_mean_np
    data['y_std_vae'] = y_std_np

    # Also fill val/tst y_obs if NP predictions exist for them
    for split, key in [('val', 'y_obs_val'), ('tst', 'y_obs_tst')]:
        split_pred_path = os.path.join(os.path.dirname(args.pred_path), f'{split}.npy')
        if os.path.exists(split_pred_path):
            split_preds = np.load(split_pred_path)
            data[key] = split_preds.astype(np.float32)
            print(f'Filled {key} from {split_pred_path}')
        else:
            # If no NP predictions for this split, fill with y_raw (ground truth)
            # so diffusion has valid inputs for val/tst loaders
            raw_key = f'y_raw_{split}'
            if raw_key in data:
                data[key] = data[raw_key].copy()
                print(f'Filled {key} with y_raw (no NP predictions for {split})')

    np.savez(args.npz_path, **data)
    print(f'\nSaved to {args.npz_path}')
    print(f'  y_obs_trn: {data["y_obs_trn"].shape}')
    print(f'  y_mean_vae: {data["y_mean_vae"].shape}')
    print(f'  y_std_vae: {data["y_std_vae"].shape}')


if __name__ == '__main__':
    main()
