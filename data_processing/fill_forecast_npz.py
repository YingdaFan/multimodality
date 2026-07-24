#!/usr/bin/env python3
"""
Bridge Script: fill_forecast_npz.py

Convert imputation output (windowed predictions) → y_imputed in forecast NPZ.

Workflow:
  1. Run imputation pipeline → predictions in diffusion/output/pred/trn.npy
  2. Extract metadata (before forecast preprocessing overwrites prepped.npz):
       python fill_forecast_npz.py prepare --imputation_npz data/prepped.npz
  3. Run forecast preprocessing:
       python preprocess_camelsh_forecast.py
  4. Inject y_imputed into forecast NPZ:
       python fill_forecast_npz.py inject \
           --pred_dir ../diffusion/output/pred \
           --forecast_npz data/prepped.npz

Imputation predictions are in original scale (globally denormalized).
Bridge re-normalizes to per-basin scale matching the forecast NPZ.
"""

import argparse
import os
import sys
import numpy as np


DEFAULT_META_PATH = 'data/imputation_meta.npz'


# ==================================================================
# prepare: extract ids/times/basin_names from imputation NPZ
# ==================================================================

def cmd_prepare(args):
    """Extract metadata from imputation NPZ before it gets overwritten."""
    print("Prepare: extract imputation metadata")

    imp = np.load(args.imputation_npz, allow_pickle=True)
    meta = {'basin_names': imp['basin_names']}

    for partition in ['trn', 'val', 'tst']:
        ids_key = f'ids_{partition}'
        times_key = f'times_{partition}'
        if ids_key in imp:
            meta[ids_key] = imp[ids_key]
            meta[times_key] = imp[times_key]
            print(f"  {ids_key}: {imp[ids_key].shape}")

    np.savez_compressed(args.meta_output, **meta)
    print(f"Saved: {args.meta_output}")


# ==================================================================
# reconstruct: windowed predictions → continuous time series
# ==================================================================

def reconstruct_continuous(pred_npy, ids, times, basin_names):
    """
    Reconstruct continuous time series from windowed imputation predictions.

    Parameters
    ----------
    pred_npy : (n_samples, pred_len) or (n_samples, pred_len, 1) — original scale
    ids :      (n_samples, seq_len, 1) — basin IDs per sample
    times :    (n_samples, seq_len, 1) — datetime64 timestamps per sample
    basin_names : (n_basins,)

    Returns
    -------
    unique_times : (n_unique_times,)
    continuous :   (n_basins, n_unique_times, 1) — NaN where no prediction
    """
    if pred_npy.ndim == 3:
        pred_npy = pred_npy[:, :, 0]

    n_samples, pred_len = pred_npy.shape
    n_basins = len(basin_names)

    print(f"  samples={n_samples}, pred_len={pred_len}, "
          f"n_basins={n_basins}, n_windows={n_samples // n_basins}")

    # Build time index
    unique_times = np.unique(times[:, :pred_len, 0].ravel())
    unique_times.sort()
    time_to_idx = {t: i for i, t in enumerate(unique_times)}
    n_times = len(unique_times)
    print(f"  time range: {unique_times[0]} → {unique_times[-1]} ({n_times} steps)")

    # Accumulate (handles overlapping windows by averaging)
    accum = np.zeros((n_basins, n_times), dtype=np.float64)
    count = np.zeros((n_basins, n_times), dtype=np.int32)

    n_windows = n_samples // n_basins
    for w in range(n_windows):
        sample_0 = w * n_basins
        window_times = times[sample_0, :pred_len, 0]
        t_indices = np.array([time_to_idx[t] for t in window_times])

        for b in range(n_basins):
            preds = pred_npy[sample_0 + b]
            valid = ~np.isnan(preds)
            accum[b, t_indices[valid]] += preds[valid]
            count[b, t_indices[valid]] += 1

    continuous = np.full((n_basins, n_times, 1), np.nan, dtype=np.float32)
    valid = count > 0
    continuous[valid, 0] = (accum[valid] / count[valid]).astype(np.float32)

    filled = valid.sum()
    print(f"  coverage: {filled}/{n_basins * n_times} "
          f"({filled / (n_basins * n_times) * 100:.1f}%)")

    return unique_times, continuous


# ==================================================================
# inject: reconstruct + slice by forecast partitions + re-normalize
# ==================================================================

def cmd_inject(args):
    """Load imputation trn.npy, slice by forecast partition times, inject."""
    pred_file = os.path.join(args.pred_dir, 'trn.npy')
    print(f"Loading prediction: {pred_file}")

    meta = np.load(args.meta, allow_pickle=True)
    basin_names = meta['basin_names']

    if 'ids_trn' not in meta:
        print(f"ERROR: ids_trn not found in metadata {args.meta}")
        sys.exit(1)

    pred = np.load(pred_file)
    print(f"  pred shape: {pred.shape}")

    imp_times, imp_continuous = reconstruct_continuous(
        pred, meta['ids_trn'], meta['times_trn'], basin_names
    )

    # ---- Load forecast NPZ ----
    print(f"\nLoading forecast NPZ: {args.forecast_npz}")
    fc_data = np.load(args.forecast_npz, allow_pickle=True)
    fc_dict = {key: fc_data[key] for key in fc_data.files}

    fc_basins = fc_dict['basin_names']
    fc_y_mean = fc_dict['y_mean']
    fc_y_std = fc_dict['y_std']

    imp_b2i = {str(b): i for i, b in enumerate(basin_names)}
    imp_t2i = {t: i for i, t in enumerate(imp_times)}

    common = sum(1 for b in fc_basins if str(b) in imp_b2i)
    print(f"  basins: {len(fc_basins)} forecast, {common} matched with imputation")

    # ---- For each forecast partition, slice + re-normalize ----
    for partition in ['trn', 'val', 'tst']:
        times_key = f'times_{partition}'
        if times_key not in fc_dict:
            continue

        fc_times = fc_dict[times_key]
        n_fc_times = len(fc_times)
        n_fc_basins = len(fc_basins)

        # Time alignment: find which forecast times exist in imputation
        fc_to_imp_t = np.array([imp_t2i.get(t, -1) for t in fc_times], dtype=np.int64)
        matched_fc_t = np.where(fc_to_imp_t >= 0)[0]
        matched_imp_t = fc_to_imp_t[matched_fc_t]

        n_matched = len(matched_fc_t)
        n_missing = n_fc_times - n_matched
        print(f"  [{partition}] {fc_times[0]} → {fc_times[-1]}, "
              f"matched: {n_matched}/{n_fc_times}"
              + (f" (WARNING: {n_missing} unmatched)" if n_missing > 0 else ""))

        # Build y_imputed: re-normalize from original scale → per-basin normalized
        y_imputed = np.full((n_fc_basins, n_fc_times, 1), np.nan, dtype=np.float32)

        for fc_b_idx, basin in enumerate(fc_basins):
            if str(basin) not in imp_b2i:
                continue
            imp_b_idx = imp_b2i[str(basin)]
            b_mean = fc_y_mean[fc_b_idx]
            b_std = fc_y_std[fc_b_idx]

            raw_vals = imp_continuous[imp_b_idx, matched_imp_t, 0]
            valid = ~np.isnan(raw_vals)
            y_imputed[fc_b_idx, matched_fc_t[valid], 0] = (
                (raw_vals[valid] - b_mean) / (b_std + 1e-10)
            )

        filled = (~np.isnan(y_imputed[:, :, 0])).sum()
        total = n_fc_basins * n_fc_times
        nan_count = total - filled
        print(f"  [{partition}] coverage: {filled}/{total} ({filled / total * 100:.1f}%)")
        if nan_count > 0:
            print(f"  [{partition}] WARNING: {nan_count} NaN remaining in y_imputed")

        fc_dict[f'y_imputed_{partition}'] = y_imputed

    # Save
    output_path = args.output or args.forecast_npz
    np.savez_compressed(output_path, **fc_dict)
    print(f"\nSaved: {output_path}")

    for key in ['y_imputed_trn', 'y_imputed_val', 'y_imputed_tst']:
        if key in fc_dict:
            arr = fc_dict[key]
            print(f"  {key}: {arr.shape}, "
                  f"[{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]")


# ==================================================================
# CLI
# ==================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Bridge: imputation predictions → forecast NPZ y_imputed',
    )
    sub = parser.add_subparsers(dest='command')

    # prepare
    p = sub.add_parser('prepare',
                       help='Extract ids/times metadata from imputation NPZ')
    p.add_argument('--imputation_npz', required=True)
    p.add_argument('--meta_output', default=DEFAULT_META_PATH)

    # inject
    p = sub.add_parser('inject',
                       help='Reconstruct + inject y_imputed into forecast NPZ')
    p.add_argument('--pred_dir', required=True,
                   help='Directory containing trn.npy')
    p.add_argument('--forecast_npz', required=True,
                   help='Forecast prepped.npz (continuous format)')
    p.add_argument('--meta', default=DEFAULT_META_PATH,
                   help=f'Metadata from prepare step (default: {DEFAULT_META_PATH})')
    p.add_argument('--output', default=None,
                   help='Output path (default: overwrite --forecast_npz)')

    args = parser.parse_args()
    if args.command == 'prepare':
        cmd_prepare(args)
    elif args.command == 'inject':
        cmd_inject(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
