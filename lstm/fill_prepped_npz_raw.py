#!/usr/bin/env python3


import numpy as np
import argparse
import os
import sys


def fill_partition_raw(data_dict, partition, pred_file, y_mean, y_std, basin_names, output_npy_dir=None):
    """
    Fill y_obs_{partition} with denormalized LSTM predictions for ALL basins.
    Also outputs denormalized .npy file if output_npy_dir is specified.

    Returns:
        tuple: (total_filled, denormalized_predictions)
    """
    obs_key = f'y_obs_{partition}'
    raw_key = f'y_raw_{partition}'
    ids_key = f'ids_{partition}'

    if obs_key not in data_dict or raw_key not in data_dict:
        print(f"  Warning: {obs_key} or {raw_key} not found, skipping...")
        return 0, None

    if not os.path.exists(pred_file):
        print(f"  Warning: Prediction file {pred_file} not found, skipping...")
        return 0, None

    print(f"\n  Loading LSTM predictions from {pred_file}...")
    lstm_pred_normalized = np.load(pred_file)

    ids = data_dict[ids_key]
    y_obs = np.zeros_like(lstm_pred_normalized)  # Will be filled with denormalized LSTM predictions

    basin_name_to_idx = {name: idx for idx, name in enumerate(basin_names)}

    if data_dict[obs_key].shape != lstm_pred_normalized.shape:
        print(f"  Warning: Shape mismatch: {obs_key} {data_dict[obs_key].shape} vs lstm_pred {lstm_pred_normalized.shape}")
        return 0, None

    n_samples = len(ids)
    n_basins = len(basin_names)
    n_windows = n_samples // n_basins

    print(f"  {partition}: {n_windows} windows x {n_basins} basins = {n_samples} samples")

    # Denormalize LSTM predictions for ALL basins
    total_filled = 0
    for basin_name in basin_names:
        basin_idx = basin_name_to_idx[basin_name]
        b_mean = y_mean[basin_idx]
        b_std = y_std[basin_idx]

        # Find all samples for this basin
        sample_indices = []
        for i in range(n_samples):
            if ids[i, 0, 0] == basin_name:
                sample_indices.append(i)

        if len(sample_indices) == 0:
            continue

        # Denormalize LSTM predictions
        for idx in sample_indices:
            y_obs[idx] = lstm_pred_normalized[idx] * (b_std + 1e-10) + b_mean

        total_filled += len(sample_indices)

    # Update y_obs_* (y_raw_* remains unchanged!)
    data_dict[obs_key] = y_obs

    # Save denormalized .npy file if output directory specified
    if output_npy_dir is not None:
        os.makedirs(output_npy_dir, exist_ok=True)
        output_npy_file = os.path.join(output_npy_dir, f'{partition}.npy')
        np.save(output_npy_file, y_obs)
        print(f"  {partition}: Saved denormalized predictions to {output_npy_file}")

    print(f"  {partition}: y_obs_* filled with denormalized LSTM predictions ({total_filled} samples)")
    return total_filled, y_obs


def fill_prepped_npz_raw(npz_file, pred_dir, masked_basin_names, output_file=None, output_npy_dir=None):
    """
    Fill y_obs_* with denormalized LSTM predictions for ALL basins.

    Key improvement: uses y_mean_vae/y_std_vae for denormalization
    - All basins use VAE-predicted statistics
    - Simulates real-world application scenario

    Parameters:
    -----------
    npz_file : str
        Path to prepped.npz
    pred_dir : str
        Directory containing normalized LSTM predictions (trn.npy, val.npy, tst.npy)
    masked_basin_names : list
        Basin IDs that were masked (for Stage 2 label)
    output_file : str, optional
        Output npz file path
    output_npy_dir : str, optional
        Directory to save denormalized .npy files for evaluation
    """
    if isinstance(masked_basin_names, str):
        masked_basin_names = [masked_basin_names]

    print(f"Loading npz data from {npz_file}...")
    data = np.load(npz_file, allow_pickle=True)
    data_dict = {key: data[key] for key in data.files}

    basin_names = data_dict['basin_names']

    # Use y_mean_vae/y_std_vae for denormalization (all basins use VAE-predicted statistics)
    if 'y_mean_vae' in data_dict and 'y_std_vae' in data_dict:
        y_mean_for_denorm = data_dict['y_mean_vae']
        y_std_for_denorm = data_dict['y_std_vae']
        print(f"\nUsing y_mean_vae/y_std_vae for denormalization (all basins use VAE statistics)")
    else:
        # Fallback: use y_mean/y_std (backward compatibility)
        y_mean_for_denorm = data_dict['y_mean']
        y_std_for_denorm = data_dict['y_std']
        print(f"\nWarning: y_mean_vae/y_std_vae not found, falling back to y_mean/y_std")
        print(f"  Please ensure apply_vae.py has been run to generate y_mean_vae/y_std_vae")

    print(f"Number of basins: {len(basin_names)}")
    print(f"Masked basins (for Stage 2 label): {masked_basin_names}")
    print(f"\nNOTE: ALL basins will be filled with LSTM predictions (denormalized with VAE stats)")

    total_filled = 0
    partitions = ['trn', 'val', 'tst']

    for partition in partitions:
        pred_file = os.path.join(pred_dir, f'{partition}.npy')
        filled_count, _ = fill_partition_raw(
            data_dict, partition, pred_file,
            y_mean_for_denorm, y_std_for_denorm, basin_names,
            output_npy_dir=output_npy_dir
        )
        total_filled += filled_count

    if output_file is None:
        output_file = npz_file

    print(f"\nSaving filled data to {output_file}...")
    np.savez_compressed(output_file, **data_dict)

 
    print("Summary (RAW Pipeline - VAE DENORMALIZATION)")
    print(f"  Arrays modified: y_obs_trn, y_obs_val, y_obs_tst")
    print(f"  Arrays unchanged: y_raw_trn, y_raw_val, y_raw_tst (ground truth)")
    print(f"  Denormalization: y_mean_vae/y_std_vae (all basins use VAE stats)")
    print(f"  y_obs_* content: ALL basins = LSTM predictions (denormalized)")
    print(f"  Total samples filled: {total_filled}")
    if output_npy_dir:
        print(f"  Denormalized .npy files saved to: {output_npy_dir}")
    print(f"")
    print(f"  Stage 2 will use:")
    print(f"    - Input: y_obs_* (LSTM predictions, denormalized with VAE stats)")
    print(f"    - Label: y_raw (non-masked) / y_obs_* (masked)")
    print(f"    - Evaluation: y_raw_* as ground truth")


    return masked_basin_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fill y_obs_* with denormalized LSTM predictions (RAW pipeline)'
    )
    parser.add_argument('basins', nargs='+', help='Basin IDs that were masked (for Stage 2 label)')
    parser.add_argument('--npz_file', type=str, default=None)
    parser.add_argument('--pred_dir', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--output_npy_dir', type=str, default=None,
                        help='Directory to save denormalized .npy files for evaluation')
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    imputation_dir = os.path.dirname(current_dir)

    if args.npz_file is None:
        args.npz_file = os.path.join(imputation_dir, 'data_processing', 'data', 'prepped.npz')
    if args.pred_dir is None:
        args.pred_dir = os.path.join(current_dir, 'output', 'preds')
    # output_npy_dir defaults to None (no .npy output)


    print("LSTM -> NPZ Fill Script (RAW Pipeline - VAE DENORMALIZATION)")
    print(f"NPZ file: {args.npz_file}")
    print(f"Prediction directory (normalized): {args.pred_dir}")
    if args.output_npy_dir:
        print(f"Output npy directory (denormalized): {args.output_npy_dir}")
    print(f"Masked basins (for Stage 2 label): {args.basins}")
    print(f"Denormalization: y_mean_vae/y_std_vae (all basins use VAE stats)")


    if not os.path.exists(args.npz_file):
        print(f"ERROR: NPZ file not found: {args.npz_file}")
        sys.exit(1)

    if not os.path.exists(args.pred_dir):
        print(f"ERROR: Prediction directory not found: {args.pred_dir}")
        sys.exit(1)

    fill_prepped_npz_raw(args.npz_file, args.pred_dir, args.basins, args.output_file, args.output_npy_dir)

    print("\nDone! y_obs_* now contains LSTM predictions (denormalized with VAE stats) for ALL basins.")
    if args.output_npy_dir:
        print(f"Denormalized .npy files saved to: {args.output_npy_dir}")
    print("y_raw_* remains unchanged as ground truth.")
