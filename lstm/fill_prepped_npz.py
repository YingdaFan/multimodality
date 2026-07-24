#!/usr/bin/env python3


import numpy as np
import argparse
import os
import sys


def fill_partition(data_dict, partition, pred_file, basin_names):
    """
    Fill y_obs_{partition} for ALL basins with LSTM predictions (normalized).

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing npz data (will be modified in-place)
    partition : str
        'trn', 'val', or 'tst'
    pred_file : str
        Path to the LSTM prediction file ({partition}.npy)
    basin_names : list of str
        List of all basin names

    Returns:
    --------
    tuple: (total_filled, basin_names)
    """
    y_key = f'y_obs_{partition}'
    ids_key = f'ids_{partition}'

    if y_key not in data_dict:
        print(f"  Warning: {y_key} not found in npz, skipping...")
        return 0, []

    if not os.path.exists(pred_file):
        print(f"  Warning: Prediction file {pred_file} not found, skipping...")
        return 0, []

    print(f"\n  Loading LSTM predictions from {pred_file}...")
    lstm_pred = np.load(pred_file)

    ids = data_dict[ids_key]

    # Verify shapes match
    if data_dict[y_key].shape != lstm_pred.shape:
        print(f"  Warning: Shape mismatch: {y_key} {data_dict[y_key].shape} vs lstm_pred {lstm_pred.shape}")
        return 0, []

    n_samples = len(ids)
    n_basins = len(basin_names)
    n_windows = n_samples // n_basins

    print(f"  {partition}: {n_windows} windows x {n_basins} basins = {n_samples} samples")

    # Fill ALL basins with LSTM predictions (normalized)
    # Simply replace y_obs_* with lstm_pred directly
    data_dict[y_key] = lstm_pred.copy()

    print(f"  {partition}: y_obs_* filled with normalized LSTM predictions ({n_samples} samples)")
    return n_samples, list(basin_names)


def fill_prepped_npz(npz_file, pred_dir, masked_basin_names, output_file=None):
    """
    Fill y_obs_trn/val/tst for ALL basins with LSTM predictions (normalized).

    Parameters:
    -----------
    npz_file : str
        Path to the prepped.npz file
    pred_dir : str
        Directory containing LSTM predictions (trn.npy, val.npy, tst.npy)
    masked_basin_names : list of str
        List of basin names that were masked (for Stage 2 label reference)
    output_file : str, optional
        Output file path. If None, overwrites npz_file.
    """
    # Convert single basin to list
    if isinstance(masked_basin_names, str):
        masked_basin_names = [masked_basin_names]

    print(f"Loading npz data from {npz_file}...")
    data = np.load(npz_file, allow_pickle=True)

    # Convert to dict for modification
    data_dict = {key: data[key] for key in data.files}

    basin_names = data_dict['basin_names']
    print(f"Number of basins: {len(basin_names)}")
    print(f"Masked basins (for Stage 2 label): {masked_basin_names}")
    print(f"\nNOTE: ALL basins will be filled with normalized LSTM predictions")

    # Fill all three partitions
    total_filled = 0
    partitions = ['trn', 'val', 'tst']

    for partition in partitions:
        pred_file = os.path.join(pred_dir, f'{partition}.npy')
        filled_count, _ = fill_partition(data_dict, partition, pred_file, basin_names)
        total_filled += filled_count

    # Verify y_raw_* are not modified (sanity check)
    print("\n  Verifying y_raw_* arrays are unchanged...")
    for partition in partitions:
        raw_key = f'y_raw_{partition}'
        if raw_key in data_dict:
            original = np.load(npz_file, allow_pickle=True)[raw_key]
            if np.array_equal(data_dict[raw_key], original, equal_nan=True):
                print(f"  ✓ {raw_key} unchanged (will be used for evaluation)")
            else:
                print(f"  ✗ WARNING: {raw_key} was modified!")
        else:
            print(f"  ⚠ {raw_key} not found in npz (run updated preprocess first)")

    # Save
    if output_file is None:
        output_file = npz_file

    print(f"\nSaving filled data to {output_file}...")
    np.savez_compressed(output_file, **data_dict)

    print("Summary (Standard Pipeline)")
    print(f"  Arrays modified: y_obs_trn, y_obs_val, y_obs_tst")
    print(f"  Arrays unchanged: y_raw_trn, y_raw_val, y_raw_tst (ground truth)")
    print(f"  y_obs_* content: ALL basins = LSTM predictions (normalized)")
    print(f"  Total samples filled: {total_filled}")
    print(f"")
    print(f"  Stage 2 will use:")
    print(f"    - Input: y_obs_* (LSTM predictions, normalized)")
    print(f"    - Label: y_raw (non-masked) / y_obs_* (masked)")
    print(f"    - Evaluation: y_raw_* as ground truth")


    return masked_basin_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fill y_obs_* with normalized LSTM predictions for ALL basins'
    )
    parser.add_argument('basins', nargs='+', help='Basin IDs that were masked (for Stage 2 label)')
    parser.add_argument('--npz_file', type=str, default=None,
                        help='Path to prepped.npz (default: ../data_processing/data/prepped.npz)')
    parser.add_argument('--pred_dir', type=str, default=None,
                        help='Directory containing LSTM predictions (default: ./output/preds/)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file path (default: overwrite npz_file)')
    args = parser.parse_args()

    # Set default paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    imputation_dir = os.path.dirname(current_dir)

    if args.npz_file is None:
        args.npz_file = os.path.join(imputation_dir, 'data_processing', 'data', 'prepped.npz')

    if args.pred_dir is None:
        args.pred_dir = os.path.join(current_dir, 'output', 'preds')


    print("LSTM -> NPZ Fill Script (Standard Pipeline)")
    print(f"NPZ file: {args.npz_file}")
    print(f"Prediction directory (normalized): {args.pred_dir}")
    print(f"Masked basins (for Stage 2 label): {args.basins}")
    print(f"Fill mode: ALL basins = LSTM predictions (normalized)")


    # Verify files exist
    if not os.path.exists(args.npz_file):
        print(f"ERROR: NPZ file not found: {args.npz_file}")
        sys.exit(1)

    if not os.path.exists(args.pred_dir):
        print(f"ERROR: Prediction directory not found: {args.pred_dir}")
        sys.exit(1)

    # Run fill
    fill_prepped_npz(args.npz_file, args.pred_dir, args.basins, args.output_file)

    print("\nDone! y_obs_* now contains normalized LSTM predictions for ALL basins.")
    print("y_raw_* arrays preserved as ground truth for evaluation.")
