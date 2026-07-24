#!/usr/bin/env python3
"""
Use VAE to predict y_mean and y_std for masked basins to avoid information leakage.
Calls spatial_extrapolation/vae_basin_flow_model.py
"""

import numpy as np
import sys
import os

# Add spatial_extrapolation path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 '../spatial_extrapolation'))

from vae_basin_flow_model import apply_vae_to_basin_prediction
import torch


def detect_masked_basins(npz_file):
    """Detect which basins are masked"""
    data = np.load(npz_file, allow_pickle=True)

    basin_names = data['basin_names']
    y_obs_trn = data['y_obs_trn']
    ids_trn = data['ids_trn']

    masked_basin_indices = []
    masked_basin_names = []

    for basin_idx, basin_name in enumerate(basin_names):
        # Find all samples for this basin
        basin_mask = (ids_trn[:, 0, 0] == basin_name)

        if np.sum(basin_mask) > 0:
            basin_y = y_obs_trn[basin_mask]
            # Check if all values are NaN
            if np.all(np.isnan(basin_y)):
                masked_basin_indices.append(basin_idx)
                masked_basin_names.append(basin_name)

    return masked_basin_indices, masked_basin_names


def update_npz_with_vae_predictions(npz_file, masked_basin_names, output_file=None):
    """
    Use VAE to predict Y statistics for masked basins and update npz file

    Parameters:
    -----------
    npz_file : str
        Path to preprocessed npz file
    masked_basin_names : list
        List of masked basin names
    output_file : str, optional
        Output file path (default: overwrite original file)
    """

    if len(masked_basin_names) == 0:
        print("No masked basins detected, skipping VAE prediction")
        return

    print(f"\nDetected {len(masked_basin_names)} masked basins:")
    print(f"  {masked_basin_names}")

    # TODO: Call your VAE model
    # Here you need to call based on the actual interface of your vae_basin_flow_model.py
    # Example:
    # model, predictions = apply_vae_to_basin_prediction()

    print("
Warning: VAE interface not fully implemented")
    print("Please complete this function based on the vae_basin_flow_model.py interface")

    # The following is an example code framework:
    # 1. Call VAE model to get predictions
    # 2. Update y_mean and y_std for the corresponding basins in the npz file

    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Use VAE to update statistics for masked basins')
    parser.add_argument('--npz_file', type=str,
                       default='data/prepped.npz',
                       help='Path to preprocessed npz file')
    parser.add_argument('--output_file', type=str,
                       default=None,
                       help='Output file path (default: overwrite input file)')

    args = parser.parse_args()

    # Get current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Handle relative paths
    if not os.path.isabs(args.npz_file):
        npz_file = os.path.join(current_dir, args.npz_file)
    else:
        npz_file = args.npz_file

    if args.output_file and not os.path.isabs(args.output_file):
        output_file = os.path.join(current_dir, args.output_file)
    else:
        output_file = args.output_file

    print("=" * 80)
    print("Use VAE to predict Y statistics for masked basins")
    print("=" * 80)
    print(f"Input file: {npz_file}")

    # Detect masked basins
    masked_indices, masked_names = detect_masked_basins(npz_file)

    if len(masked_names) > 0:
        # Use VAE to predict and update
        update_npz_with_vae_predictions(npz_file, masked_names, output_file)
    else:
        print("
No masked basins detected, no update needed")
