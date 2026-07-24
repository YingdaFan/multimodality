"""
Script to set multiple basins' observations to NaN in the training data
Enhanced version supporting multiple basins
"""

import numpy as np
import sys
import os

def set_basin_to_nan(npz_file, basin_names, output_file=None):
    """
    Set all training observations for specific basin(s) to NaN

    Args:
        npz_file: Path to the input npz file
        basin_names: Name of the basin to modify, or list of basin names
        output_file: Path to save modified data (if None, overwrites input file)
    """
    # Convert single basin to list
    if isinstance(basin_names, str):
        basin_names = [basin_names]

    # Load the data
    data = np.load(npz_file, allow_pickle=True)

    # Convert to dict for modification
    data_dict = {key: data[key] for key in data.files}

    # Get the training IDs and observations
    ids_trn = data_dict['ids_trn']
    y_obs_trn = data_dict['y_obs_trn'].copy()  # Make a copy to modify

    # Process each basin
    total_samples = 0
    masked_basins = []

    for basin_name in basin_names:
        # Find all samples that belong to the target basin
        # Since each sample has shape [365, 1] for ids, we need to check all positions
        # But all 365 positions in a sample should have the same basin ID
        basin_mask = np.zeros(len(ids_trn), dtype=bool)

        for i in range(len(ids_trn)):
            # Check if this sample belongs to the target basin
            # Convert both to string for safe comparison (handles int vs str mismatch)
            if str(ids_trn[i, 0, 0]) == str(basin_name):
                basin_mask[i] = True

        n_samples = np.sum(basin_mask)

        if n_samples == 0:
            print(f"Warning: No samples found for basin '{basin_name}'")
            continue

        # Set those samples to NaN
        y_obs_trn[basin_mask] = np.nan
        total_samples += n_samples
        masked_basins.append(basin_name)

    # Update the dictionary
    data_dict['y_obs_trn'] = y_obs_trn

    # Save the modified data
    if output_file is None:
        output_file = npz_file

    np.savez_compressed(output_file, **data_dict)

    # Print summary info
    print(f"Shape of y_obs_trn: {y_obs_trn.shape}")
    print(f"Number of basins masked: {len(masked_basins)}")
    print(f"Total samples set to NaN: {total_samples}")
    print(f"Masked basins: {', '.join(masked_basins)}")


def list_basins(npz_file):
    """List all available basins in the data"""
    data = np.load(npz_file, allow_pickle=True)
    ids_trn = data['ids_trn']
    basin_names = np.unique(ids_trn[:, 0, 0])
    print(f"Available basins ({len(basin_names)}):")
    for basin in basin_names:
        print(f"  - {basin}")
    return basin_names


if __name__ == "__main__":
    # Default paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    input_file = os.path.join(data_dir, 'prepped.npz')

    if len(sys.argv) < 2:

        sys.exit(1)

    if sys.argv[1] == '--list':
        list_basins(input_file)
    else:
        # Collect all basin names (all args except possibly the last one if it ends with .npz)
        if sys.argv[-1].endswith('.npz'):
            basin_names = sys.argv[1:-1]
            output_file = sys.argv[-1]
        else:
            basin_names = sys.argv[1:]
            output_file = None

        set_basin_to_nan(input_file, basin_names, output_file)
