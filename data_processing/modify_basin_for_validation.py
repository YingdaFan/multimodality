"""
Script to prepare validation set for spatial extrapolation:
Keep ONLY target_val basins' Y visible, mask all other basins' Y to NaN

This ensures validation set can properly evaluate spatial extrapolation ability
"""

import numpy as np
import sys
import os

def prepare_validation_set(npz_file, target_val_basins, output_file=None):
    """
    Prepare validation set: keep ONLY target_val basins' Y, mask others

    Args:
        npz_file: Path to the input npz file
        target_val_basins: List of basin names to keep visible in validation
        output_file: Path to save modified data (if None, overwrites input file)
    """
    # Convert single basin to list
    if isinstance(target_val_basins, str):
        target_val_basins = [target_val_basins]

    # Load the data
    print(f"Loading data from {npz_file}...")
    data = np.load(npz_file, allow_pickle=True)

    # Convert to dict for modification
    data_dict = {key: data[key] for key in data.files}

    # Get validation IDs and observations
    ids_val = data_dict['ids_val']
    y_obs_val = data_dict['y_obs_val'].copy()  # Make a copy to modify

    print(f"\nOriginal validation set shape: {y_obs_val.shape}")
    print(f"Target validation basins: {target_val_basins}")

    # Get all unique basins in validation set
    all_basins = np.unique(ids_val[:, 0, 0])
    print(f"All basins in validation set: {len(all_basins)}")

    # Strategy: Mask ALL basins first, then unmask target_val basins
    print("\nStep 1: Masking all basins...")
    y_obs_val[:] = np.nan

    print("Step 2: Unmasking target validation basins...")
    unmasked_count = 0
    for basin_name in target_val_basins:
        # Find all samples that belong to this validation basin
        basin_mask = np.zeros(len(ids_val), dtype=bool)

        for i in range(len(ids_val)):
            if ids_val[i, 0, 0] == basin_name:
                basin_mask[i] = True

        n_samples = np.sum(basin_mask)

        if n_samples == 0:
            print(f"  Warning: Basin '{basin_name}' not found in validation set")
            continue

        # Restore this basin's data from original
        y_obs_val[basin_mask] = data['y_obs_val'][basin_mask]
        unmasked_count += n_samples
        print(f"  ✓ Unmasked {n_samples} samples for basin {basin_name}")

    # Verify the modification
    n_visible_samples = np.sum(~np.all(np.isnan(y_obs_val), axis=(1, 2)))
    n_masked_samples = np.sum(np.all(np.isnan(y_obs_val), axis=(1, 2)))


    print(f"Validation Set Preparation Summary:")
    print(f"Total samples: {len(y_obs_val)}")
    print(f"Visible samples (target_val basins): {n_visible_samples}")
    print(f"Masked samples (other basins): {n_masked_samples}")
    print(f"Target validation basins kept: {len(target_val_basins)}")


    # Update the dictionary
    data_dict['y_obs_val'] = y_obs_val

    # Save the modified data
    if output_file is None:
        output_file = npz_file

    print(f"Saving modified data to {output_file}...")
    np.savez_compressed(output_file, **data_dict)

    print("✓ Validation set preparation complete!")

    return data_dict


def list_basins_in_partition(npz_file, partition='val'):
    """List all available basins in a specific partition"""
    data = np.load(npz_file, allow_pickle=True)
    ids_key = f'ids_{partition}'

    if ids_key not in data:
        print(f"Error: '{ids_key}' not found in data file")
        return None

    ids = data[ids_key]
    basin_names = np.unique(ids[:, 0, 0])
    print(f"Available basins in {partition} partition ({len(basin_names)}):")
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
        list_basins_in_partition(input_file, 'val')
    else:
        # Collect all basin names
        if sys.argv[-1].endswith('.npz'):
            target_val_basins = sys.argv[1:-1]
            output_file = sys.argv[-1]
        else:
            target_val_basins = sys.argv[1:]
            output_file = None

        prepare_validation_set(input_file, target_val_basins, output_file)
