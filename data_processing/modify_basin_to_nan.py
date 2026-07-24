"""
Script to set multiple basins' observations to NaN in the training data,
but keep a specified year of data and recalculate mean/std based on that year
"""

import numpy as np
import pandas as pd
import sys
import os

def set_basin_to_nan_keep_year(npz_file, basin_names, keep_year='last', output_file=None):
    """
    Set training observations for multiple basin(s) to NaN except for a specified year,
    and recalculate mean/std for those basins based on that year's data

    Args:
        npz_file: Path to the input npz file
        basin_names: Name of the basin to modify, or list of basin names
        keep_year: Which year to keep. Options:
                   - 'last': Keep the last year (default)
                   - 'first': Keep the first year
                   - int: Keep specific year (e.g., 2005)
                   - 'year_N': Keep the Nth year (e.g., 'year_3' keeps the 3rd year)
                   - 'none': Mask all years (no data kept)
        output_file: Path to save modified data (if None, overwrites input file)
    """
    # Convert single basin to list
    if isinstance(basin_names, str):
        basin_names = [basin_names]
    # Load the data
    print(f"Loading data from {npz_file}...")
    data = np.load(npz_file, allow_pickle=True)

    # Convert to dict for modification
    data_dict = {key: data[key] for key in data.files}

    # Get the training IDs, observations, and times
    ids_trn = data_dict['ids_trn']
    y_obs_trn = data_dict['y_obs_trn'].copy()  # Make a copy to modify
    times_trn = data_dict['times_trn']

    # Get basin names array
    all_basin_names = data_dict['basin_names']

    # Make copies for modification
    y_mean_updated = data_dict['y_mean'].copy()
    y_std_updated = data_dict['y_std'].copy()

    total_masked = 0
    total_kept = 0
    processed_basins = []

    # Process each basin
    for basin_name in basin_names:
        print(f"\n{'='*60}")
        print(f"Processing basin: {basin_name}")
        print(f"{'='*60}")

        # Find basin index in basin_names array
        basin_idx = np.where(all_basin_names == basin_name)[0]

        if len(basin_idx) == 0:
            print(f"Warning: Basin '{basin_name}' not found in basin_names")
            print("Available basins:", all_basin_names[:10], "...")
            continue

        basin_idx = basin_idx[0]
        print(f"Found basin '{basin_name}' at index {basin_idx}")

        # Find all samples that belong to the target basin
        print(f"Finding samples for basin: {basin_name}")
        basin_mask = np.zeros(len(ids_trn), dtype=bool)

        for i in range(len(ids_trn)):
            # Check if this sample belongs to the target basin
            if ids_trn[i, 0, 0] == basin_name:
                basin_mask[i] = True

        n_samples = np.sum(basin_mask)
        if n_samples == 0:
            print(f"Warning: No samples found for basin '{basin_name}'")
            continue

        print(f"Found {n_samples} samples for basin {basin_name}")
    
        # Get all unique years from the training times for this basin
        basin_indices = np.where(basin_mask)[0]
        all_years = []

        for idx in basin_indices:
            # Get the year from the first date of each sample
            year = pd.Timestamp(times_trn[idx, 0, 0]).year
            all_years.append(year)

        unique_years = sorted(list(set(all_years)))
        print(f"Available years: {unique_years}")

        # Determine which year to keep
        if keep_year == 'none':
            target_year = None
            print("Masking all years (no data kept)")
        elif keep_year == 'last':
            target_year = unique_years[-1]
            print(f"Keeping last year: {target_year}")
        elif keep_year == 'first':
            target_year = unique_years[0]
            print(f"Keeping first year: {target_year}")
        elif isinstance(keep_year, int):
            if keep_year in unique_years:
                target_year = keep_year
                print(f"Keeping year: {target_year}")
            else:
                print(f"Error: Year {keep_year} not found in available years: {unique_years}")
                continue
        elif isinstance(keep_year, str) and keep_year.startswith('year_'):
            try:
                year_idx = int(keep_year.split('_')[1]) - 1  # Convert to 0-based index
                if 0 <= year_idx < len(unique_years):
                    target_year = unique_years[year_idx]
                    print(f"Keeping year #{year_idx+1}: {target_year}")
                else:
                    print(f"Error: Year index {year_idx+1} out of range (1-{len(unique_years)})")
                    continue
            except:
                print(f"Error: Invalid year specification '{keep_year}'")
                continue
        else:
            print(f"Error: Invalid keep_year parameter '{keep_year}'")
            print("Valid options: 'last', 'first', 'none', specific year (e.g., 2005), or 'year_N' (e.g., 'year_3')")
            continue

        # Get original mean and std for this basin
        original_mean = data_dict['y_mean'][basin_idx]
        original_std = data_dict['y_std'][basin_idx]

        # Collect target year's data indices and denormalized data
        target_year_indices = []
        target_year_data = []

        # Process all samples for this basin
        n_masked_basin = 0
        n_kept_basin = 0

        for idx in basin_indices:
            # Get the year of this sample
            sample_year = pd.Timestamp(times_trn[idx, 0, 0]).year

            if target_year is not None and sample_year == target_year:
                # Keep this year's data
                target_year_indices.append(idx)
                # Denormalize the data back to original scale
                normalized_data = y_obs_trn[idx].copy()
                denormalized_data = normalized_data * original_std + original_mean
                target_year_data.append(denormalized_data)
                n_kept_basin += 1
            else:
                # Mask this sample
                y_obs_trn[idx] = np.nan
                n_masked_basin += 1

        # Calculate new mean and std if we kept any data
        if len(target_year_data) > 0:
            target_year_data_concat = np.concatenate(target_year_data, axis=0)
            valid_data = target_year_data_concat[~np.isnan(target_year_data_concat)]

            if len(valid_data) > 0:
                new_mean = np.mean(valid_data)
                new_std = np.std(valid_data)
                if new_std < 1e-8:
                    new_std = 1.0
            else:
                new_mean = 0.0
                new_std = 1.0

            # Re-normalize the target year's data using the NEW mean and std
            print(f"Re-normalizing year {target_year}'s data with new statistics...")
            for i, idx in enumerate(target_year_indices):
                denormalized = target_year_data[i]
                renormalized = (denormalized - new_mean) / (new_std + 1e-10)
                y_obs_trn[idx] = renormalized

        else:
            # No data kept, use default or original statistics
            if keep_year == 'none':
                print("No data kept, using default mean=0, std=1")
                new_mean = 0.0
                new_std = 1.0
            else:
                print("Warning: No data found for target year, using default mean=0, std=1")
                new_mean = 0.0
                new_std = 1.0

        print(f"Original mean: {original_mean:.4f}, std: {original_std:.4f}")
        print(f"New mean (year {target_year if target_year else 'none'}): {new_mean:.4f}, std: {new_std:.4f}")
        print(f"Samples masked: {n_masked_basin}, kept: {n_kept_basin}")

        # Update the y_mean and y_std arrays
        y_mean_updated[basin_idx] = new_mean
        y_std_updated[basin_idx] = new_std

        total_masked += n_masked_basin
        total_kept += n_kept_basin
        processed_basins.append(basin_name)
    
    # Update the dictionary
    data_dict['y_obs_trn'] = y_obs_trn
    data_dict['y_mean'] = y_mean_updated
    data_dict['y_std'] = y_std_updated

    # Save the modified data
    if output_file is None:
        output_file = npz_file


    np.savez_compressed(output_file, **data_dict)

    print("Done!")





def list_basins_and_years(npz_file):
    """List all available basins and their years in the data"""
    import pandas as pd
    
    data = np.load(npz_file, allow_pickle=True)
    ids_trn = data['ids_trn']
    times_trn = data['times_trn']
    
    basin_names = np.unique(ids_trn[:, 0, 0])
    print(f"Available basins ({len(basin_names)}):")
    
    for basin in basin_names[:5]:  # Show first 5 basins as example
        # Find samples for this basin
        basin_mask = np.array([ids_trn[i, 0, 0] == basin for i in range(len(ids_trn))])
        basin_indices = np.where(basin_mask)[0]
        
        # Get years
        years = []
        for idx in basin_indices:
            year = pd.Timestamp(times_trn[idx, 0, 0]).year
            years.append(year)
        unique_years = sorted(list(set(years)))
        
        print(f"  - {basin}: years {unique_years}")
    
    if len(basin_names) > 5:
        print(f"  ... and {len(basin_names) - 5} more basins")
    
    return basin_names


def verify_modification(npz_file, basin_name):
    """Verify the modification was applied correctly"""
    import pandas as pd
    
    print(f"\nVerifying modification for basin '{basin_name}'...")
    data = np.load(npz_file, allow_pickle=True)
    
    ids_trn = data['ids_trn']
    y_obs_trn = data['y_obs_trn']
    times_trn = data['times_trn']
    y_mean = data['y_mean']
    y_std = data['y_std']
    basin_names = data['basin_names']
    
    # Find basin index
    basin_idx = np.where(basin_names == basin_name)[0]
    if len(basin_idx) == 0:
        print(f"Basin '{basin_name}' not found")
        return
    basin_idx = basin_idx[0]
    
    # Find samples for this basin
    basin_mask = np.array([ids_trn[i, 0, 0] == basin_name for i in range(len(ids_trn))])
    basin_indices = np.where(basin_mask)[0]
    
    # Group by year
    year_info = {}
    for idx in basin_indices:
        year = pd.Timestamp(times_trn[idx, 0, 0]).year
        is_nan = np.all(np.isnan(y_obs_trn[idx]))
        
        if year not in year_info:
            year_info[year] = {'total': 0, 'nan': 0, 'valid': 0}
        
        year_info[year]['total'] += 1
        if is_nan:
            year_info[year]['nan'] += 1
        else:
            year_info[year]['valid'] += 1
    
    print(f"Basin '{basin_name}' statistics:")
    print(f"  Total samples: {len(basin_indices)}")
    print(f"  Mean: {y_mean[basin_idx]:.4f}")
    print(f"  Std: {y_std[basin_idx]:.4f}")
    print(f"\n  Year-by-year breakdown:")
    
    for year in sorted(year_info.keys()):
        info = year_info[year]
        status = "KEPT" if info['valid'] > 0 else "MASKED"
        print(f"    {year}: {info['valid']} valid, {info['nan']} NaN ({status})")


if __name__ == "__main__":
    import pandas as pd

    # Default paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    input_file = os.path.join(data_dir, 'prepped.npz')

    if len(sys.argv) < 2:
        sys.exit(1)

    if sys.argv[1] == '--list':
        list_basins_and_years(input_file)
    elif sys.argv[1] == '--verify':
        if len(sys.argv) < 3:
            print("Please specify basin name to verify")
            sys.exit(1)

        # Allow specifying which file to verify
        verify_file = sys.argv[3] if len(sys.argv) > 3 else input_file
        verify_modification(verify_file, sys.argv[2])
    else:

        # Check if last arg is output file
        if sys.argv[-1].endswith('.npz'):
            output_file = sys.argv[-1]
            remaining_args = sys.argv[1:-1]
        else:
            output_file = None
            remaining_args = sys.argv[1:]

        # Helper function to check if an argument looks like keep_year
        def is_keep_year(arg):
            if arg in ['last', 'first', 'none']:
                return True
            if arg.startswith('year_'):
                return True
            try:
                int(arg)
                return True
            except ValueError:
                return False

        # Check if last remaining arg looks like keep_year
        if len(remaining_args) > 1 and is_keep_year(remaining_args[-1]):
            keep_year = remaining_args[-1]
            basin_names = remaining_args[:-1]
        else:
            # No keep_year specified, use default 'last'
            keep_year = 'last'
            basin_names = remaining_args

        # Convert keep_year to appropriate type
        try:
            keep_year = int(keep_year)
        except ValueError:
            pass  # It's a string like 'last', 'first', etc.

        if len(basin_names) == 0:
            print("Error: No basin names provided")
            sys.exit(1)

        print(f"Basin names: {basin_names}")
        print(f"Keep year: {keep_year}")
        print(f"Output file: {output_file if output_file else '(overwrite input)'}")
        print()

        set_basin_to_nan_keep_year(input_file, basin_names, keep_year, output_file)