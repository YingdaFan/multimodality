"""
CAMELS Data Preprocessing Script - All-Basin Normalization with Time Alignment
Uses time-aligned batch organization AND all-basin (global) normalization for Y
Uses original CSV column names directly without renaming
"""

import pandas as pd
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import os




def load_camels_data(csv_path):
    """Load CAMELS CSV data and convert to xarray dataset.
    Uses original column names directly without renaming."""
    df = pd.read_parquet(csv_path)

    # Convert Time to datetime
    df['Time'] = pd.to_datetime(df['Time'])

    # Calculate mean temperature (from tmax and tmin)
    df['tmean(C)'] = (df['tmax(C)'] + df['tmin(C)']) / 2.0

    # Get unique basins
    basins = df['basin_id'].unique()
    print(f"Found {len(basins)} basins")

    # Create xarray dataset with proper dimensions
    df_pivot = df.pivot(index='Time', columns='basin_id')

    # Get time and basin coordinates
    times = df_pivot.index.values
    basin_names = df_pivot['QObs(mm/d)'].columns.values

    # Create data arrays for each variable
    data_vars = {}

    # Dynamic features (time-varying)
    # Using original column names from CSV
    dynamic_vars = [
        'prcp(mm/day)',        # precipitation
        'srad(W/m2)',          # solar radiation
        'tmax(C)',             # max temperature
        'tmin(C)',             # min temperature
        'vp(Pa)',              # vapor pressure
        'tmean(C)',            # mean temperature (calculated)
    ]

    for var in dynamic_vars:
        if var in df_pivot.columns.levels[0]:
            data_vars[var] = xr.DataArray(
                df_pivot[var].values,
                dims=['date', 'seg_id_nat'],
                coords={'date': times, 'seg_id_nat': basin_names}
            )

    # Target variable (using original column name)
    data_vars['QObs(mm/d)'] = xr.DataArray(
        df_pivot['QObs(mm/d)'].values,
        dims=['date', 'seg_id_nat'],
        coords={'date': times, 'seg_id_nat': basin_names}
    )

    # Static features (constant per basin, time-invariant)
    static_vars = [
        'latitude', 'longitude', 'elev_mean',
        # Climate-related
        'aridity', 'frac_snow', 'high_prec_dur', 'high_prec_freq',
        'low_prec_dur', 'low_prec_freq', 'p_mean', 'p_seasonality', 'pet_mean',
        # Soil-related
        'clay_frac', 'max_water_content', 'sand_frac', 'silt_frac',
        'soil_conductivity', 'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity',
        # Vegetation-related
        'frac_forest', 'gvf_diff', 'gvf_max', 'lai_diff', 'lai_max',
        # Topography/location-related
        'area_gages2', 'slope_mean',
        # Geology-related
        'carbonate_rocks_frac', 'geol_permeability'
    ]
    for var in static_vars:
        if var in df_pivot.columns.levels[0]:
            static_values = df.groupby('basin_id')[var].first().reindex(basin_names).values
            broadcasted = np.tile(static_values[np.newaxis, :], (len(times), 1))
            data_vars[var] = xr.DataArray(
                broadcasted,
                dims=['date', 'seg_id_nat'],
                coords={'date': times, 'seg_id_nat': basin_names}
            )

    # Create dataset
    ds = xr.Dataset(data_vars)

    return ds


def add_temporal_features(ds):
    """Add temporal features (cyclic encoding of month and day-of-year)"""
    times = pd.to_datetime(ds.date.values)
    n_basins = len(ds.seg_id_nat)

    month = times.month.values
    month_broadcasted = np.tile(month[:, np.newaxis], (1, n_basins))
    ds['month_sin'] = xr.DataArray(
        np.sin(2 * np.pi * month_broadcasted / 12),
        dims=['date', 'seg_id_nat'],
        coords={'date': ds.date, 'seg_id_nat': ds.seg_id_nat},
    )
    ds['month_cos'] = xr.DataArray(
        np.cos(2 * np.pi * month_broadcasted / 12),
        dims=['date', 'seg_id_nat'],
        coords={'date': ds.date, 'seg_id_nat': ds.seg_id_nat},
    )

    day_of_year = times.dayofyear.values
    doy_broadcasted = np.tile(day_of_year[:, np.newaxis], (1, n_basins))
    ds['doy_sin'] = xr.DataArray(
        np.sin(2 * np.pi * doy_broadcasted / 365),
        dims=['date', 'seg_id_nat'],
        coords={'date': ds.date, 'seg_id_nat': ds.seg_id_nat},
    )
    ds['doy_cos'] = xr.DataArray(
        np.cos(2 * np.pi * doy_broadcasted / 365),
        dims=['date', 'seg_id_nat'],
        coords={'date': ds.date, 'seg_id_nat': ds.seg_id_nat},
    )

    return ds


def add_cumulative_features(ds, window_sizes=[7, 14, 30]):
    """Add cumulative features (sliding window statistics)"""
    for window in window_sizes:
        ds[f'prcp_sum_{window}d'] = ds['prcp(mm/day)'].rolling(
            date=window, min_periods=1
        ).sum()
        ds[f'tmean_avg_{window}d'] = ds['tmean(C)'].rolling(
            date=window, min_periods=1
        ).mean()
        temp_positive = xr.where(ds['tmean(C)'] > 0, ds['tmean(C)'], 0)
        ds[f'melt_potential_{window}d'] = temp_positive.rolling(
            date=window, min_periods=1
        ).sum()

    return ds


def create_distance_matrix(ds, spatial_idx_name='seg_id_nat'):
    """Create distance matrix based on lat/lon coordinates"""
    # Get unique basins and their coordinates
    basins = ds[spatial_idx_name].values
    n_basins = len(basins)

    # Extract coordinates for each basin (use first time step since they're constant)
    coords = np.zeros((n_basins, 2))
    for i, basin in enumerate(basins):
        basin_data = ds.sel({spatial_idx_name: basin}).isel(date=0)
        coords[i, 0] = basin_data['latitude'].values
        coords[i, 1] = basin_data['longitude'].values

    # Calculate pairwise distances
    dist_matrix_raw = cdist(coords, coords, metric='euclidean')

    # Process distance matrix (following original prep_adj_matrix logic)
    adj = -dist_matrix_raw  # Negate distances

    # Calculate mean and std of non-zero elements
    mean_adj = np.mean(adj[adj != 0])
    std_adj = np.std(adj[adj != 0])

    # Normalize
    adj[adj != 0] = adj[adj != 0] - mean_adj
    adj[adj != 0] = adj[adj != 0] / std_adj
    adj[adj != 0] = 1 / (1 + np.exp(-adj[adj != 0]))

    # Add identity matrix and normalize
    I = np.eye(n_basins)
    A_hat = adj + I
    D = np.sum(A_hat, axis=1)
    D_inv = D ** -1.0
    D_inv = np.diag(D_inv)
    A_hat = np.matmul(D_inv, A_hat)

    return A_hat, mean_adj, std_adj




def convert_batch_reshape(data_array, seq_len=365, offset=1.0):
    """
    Time-aligned batch creation
    Input: data_array shape: [n_basins, n_times, n_features]
    Output: time-aligned batches [n_time_windows * n_basins, seq_len, n_features]
    """
    n_basins, n_times, n_features = data_array.shape

    # Calculate step size
    if offset > 1:
        step = int(offset)
    else:
        step = int(seq_len * offset)

    # Calculate number of time windows
    n_time_windows = (n_times - seq_len) // step + 1

    # Pre-allocate output array
    output_shape = (n_time_windows * n_basins, seq_len, n_features)
    batched_data = np.zeros(output_shape)

    # Organize data by time windows (time-aligned)
    sample_idx = 0
    for time_idx in range(n_time_windows):
        start_t = time_idx * step
        end_t = start_t + seq_len

        # For this time window, add all basin data
        for basin_idx in range(n_basins):
            batched_data[sample_idx] = data_array[basin_idx, start_t:end_t, :]
            sample_idx += 1

    return batched_data


def create_ids_times_arrays(basin_names, dates, n_time_windows, n_segs, seq_len, offset):
    """
    Create aligned IDs and times arrays for time-aligned batches
    """
    n_samples = n_time_windows * n_segs
    ids_array = np.empty((n_samples, seq_len), dtype=object)
    times_array = np.empty((n_samples, seq_len), dtype='datetime64[ns]')

    # Calculate step size
    if offset > 1:
        step = int(offset)
    else:
        step = int(seq_len * offset)

    sample_idx = 0
    for time_idx in range(n_time_windows):
        start_t = time_idx * step
        end_t = start_t + seq_len

        # Get dates for this time window
        window_dates = dates[start_t:end_t]

        # Set data for all basins in this time window
        for seg_idx in range(n_segs):
            ids_array[sample_idx, :] = basin_names[seg_idx]
            times_array[sample_idx, :] = window_dates
            sample_idx += 1

    return ids_array[..., np.newaxis], times_array[..., np.newaxis]


def prep_data(
    csv_file,
    train_dates,
    val_dates,
    test_dates,
    seq_len=365,
    offset=1.0,
    out_file='prepped.npz',
    camels_folder=None,
    exclude_basins=None,
    add_temporal=False,
    add_cumulative=True,
):
    """Preprocess CAMELS data - using global normalization and time-aligned batches

    Args:
        csv_file: Path to denormalized_camels_data_time.csv
        train_dates: Tuple of (start_date, end_date) for training
        val_dates: Tuple of (start_date, end_date) for validation
        test_dates: Tuple of (start_date, end_date) for testing
        seq_len: Sequence length for batches
        offset: Offset for batch creation
        out_file: Output file path
        camels_folder: Path to CAMELS folder for additional static features
        exclude_basins: List of basin names to exclude from Y normalization calculation
                       (these basins will still be normalized using the calculated params)
        add_temporal: Whether to add cyclic temporal features (month/doy sin/cos)
        add_cumulative: Whether to add rolling-window features (prcp sum, tmean avg, melt potential)
    """

    print("Loading CAMELS data...")
    ds = load_camels_data(csv_file)

    # Base dynamic features (using original column names)
    x_vars = [
        'prcp(mm/day)',      # precipitation
        'tmean(C)',          # mean temperature
        'tmax(C)',           # max temperature
        'tmin(C)',           # min temperature
        'srad(W/m2)',        # solar radiation
        'vp(Pa)',            # vapor pressure

        # Climate-related (9)
        'aridity',
        'frac_snow',
        'high_prec_dur',
        'high_prec_freq',
        'low_prec_dur',
        'low_prec_freq',
        'p_mean',
        'p_seasonality',
        'pet_mean',

        # Soil-related (8)
        'clay_frac',
        'max_water_content',
        'sand_frac',
        'silt_frac',
        'soil_conductivity',
        'soil_depth_pelletier',
        'soil_depth_statsgo',
        'soil_porosity',

        # Vegetation-related (5)
        'frac_forest',
        'gvf_diff',
        'gvf_max',
        'lai_diff',
        'lai_max',

        # Topography/location-related (3)
        'area_gages2',
        'elev_mean',
        'slope_mean',

        # Geology-related (2)
        'carbonate_rocks_frac',
        'geol_permeability'
    ]

    # Apply optional feature engineering BEFORE split so rolling windows are causal
    if add_temporal:
        print("Adding temporal features...")
        ds = add_temporal_features(ds)
        x_vars.extend(['month_sin', 'month_cos', 'doy_sin', 'doy_cos'])

    if add_cumulative:
        print("Adding cumulative features...")
        ds = add_cumulative_features(ds)
        x_vars.extend([
            'prcp_sum_7d', 'prcp_sum_14d', 'prcp_sum_30d',
            'tmean_avg_7d', 'tmean_avg_14d', 'tmean_avg_30d',
            'melt_potential_7d', 'melt_potential_14d', 'melt_potential_30d',
        ])

    # Target variable (using original column name)
    y_var = 'QObs(mm/d)'

    basin_names = ds['seg_id_nat'].values
    n_segs = len(basin_names)

    # Select train, val, test periods
    print("Splitting data into train/val/test...")
    ds_train = ds.sel(date=slice(train_dates[0], train_dates[1]))
    ds_val = ds.sel(date=slice(val_dates[0], val_dates[1]))
    ds_test = ds.sel(date=slice(test_dates[0], test_dates[1]))

    # Prepare X data
    x_train = ds_train[x_vars]
    x_val = ds_val[x_vars]
    x_test = ds_test[x_vars]

    # Calculate scaling parameters from training data
    print("Scaling features...")
    # x_train.to_array().values has shape (n_var, n_time, n_basin).
    # The previous version called .reshape(-1, n_var) directly, which mixes
    # variables into rows under C-order and produces near-identical mean/std
    # for every feature. Transpose to (n_time, n_basin, n_var) first so each
    # row of the flat matrix is one (time, basin) sample's full feature vector.
    x_train_flat = x_train.to_array().values.transpose(1, 2, 0).reshape(-1, len(x_vars))
    scaler_x = StandardScaler()
    scaler_x.fit(x_train_flat[~np.isnan(x_train_flat).any(axis=1)])

    # Scale X data
    x_mean = xr.Dataset({var: val for var, val in zip(x_vars, scaler_x.mean_)})
    x_std = xr.Dataset({var: val for var, val in zip(x_vars, scaler_x.scale_)})

    x_train_scaled = (x_train - x_mean) / (x_std + 1e-10)
    x_val_scaled = (x_val - x_mean) / (x_std + 1e-10)
    x_test_scaled = (x_test - x_mean) / (x_std + 1e-10)

    # Prepare Y data
    y_train = ds_train[[y_var]]
    y_val = ds_val[[y_var]]
    y_test = ds_test[[y_var]]

    # ===== ALL-BASIN NORMALIZATION (GLOBAL) =====
    # Calculate Y scaling from training data, optionally excluding certain basins
    print("Calculating all-basin (global) Y scaling parameters...")

    if exclude_basins is not None and len(exclude_basins) > 0:
        # Filter out excluded basins from normalization calculation
        print(f"  Excluding {len(exclude_basins)} basin(s) from Y normalization: {exclude_basins}")

        # Create a mask for basins to include
        include_mask = np.isin(basin_names, exclude_basins, invert=True)
        included_basins = basin_names[include_mask]
        print(f"  Using {len(included_basins)} basin(s) for Y normalization calculation")

        # Select only included basins for calculating normalization parameters
        y_train_for_norm = y_train[y_var].sel(seg_id_nat=included_basins)
        y_train_flat = y_train_for_norm.values.flatten()
        y_train_valid = y_train_flat[~np.isnan(y_train_flat)]

        if len(y_train_valid) == 0:
            raise ValueError("No valid data remaining after excluding basins!")

        y_mean_val = np.mean(y_train_valid)
        y_std_val = np.std(y_train_valid)
    else:
        # Use all basins for normalization calculation
        print("  Using all basins for Y normalization calculation")
        y_train_flat = y_train[y_var].values.flatten()
        y_train_valid = y_train_flat[~np.isnan(y_train_flat)]
        y_mean_val = np.mean(y_train_valid)
        y_std_val = np.std(y_train_valid)

    # Scale Y data using global parameters
    print("Applying all-basin Y scaling...")
    y_train_scaled = (y_train - y_mean_val) / (y_std_val + 1e-10)
    y_val_scaled = (y_val - y_mean_val) / (y_std_val + 1e-10)
    y_test_scaled = (y_test - y_mean_val) / (y_std_val + 1e-10)

    # Convert to numpy arrays
    print("Converting to numpy arrays...")
    x_train_arr = x_train_scaled.to_array().values.transpose(2, 1, 0)  # [seg, time, feat]
    x_val_arr = x_val_scaled.to_array().values.transpose(2, 1, 0)
    x_test_arr = x_test_scaled.to_array().values.transpose(2, 1, 0)

    y_train_arr = y_train_scaled[y_var].values.T[:, :, np.newaxis]  # [seg, time, 1]
    y_val_arr = y_val_scaled[y_var].values.T[:, :, np.newaxis]
    y_test_arr = y_test_scaled[y_var].values.T[:, :, np.newaxis]

    # Save raw Y values (not normalized, for ground truth evaluation)
    print("Creating raw Y arrays for evaluation...")
    y_train_raw_arr = y_train[y_var].values.T[:, :, np.newaxis]  # [seg, time, 1]
    y_val_raw_arr = y_val[y_var].values.T[:, :, np.newaxis]
    y_test_raw_arr = y_test[y_var].values.T[:, :, np.newaxis]

    # Create time-aligned batches
    print("Creating time-aligned batches...")
    x_trn_batched = convert_batch_reshape(x_train_arr, seq_len=seq_len, offset=offset)
    x_val_batched = convert_batch_reshape(x_val_arr, seq_len=seq_len, offset=1.0)
    x_tst_batched = convert_batch_reshape(x_test_arr, seq_len=seq_len, offset=1.0)

    y_trn_batched = convert_batch_reshape(y_train_arr, seq_len=seq_len, offset=offset)
    y_val_batched = convert_batch_reshape(y_val_arr, seq_len=seq_len, offset=1.0)
    y_tst_batched = convert_batch_reshape(y_test_arr, seq_len=seq_len, offset=1.0)

    # Raw Y value batches (for ground truth evaluation)
    y_raw_trn_batched = convert_batch_reshape(y_train_raw_arr, seq_len=seq_len, offset=offset)
    y_raw_val_batched = convert_batch_reshape(y_val_raw_arr, seq_len=seq_len, offset=1.0)
    y_raw_tst_batched = convert_batch_reshape(y_test_raw_arr, seq_len=seq_len, offset=1.0)

    # Calculate number of time windows
    step_trn = int(offset) if offset > 1 else int(seq_len * offset)
    step_val = int(seq_len * 1.0)
    step_tst = int(seq_len * 1.0)
    n_trn_windows = (x_train_arr.shape[1] - seq_len) // step_trn + 1
    n_val_windows = (x_val_arr.shape[1] - seq_len) // step_val + 1
    n_tst_windows = (x_test_arr.shape[1] - seq_len) // step_tst + 1

    # Create aligned IDs and times arrays
    print("Creating aligned ID and time arrays...")
    ids_trn, times_trn = create_ids_times_arrays(
        basin_names, ds_train.date.values, n_trn_windows, n_segs, seq_len, offset
    )
    ids_val, times_val = create_ids_times_arrays(
        basin_names, ds_val.date.values, n_val_windows, n_segs, seq_len, 1.0
    )
    ids_tst, times_tst = create_ids_times_arrays(
        basin_names, ds_test.date.values, n_tst_windows, n_segs, seq_len, 1.0
    )

    # Create distance matrix
    print("Creating distance matrix...")
    dist_matrix, dist_mean, dist_std = create_distance_matrix(ds)

    # Prepare output dictionary
    data_dict = {
        # Normalized data (for model training/inference)
        'x_trn': x_trn_batched,
        'y_obs_trn': y_trn_batched,
        'x_val': x_val_batched,
        'y_obs_val': y_val_batched,
        'x_tst': x_tst_batched,
        'y_obs_tst': y_tst_batched,
        # Raw Y values (for ground truth evaluation, unaffected by normalization)
        'y_raw_trn': y_raw_trn_batched,
        'y_raw_val': y_raw_val_batched,
        'y_raw_tst': y_raw_tst_batched,
        # Metadata
        'dist_matrix': dist_matrix,
        'x_vars': np.array(x_vars),
        'y_obs_vars': np.array([y_var]),
        'ids_trn': ids_trn,
        'ids_val': ids_val,
        'ids_tst': ids_tst,
        'times_trn': times_trn,
        'times_val': times_val,
        'times_tst': times_tst,
        'x_mean': scaler_x.mean_,
        'x_std': scaler_x.scale_,
        'y_mean': np.array([y_mean_val]),  # Single global value
        'y_std': np.array([y_std_val]),    # Single global value
        'dist_mean': dist_mean,
        'dist_std': dist_std,
        'n_segs': n_segs,
        'basin_names': basin_names,
        'exclude_basins': np.array(exclude_basins if exclude_basins is not None else [], dtype=object)
    }

    # Save to file
    print(f"Saving preprocessed data to {out_file}...")
    np.savez_compressed(out_file, **data_dict)

    # Print summary
    print("\nPreprocessing complete!")
    print(f"Number of basins: {n_segs}")
    print(f"Training samples: {x_trn_batched.shape[0]} ({n_trn_windows} time windows × {n_segs} basins)")
    print(f"Validation samples: {x_val_batched.shape[0]} ({n_val_windows} time windows × {n_segs} basins)")
    print(f"Test samples: {x_tst_batched.shape[0]} ({n_tst_windows} time windows × {n_segs} basins)")
    print(f"Features: {len(x_vars)}")
    print(f"Sequence length: {seq_len}")
    print(f"\nNormalization: ALL-BASIN (global)")
    if exclude_basins is not None and len(exclude_basins) > 0:
        print(f"  Excluded {len(exclude_basins)} basin(s) from normalization: {exclude_basins}")
        print(f"  Normalization based on {n_segs - len(exclude_basins)} basin(s)")
    else:
        print(f"  Normalization based on all {n_segs} basin(s)")
    print(f"Y mean: {y_mean_val:.4f}, Y std: {y_std_val:.4f}")

    # Verify time-aligned organization
    print("\nVerifying time-aligned organization:")
    print("Time window 0 (samples 0-{}):".format(n_segs-1))
    for i in range(min(n_segs, 5)):
        print(f"  Sample {i}: Basin={ids_trn[i, 0, 0]}, Start time={times_trn[i, 0, 0]}")

    if n_trn_windows > 1:
        print(f"\nTime window 1 (samples {n_segs}-{2*n_segs-1}):")
        for i in range(n_segs, min(n_segs + 5, 2*n_segs)):
            print(f"  Sample {i}: Basin={ids_trn[i, 0, 0]}, Start time={times_trn[i, 0, 0]}")

        # Verify time alignment in first window
        print("\nVerifying time alignment in first window:")
        first_window_time = times_trn[0, 0, 0]
        all_same = all(times_trn[i, 0, 0] == first_window_time for i in range(n_segs))
        print(f"  All basins in first window have same start time: {all_same}")

    return data_dict




if __name__ == "__main__":
    import sys

    # Get current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')

    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # CAMELS dataset time range: 1989-01-01 to 2007-12-31 (19 years)
    # Split: train (13 years) + val (3 years) + test (3 years)
    train_dates = ('1989-01-01', '2001-12-31')  # 13 years training data 
    val_dates = ('2002-01-01', '2004-12-31')    # 3 years validation data
    # test_dates = ('2002-01-01', '2004-12-31')   # 3 years test data
    test_dates = ('1989-01-01', '2001-12-31')   

    # Parse command line arguments for basins to exclude
    # Usage: python preprocess_allbasin_aligntime_camels.py [basin1] [basin2] ...
    exclude_basins_list = None
    if len(sys.argv) > 1:
        exclude_basins_list = sys.argv[1:]
        print(f"Command line arguments: Excluding basins {exclude_basins_list} from normalization")
    else:
        # Optional: specify basins to exclude from normalization calculation
        # These basins will still be included in the dataset but won't affect y_mean/y_std
        # Example: exclude_basins_list = ['01013500', '01022500', '01030500']
        exclude_basins_list = None  # Set to None to use all basins, or provide a list of basin IDs
        print("No command line arguments: Using all basins for normalization")

    # Generate time-aligned data with all-basin normalization
    prep_data(
        csv_file='../../denormalized_camels_data_time.parquet',
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        seq_len=365,
        offset=1.0,
        out_file=os.path.join(data_dir, 'prepped.npz'),
        exclude_basins=exclude_basins_list,
        add_temporal=False,      # set True to enable month_sin/cos + doy_sin/cos
        add_cumulative=True,     # 7/14/30d rolling features (prcp sum / tmean avg / melt potential)
    )
