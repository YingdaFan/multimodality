"""
Solar Energy Data Preprocessing Script - All-Basin Normalization with Time Alignment
Uses time-aligned batch organization AND all-basin (global) normalization for Y
Adapted from the CAMELS preprocessing script for the AMS Solar Energy dataset
"""

import pandas as pd
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import os


def load_solar_data(csv_path):
    """Load Solar CSV data and convert to xarray dataset"""
    df = pd.read_csv(csv_path, dtype={'station_id': str})

    # Convert Time to datetime
    df['Time'] = pd.to_datetime(df['Time'])

    # Get unique stations
    stations = df['station_id'].unique()
    print(f"Found {len(stations)} stations")

    # Create xarray dataset with proper dimensions
    df_pivot = df.pivot(index='Time', columns='station_id')

    # Get time and station coordinates
    times = df_pivot.index.values
    station_names = df_pivot['solar_energy_MJ'].columns.values

    # Create data arrays for each variable
    data_vars = {}

    # Dynamic features (time-varying) - 45 meteorological features
    # 15 variables x 3 statistics (max, min, mean)
    dynamic_vars = [
        # Precipitation
        'apcp_sfc_max', 'apcp_sfc_mean', 'apcp_sfc_min',
        # Downward longwave radiation
        'dlwrf_sfc_max', 'dlwrf_sfc_mean', 'dlwrf_sfc_min',
        # Downward shortwave radiation
        'dswrf_sfc_max', 'dswrf_sfc_mean', 'dswrf_sfc_min',
        # Mean sea level pressure
        'pres_msl_max', 'pres_msl_mean', 'pres_msl_min',
        # Precipitable water
        'pwat_eatm_max', 'pwat_eatm_mean', 'pwat_eatm_min',
        # 2m specific humidity
        'spfh_2m_max', 'spfh_2m_mean', 'spfh_2m_min',
        # Cloud cover
        'tcdc_eatm_max', 'tcdc_eatm_mean', 'tcdc_eatm_min',
        # Cloud liquid water
        'tcolc_eatm_max', 'tcolc_eatm_mean', 'tcolc_eatm_min',
        # 2m max temperature
        'tmax_2m_max', 'tmax_2m_mean', 'tmax_2m_min',
        # 2m min temperature
        'tmin_2m_max', 'tmin_2m_mean', 'tmin_2m_min',
        # 2m temperature
        'tmp_2m_max', 'tmp_2m_mean', 'tmp_2m_min',
        # Surface temperature
        'tmp_sfc_max', 'tmp_sfc_mean', 'tmp_sfc_min',
        # Upward longwave radiation (surface)
        'ulwrf_sfc_max', 'ulwrf_sfc_mean', 'ulwrf_sfc_min',
        # Upward longwave radiation (top of atmosphere)
        'ulwrf_tatm_max', 'ulwrf_tatm_mean', 'ulwrf_tatm_min',
        # Upward shortwave radiation
        'uswrf_sfc_max', 'uswrf_sfc_mean', 'uswrf_sfc_min',
    ]

    for var in dynamic_vars:
        if var in df_pivot.columns.levels[0]:
            data_vars[var] = xr.DataArray(
                df_pivot[var].values,
                dims=['date', 'seg_id_nat'],
                coords={'date': times, 'seg_id_nat': station_names}
            )

    # Target variable
    data_vars['solar_energy_MJ'] = xr.DataArray(
        df_pivot['solar_energy_MJ'].values,
        dims=['date', 'seg_id_nat'],
        coords={'date': times, 'seg_id_nat': station_names}
    )

    # Static features (constant per station, time-invariant)
    static_vars = ['lat', 'lon', 'elev']
    for var in static_vars:
        if var in df_pivot.columns.levels[0]:
            static_values = df.groupby('station_id')[var].first().reindex(station_names).values
            broadcasted = np.tile(static_values[np.newaxis, :], (len(times), 1))
            data_vars[var] = xr.DataArray(
                broadcasted,
                dims=['date', 'seg_id_nat'],
                coords={'date': times, 'seg_id_nat': station_names}
            )

    # Create dataset
    ds = xr.Dataset(data_vars)

    return ds


def create_distance_matrix(ds, spatial_idx_name='seg_id_nat'):
    """Create distance matrix based on lat/lon coordinates"""
    # Get unique stations and their coordinates
    stations = ds[spatial_idx_name].values
    n_stations = len(stations)

    # Extract coordinates for each station (use first time step since they're constant)
    coords = np.zeros((n_stations, 2))
    for i, station in enumerate(stations):
        station_data = ds.sel({spatial_idx_name: station}).isel(date=0)
        coords[i, 0] = station_data['lat'].values
        coords[i, 1] = station_data['lon'].values

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
    I = np.eye(n_stations)
    A_hat = adj + I
    D = np.sum(A_hat, axis=1)
    D_inv = D ** -1.0
    D_inv = np.diag(D_inv)
    A_hat = np.matmul(D_inv, A_hat)

    return A_hat, mean_adj, std_adj


def convert_batch_reshape(data_array, seq_len=365, offset=1.0):
    """
    Time-aligned batch creation
    Input: data_array shape: [n_stations, n_times, n_features]
    Output: time-aligned batches [n_time_windows * n_stations, seq_len, n_features]
    """
    n_stations, n_times, n_features = data_array.shape

    # Calculate step size
    if offset > 1:
        step = int(offset)
    else:
        step = int(seq_len * offset)

    # Calculate number of time windows
    n_time_windows = (n_times - seq_len) // step + 1

    # Pre-allocate output array
    output_shape = (n_time_windows * n_stations, seq_len, n_features)
    batched_data = np.zeros(output_shape)

    # Organize data by time windows (time-aligned)
    sample_idx = 0
    for time_idx in range(n_time_windows):
        start_t = time_idx * step
        end_t = start_t + seq_len

        # For this time window, add all station data
        for station_idx in range(n_stations):
            batched_data[sample_idx] = data_array[station_idx, start_t:end_t, :]
            sample_idx += 1

    return batched_data


def create_ids_times_arrays(station_names, dates, n_time_windows, n_segs, seq_len, offset):
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

        # Set data for all stations in this time window
        for seg_idx in range(n_segs):
            ids_array[sample_idx, :] = station_names[seg_idx]
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
    out_file='prepped_solar.npz',
    exclude_stations=None
):
    """Preprocess Solar data - using global normalization and time-aligned batches

    Args:
        csv_file: Path to solar_data.csv
        train_dates: Tuple of (start_date, end_date) for training
        val_dates: Tuple of (start_date, end_date) for validation
        test_dates: Tuple of (start_date, end_date) for testing
        seq_len: Sequence length for batches
        offset: Offset for batch creation
        out_file: Output file path
        exclude_stations: List of station names to exclude from Y normalization calculation
                         (these stations will still be normalized using the calculated params)
    """

    print("Loading Solar data...")
    ds = load_solar_data(csv_file)

    # X variables: 45 dynamic meteorological features + 3 static features
    x_vars = [
        # Dynamic features (45)
        'apcp_sfc_max', 'apcp_sfc_mean', 'apcp_sfc_min',
        'dlwrf_sfc_max', 'dlwrf_sfc_mean', 'dlwrf_sfc_min',
        'dswrf_sfc_max', 'dswrf_sfc_mean', 'dswrf_sfc_min',
        'pres_msl_max', 'pres_msl_mean', 'pres_msl_min',
        'pwat_eatm_max', 'pwat_eatm_mean', 'pwat_eatm_min',
        'spfh_2m_max', 'spfh_2m_mean', 'spfh_2m_min',
        'tcdc_eatm_max', 'tcdc_eatm_mean', 'tcdc_eatm_min',
        'tcolc_eatm_max', 'tcolc_eatm_mean', 'tcolc_eatm_min',
        'tmax_2m_max', 'tmax_2m_mean', 'tmax_2m_min',
        'tmin_2m_max', 'tmin_2m_mean', 'tmin_2m_min',
        'tmp_2m_max', 'tmp_2m_mean', 'tmp_2m_min',
        'tmp_sfc_max', 'tmp_sfc_mean', 'tmp_sfc_min',
        'ulwrf_sfc_max', 'ulwrf_sfc_mean', 'ulwrf_sfc_min',
        'ulwrf_tatm_max', 'ulwrf_tatm_mean', 'ulwrf_tatm_min',
        'uswrf_sfc_max', 'uswrf_sfc_mean', 'uswrf_sfc_min',
        # Static features (3)
        'lat', 'lon', 'elev',
    ]

    # Target variable
    y_var = 'solar_energy_MJ'

    station_names = ds['seg_id_nat'].values
    n_segs = len(station_names)

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
    x_train_flat = x_train.to_array().values.reshape(-1, len(x_vars))
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

    # ===== ALL-STATION NORMALIZATION (GLOBAL) =====
    # Calculate Y scaling from training data, optionally excluding certain stations
    print("Calculating all-station (global) Y scaling parameters...")

    if exclude_stations is not None and len(exclude_stations) > 0:
        # Filter out excluded stations from normalization calculation
        print(f"  Excluding {len(exclude_stations)} station(s) from Y normalization: {exclude_stations}")

        # Create a mask for stations to include
        include_mask = np.isin(station_names, exclude_stations, invert=True)
        included_stations = station_names[include_mask]
        print(f"  Using {len(included_stations)} station(s) for Y normalization calculation")

        # Select only included stations for calculating normalization parameters
        y_train_for_norm = y_train[y_var].sel(seg_id_nat=included_stations)
        y_train_flat = y_train_for_norm.values.flatten()
        y_train_valid = y_train_flat[~np.isnan(y_train_flat)]

        if len(y_train_valid) == 0:
            raise ValueError("No valid data remaining after excluding stations!")

        y_mean_val = np.mean(y_train_valid)
        y_std_val = np.std(y_train_valid)
    else:
        # Use all stations for normalization calculation
        print("  Using all stations for Y normalization calculation")
        y_train_flat = y_train[y_var].values.flatten()
        y_train_valid = y_train_flat[~np.isnan(y_train_flat)]
        y_mean_val = np.mean(y_train_valid)
        y_std_val = np.std(y_train_valid)

    # Scale Y data using global parameters
    print("Applying all-station Y scaling...")
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
        station_names, ds_train.date.values, n_trn_windows, n_segs, seq_len, offset
    )
    ids_val, times_val = create_ids_times_arrays(
        station_names, ds_val.date.values, n_val_windows, n_segs, seq_len, 1.0
    )
    ids_tst, times_tst = create_ids_times_arrays(
        station_names, ds_test.date.values, n_tst_windows, n_segs, seq_len, 1.0
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
        'station_names': station_names,
        'exclude_stations': np.array(exclude_stations if exclude_stations is not None else [], dtype=object)
    }

    # Save to file
    print(f"Saving preprocessed data to {out_file}...")
    np.savez_compressed(out_file, **data_dict)

    # Print summary
    print("\nPreprocessing complete!")
    print(f"Number of stations: {n_segs}")
    print(f"Training samples: {x_trn_batched.shape[0]} ({n_trn_windows} time windows × {n_segs} stations)")
    print(f"Validation samples: {x_val_batched.shape[0]} ({n_val_windows} time windows × {n_segs} stations)")
    print(f"Test samples: {x_tst_batched.shape[0]} ({n_tst_windows} time windows × {n_segs} stations)")
    print(f"Features: {len(x_vars)}")
    print(f"Sequence length: {seq_len}")
    print(f"\nNormalization: ALL-STATION (global)")
    if exclude_stations is not None and len(exclude_stations) > 0:
        print(f"  Excluded {len(exclude_stations)} station(s) from normalization: {exclude_stations}")
        print(f"  Normalization based on {n_segs - len(exclude_stations)} station(s)")
    else:
        print(f"  Normalization based on all {n_segs} station(s)")
    print(f"Y mean: {y_mean_val:.4f}, Y std: {y_std_val:.4f}")

    # Verify time-aligned organization
    print("\nVerifying time-aligned organization:")
    print("Time window 0 (samples 0-{}):".format(n_segs-1))
    for i in range(min(n_segs, 5)):
        print(f"  Sample {i}: Station={ids_trn[i, 0, 0]}, Start time={times_trn[i, 0, 0]}")

    if n_trn_windows > 1:
        print(f"\nTime window 1 (samples {n_segs}-{2*n_segs-1}):")
        for i in range(n_segs, min(n_segs + 5, 2*n_segs)):
            print(f"  Sample {i}: Station={ids_trn[i, 0, 0]}, Start time={times_trn[i, 0, 0]}")

        # Verify time alignment in first window
        print("\nVerifying time alignment in first window:")
        first_window_time = times_trn[0, 0, 0]
        all_same = all(times_trn[i, 0, 0] == first_window_time for i in range(n_segs))
        print(f"  All stations in first window have same start time: {all_same}")

    return data_dict


if __name__ == "__main__":
    import sys

    # Get current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')

    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Solar dataset time range: 1994-01-01 to 2007-12-31 (14 years)
    # Split: train (10 years) + val (2 years) + test (2 years)
    train_dates = ('1994-01-01', '2003-12-31')  # 10 years training data
    val_dates = ('2004-01-01', '2005-12-31')    # 2 years validation data
    test_dates = ('2006-01-01', '2007-12-31')   # 2 years test data

    # Parse command line arguments for stations to exclude
    # Usage: python preprocess_allbasin_aligntime_solar.py [station1] [station2] ...
    exclude_stations_list = None
    if len(sys.argv) > 1:
        exclude_stations_list = sys.argv[1:]
        print(f"Command line arguments: Excluding stations {exclude_stations_list} from normalization")
    else:
        # Optional: specify stations to exclude from normalization calculation
        # These stations will still be included in the dataset but won't affect y_mean/y_std
        # Example: exclude_stations_list = ['ACME', 'ADAX', 'ALTU']
        exclude_stations_list = None  # Set to None to use all stations, or provide a list of station IDs
        print("No command line arguments: Using all stations for normalization")

    # Generate time-aligned data with all-station normalization
    prep_data(
        csv_file='../../solar.csv',  # Path to solar_data.csv
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        seq_len=365,
        offset=1.0,
        out_file=os.path.join(data_dir, 'prepped.npz'),
        exclude_stations=exclude_stations_list
    )
