"""
TEM Data Preprocessing Script - Per-Basin Normalization with Time Alignment
Uses time-aligned batch organization AND per-basin normalization for Y
"""

import pandas as pd
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import os


def load_tem_data(csv_path):
    """
    Load TEM CSV data and convert to xarray dataset
    """
    df = pd.read_parquet(csv_path)

    # Convert basin_id to string for consistency
    df['basin_id'] = df['basin_id'].astype(str)

    # Convert Time to datetime
    df['Time'] = pd.to_datetime(df['Time'])

    # Get unique basins
    basins = df['basin_id'].unique()
    print(f"Found {len(basins)} basins")

    # Create xarray dataset with proper dimensions
    df_pivot = df.pivot(index='Time', columns='basin_id')

    # Get time and basin coordinates
    times = df_pivot.index.values
    basin_names = df_pivot['TEM'].columns.values

    # Create data arrays for each variable
    data_vars = {}

    # All variables in TEM dataset
    all_vars = [
        'lat', 'lon',
        'static_0', 'static_1', 'static_2', 'static_3', 'static_4', 'static_5', 'static_6',
        'co2', 'ch4',
        'daily_0', 'daily_1', 'daily_2', 'daily_3',
        'monthly_0',
        'TEM'
    ]

    for var in all_vars:
        if var in df_pivot.columns.levels[0]:
            data_vars[var] = xr.DataArray(
                df_pivot[var].values,
                dims=['date', 'seg_id_nat'],
                coords={'date': times, 'seg_id_nat': basin_names}
            )

    # Create dataset
    ds = xr.Dataset(data_vars)

    return ds


def add_temporal_features(ds):
    """Add temporal features (cyclical encoding of month and day of year)"""
    times = pd.to_datetime(ds.date.values)
    n_times = len(times)
    n_basins = len(ds.seg_id_nat)

    # Month cyclical encoding (1-12)
    month = times.month.values
    month_broadcasted = np.tile(month[:, np.newaxis], (1, n_basins))

    ds['month_sin'] = xr.DataArray(
        np.sin(2 * np.pi * month_broadcasted / 12),
        dims=['date', 'seg_id_nat'],
        coords={'date': ds.date, 'seg_id_nat': ds.seg_id_nat}
    )
    ds['month_cos'] = xr.DataArray(
        np.cos(2 * np.pi * month_broadcasted / 12),
        dims=['date', 'seg_id_nat'],
        coords={'date': ds.date, 'seg_id_nat': ds.seg_id_nat}
    )

    # Day of year cyclical encoding (1-365)
    day_of_year = times.dayofyear.values
    doy_broadcasted = np.tile(day_of_year[:, np.newaxis], (1, n_basins))

    ds['doy_sin'] = xr.DataArray(
        np.sin(2 * np.pi * doy_broadcasted / 365),
        dims=['date', 'seg_id_nat'],
        coords={'date': ds.date, 'seg_id_nat': ds.seg_id_nat}
    )
    ds['doy_cos'] = xr.DataArray(
        np.cos(2 * np.pi * doy_broadcasted / 365),
        dims=['date', 'seg_id_nat'],
        coords={'date': ds.date, 'seg_id_nat': ds.seg_id_nat}
    )

    return ds


def add_cumulative_features(ds, window_sizes=[7, 14, 30]):
    """Add cumulative features (rolling window statistics)"""
    for window in window_sizes:
        # Cumulative daily_0 (e.g., temperature)
        ds[f'daily0_sum_{window}d'] = ds['daily_0'].rolling(
            date=window, min_periods=1
        ).sum()

        # Average daily_1 (e.g., radiation)
        ds[f'daily1_avg_{window}d'] = ds['daily_1'].rolling(
            date=window, min_periods=1
        ).mean()

        # Average daily_2
        ds[f'daily2_avg_{window}d'] = ds['daily_2'].rolling(
            date=window, min_periods=1
        ).mean()

    return ds


def create_distance_matrix(ds, spatial_idx_name='seg_id_nat'):
    """Create distance matrix based on lat/lon"""
    basins = ds[spatial_idx_name].values
    n_basins = len(basins)

    # Extract coordinates
    coords = np.zeros((n_basins, 2))
    for i, basin in enumerate(basins):
        basin_data = ds.sel({spatial_idx_name: basin}).isel(date=0)
        coords[i, 0] = basin_data['lat'].values
        coords[i, 1] = basin_data['lon'].values

    # Calculate pairwise distances
    dist_matrix_raw = cdist(coords, coords, metric='euclidean')

    # Process distance matrix
    adj = -dist_matrix_raw
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
    """Time-aligned batch creation"""
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

    # Organize data by time windows
    sample_idx = 0
    for time_idx in range(n_time_windows):
        start_t = time_idx * step
        end_t = start_t + seq_len

        for basin_idx in range(n_basins):
            batched_data[sample_idx] = data_array[basin_idx, start_t:end_t, :]
            sample_idx += 1

    return batched_data


def create_ids_times_arrays(basin_names, dates, n_time_windows, n_segs, seq_len, offset):
    """Create basin IDs and timestamps arrays"""
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

        window_dates = dates[start_t:end_t]

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
    add_temporal=False,
    add_cumulative=True
):
    """
    Preprocess TEM data with per-basin Y normalization

    Parameters:
    -----------
    csv_file : str
        Path to TEM.csv
    train_dates : tuple
        (start_date, end_date) for training
    val_dates : tuple
        (start_date, end_date) for validation
    test_dates : tuple
        (start_date, end_date) for testing
    seq_len : int
        Sequence length (default: 365)
    offset : float
        Offset for batching (default: 1.0)
    out_file : str
        Output file path
    add_temporal : bool
        Whether to add temporal features (month_sin, month_cos, doy_sin, doy_cos)
    add_cumulative : bool
        Whether to add cumulative features
    """

    print("Loading TEM data...")
    ds = load_tem_data(csv_file)

    # Base X variables (16 features)
    x_vars = [
        'lat', 'lon',
        'static_0', 'static_1', 'static_2', 'static_3', 'static_4', 'static_5', 'static_6',
        'co2', 'ch4',
        'daily_0', 'daily_1', 'daily_2', 'daily_3',
        'monthly_0'
    ]

    # Add temporal features (cyclical encoding)
    if add_temporal:
        print("Adding temporal features...")
        ds = add_temporal_features(ds)
        x_vars.extend([
            'month_sin', 'month_cos', 'doy_sin', 'doy_cos'
        ])

    if add_cumulative:
        print("Adding cumulative features...")
        ds = add_cumulative_features(ds)
        x_vars.extend([
            'daily0_sum_7d', 'daily0_sum_14d', 'daily0_sum_30d',
            'daily1_avg_7d', 'daily1_avg_14d', 'daily1_avg_30d',
            'daily2_avg_7d', 'daily2_avg_14d', 'daily2_avg_30d'
        ])

    # Y variable
    y_var = 'TEM'

    basin_names = ds['seg_id_nat'].values
    n_segs = len(basin_names)

    # Split data
    print("Splitting data into train/val/test...")
    ds_train = ds.sel(date=slice(train_dates[0], train_dates[1]))
    ds_val = ds.sel(date=slice(val_dates[0], val_dates[1]))
    ds_test = ds.sel(date=slice(test_dates[0], test_dates[1]))

    # Prepare X data
    x_train = ds_train[x_vars]
    x_val = ds_val[x_vars]
    x_test = ds_test[x_vars]

    # Scale X data
    print("Scaling features...")
    x_train_flat = x_train.to_array().values.reshape(-1, len(x_vars))

    scaler_x = StandardScaler()
    scaler_x.fit(x_train_flat[~np.isnan(x_train_flat).any(axis=1)])

    x_mean = xr.Dataset({var: val for var, val in zip(x_vars, scaler_x.mean_)})
    x_std = xr.Dataset({var: val for var, val in zip(x_vars, scaler_x.scale_)})

    x_train_scaled = (x_train - x_mean) / (x_std + 1e-10)
    x_val_scaled = (x_val - x_mean) / (x_std + 1e-10)
    x_test_scaled = (x_test - x_mean) / (x_std + 1e-10)

    # Prepare Y data
    y_train = ds_train[[y_var]]
    y_val = ds_val[[y_var]]
    y_test = ds_test[[y_var]]

    # Calculate Y scaling per basin
    print("Calculating per-basin Y scaling parameters...")
    y_means = np.zeros(n_segs)
    y_stds = np.zeros(n_segs)

    for i, basin in enumerate(basin_names):
        basin_data = y_train[y_var].sel(seg_id_nat=basin).values
        valid_data = basin_data[~np.isnan(basin_data)]
        if len(valid_data) > 0:
            y_means[i] = np.mean(valid_data)
            y_stds[i] = np.std(valid_data)
            if y_stds[i] < 1e-8:
                y_stds[i] = 1.0
        else:
            y_means[i] = 0.0
            y_stds[i] = 1.0

    # Apply per-basin scaling
    print("Applying per-basin Y scaling...")
    y_train_scaled = y_train.copy(deep=True)
    y_val_scaled = y_val.copy(deep=True)
    y_test_scaled = y_test.copy(deep=True)

    for i, basin in enumerate(basin_names):
        y_train_scaled[y_var].loc[dict(seg_id_nat=basin)] = (
            (y_train[y_var].sel(seg_id_nat=basin) - y_means[i]) / (y_stds[i] + 1e-10)
        )
        y_val_scaled[y_var].loc[dict(seg_id_nat=basin)] = (
            (y_val[y_var].sel(seg_id_nat=basin) - y_means[i]) / (y_stds[i] + 1e-10)
        )
        y_test_scaled[y_var].loc[dict(seg_id_nat=basin)] = (
            (y_test[y_var].sel(seg_id_nat=basin) - y_means[i]) / (y_stds[i] + 1e-10)
        )

    # Convert to numpy arrays
    print("Converting to numpy arrays...")
    x_train_arr = x_train_scaled.to_array().values.transpose(2, 1, 0)  # [seg, time, feat]
    x_val_arr = x_val_scaled.to_array().values.transpose(2, 1, 0)
    x_test_arr = x_test_scaled.to_array().values.transpose(2, 1, 0)

    y_train_arr = y_train_scaled[y_var].values.T[:, :, np.newaxis]  # [seg, time, 1]
    y_val_arr = y_val_scaled[y_var].values.T[:, :, np.newaxis]
    y_test_arr = y_test_scaled[y_var].values.T[:, :, np.newaxis]

    # Raw Y values for evaluation
    print("Creating raw Y arrays for evaluation...")
    y_train_raw_arr = y_train[y_var].values.T[:, :, np.newaxis]
    y_val_raw_arr = y_val[y_var].values.T[:, :, np.newaxis]
    y_test_raw_arr = y_test[y_var].values.T[:, :, np.newaxis]

    print("Creating time-aligned batches...")
    x_trn_batched = convert_batch_reshape(x_train_arr, seq_len=seq_len, offset=offset)
    x_val_batched = convert_batch_reshape(x_val_arr, seq_len=seq_len, offset=1.0)
    x_tst_batched = convert_batch_reshape(x_test_arr, seq_len=seq_len, offset=1.0)

    y_trn_batched = convert_batch_reshape(y_train_arr, seq_len=seq_len, offset=offset)
    y_val_batched = convert_batch_reshape(y_val_arr, seq_len=seq_len, offset=1.0)
    y_tst_batched = convert_batch_reshape(y_test_arr, seq_len=seq_len, offset=1.0)

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

    # Create IDs and times arrays
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
        'x_trn': x_trn_batched,
        'y_obs_trn': y_trn_batched,
        'x_val': x_val_batched,
        'y_obs_val': y_val_batched,
        'x_tst': x_tst_batched,
        'y_obs_tst': y_tst_batched,
        'y_raw_trn': y_raw_trn_batched,
        'y_raw_val': y_raw_val_batched,
        'y_raw_tst': y_raw_tst_batched,
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
        'y_mean': y_means,
        'y_std': y_stds,
        'dist_mean': dist_mean,
        'dist_std': dist_std,
        'n_segs': n_segs,
        'basin_names': basin_names
    }

    # Save to file
    print(f"Saving preprocessed data to {out_file}...")
    np.savez_compressed(out_file, **data_dict)

    # Print summary
    print("\nPreprocessing complete!")
    print(f"Number of basins: {n_segs}")
    print(f"Training samples: {x_trn_batched.shape[0]} ({n_trn_windows} time windows x {n_segs} basins)")
    print(f"Validation samples: {x_val_batched.shape[0]} ({n_val_windows} time windows x {n_segs} basins)")
    print(f"Test samples: {x_tst_batched.shape[0]} ({n_tst_windows} time windows x {n_segs} basins)")
    print(f"Features: {len(x_vars)}")
    print(f"Feature list: {x_vars}")
    print(f"Sequence length: {seq_len}")
    print(f"\nNormalization: PER-BASIN")
    print(f"Y mean range: [{y_means.min():.4f}, {y_means.max():.4f}]")
    print(f"Y std range: [{y_stds.min():.4f}, {y_stds.max():.4f}]")

    return data_dict


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # TEM dataset time range: 2008-01-01 to 2018-12-31 (11 years)
    # For calibration task: train/val/test use same time range (not forecasting)
    train_dates = ('2008-01-01', '2018-12-31')  # 11 years
    val_dates = ('2008-01-01', '2018-12-31')    # 11 years (same as train)
    test_dates = ('2008-01-01', '2018-12-31')   # 11 years (same as train)

    prep_data(
        csv_file='../../TEM.parquet',
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        seq_len=365,
        offset=1.0,
        out_file=os.path.join(data_dir, 'prepped.npz'),
        add_temporal=False,      # Whether to add temporal features (month_sin, month_cos, doy_sin, doy_cos)
        add_cumulative=True      # Whether to add cumulative features
    )
