"""
Solar Energy Data Preprocessing Script - Per-Station Normalization with Time Alignment
Uses time-aligned batch organization AND per-basin normalization for Y
Adapted from CAMELS perseg preprocessing script for AMS Solar Energy dataset
"""

import pandas as pd
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.spatial.distance import cdist
import datetime
import os
import sys


def load_solar_data(csv_path):
    """
    Load Solar CSV data and convert to xarray dataset
    """
    df = pd.read_parquet(csv_path)

    # Convert Time to datetime
    df['Time'] = pd.to_datetime(df['Time'])

    # Get unique basins
    basins = df['basin_id'].unique()
    print(f"Found {len(basins)} basins")

    # Create xarray dataset with proper dimensions
    df_pivot = df.pivot(index='Time', columns='basin_id')

    # Get time and basin coordinates
    times = df_pivot.index.values
    basin_names = df_pivot['solar_energy_MJ'].columns.values

    # Create data arrays for each variable
    data_vars = {}

    # Dynamic features (time-varying) - 45 meteorological features
    dynamic_vars = [
        # Precipitation
        'apcp_sfc_max', 'apcp_sfc_mean', 'apcp_sfc_min',
        # Downward longwave radiation
        'dlwrf_sfc_max', 'dlwrf_sfc_mean', 'dlwrf_sfc_min',
        # Downward shortwave radiation
        'dswrf_sfc_max', 'dswrf_sfc_mean', 'dswrf_sfc_min',
        # Sea-level pressure
        'pres_msl_max', 'pres_msl_mean', 'pres_msl_min',
        # Atmospheric precipitable water
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
                coords={'date': times, 'seg_id_nat': basin_names}
            )

    # Target variable
    data_vars['solar_energy_MJ'] = xr.DataArray(
        df_pivot['solar_energy_MJ'].values,
        dims=['date', 'seg_id_nat'],
        coords={'date': times, 'seg_id_nat': basin_names}
    )

    # Static features (constant per basin over time)
    static_vars = ['lat', 'lon', 'elev']
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
    n_times = len(times)
    n_basins = len(ds.seg_id_nat)

    # Add sine/cosine encoding for month
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

    # Add day-of-year encoding
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
    """Add cumulative features (sliding window statistics) - for solar energy prediction"""
    for window in window_sizes:
        # Cumulative shortwave radiation (most correlated with solar energy)
        ds[f'dswrf_sum_{window}d'] = ds['dswrf_sfc_mean'].rolling(
            date=window, min_periods=1
        ).sum()

        # Mean temperature
        ds[f'tmp_avg_{window}d'] = ds['tmp_2m_mean'].rolling(
            date=window, min_periods=1
        ).mean()

        # Mean cloud cover (affects solar energy)
        ds[f'tcdc_avg_{window}d'] = ds['tcdc_eatm_mean'].rolling(
            date=window, min_periods=1
        ).mean()

    return ds


def create_distance_matrix(ds, spatial_idx_name='seg_id_nat'):
    """Create distance matrix based on lat/lon coordinates"""
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
    """Generate time-aligned batches"""
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

    # Organize data by time window
    sample_idx = 0
    for time_idx in range(n_time_windows):
        start_t = time_idx * step
        end_t = start_t + seq_len

        for basin_idx in range(n_basins):
            batched_data[sample_idx] = data_array[basin_idx, start_t:end_t, :]
            sample_idx += 1

    return batched_data


def create_ids_times_arrays(basin_names, dates, n_time_windows, n_segs, seq_len, offset):
    """Create basin IDs and timestamp arrays"""
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
    Preprocess Solar data - using per-basin normalization

    Parameters:
    -----------
    csv_file : str
        Path to solar.csv
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
        Whether to add temporal features (month_sin, month_cos, etc.)
    add_cumulative : bool
        Whether to add cumulative features (rolling windows)
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

    # Apply feature engineering
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
            'dswrf_sum_7d', 'dswrf_sum_14d', 'dswrf_sum_30d',
            'tmp_avg_7d', 'tmp_avg_14d', 'tmp_avg_30d',
            'tcdc_avg_7d', 'tcdc_avg_14d', 'tcdc_avg_30d'
        ])

    # Target variable
    y_var = 'solar_energy_MJ'

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

    # Calculate Y scaling per basin (PER-STATION NORMALIZATION)
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

    # Save raw Y values (unnormalized, for ground truth evaluation)
    print("Creating raw Y arrays for evaluation...")
    y_train_raw_arr = y_train[y_var].values.T[:, :, np.newaxis]  # [seg, time, 1]
    y_val_raw_arr = y_val[y_var].values.T[:, :, np.newaxis]
    y_test_raw_arr = y_test[y_var].values.T[:, :, np.newaxis]

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
    print(f"Training samples: {x_trn_batched.shape[0]}")
    print(f"Validation samples: {x_val_batched.shape[0]}")
    print(f"Test samples: {x_tst_batched.shape[0]}")
    print(f"Features: {len(x_vars)}")
    print(f"Sequence length: {seq_len}")
    print(f"\nNormalization: PER-STATION")
    print(f"Y mean range: {y_means.min():.4f} ~ {y_means.max():.4f}")
    print(f"Y std range: {y_stds.min():.4f} ~ {y_stds.max():.4f}")

    return data_dict


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Solar dataset time range: 1994-01-01 to 2007-12-31 (14 years)
    # Split: training (10 years) + validation (2 years) + test (2 years)
    train_dates = ('1994-01-01', '2004-12-31')  # 11 years of training data
    val_dates = ('2005-01-01', '2006-12-31')    # 2 years of validation data
    test_dates = ('1994-01-01', '2004-12-31')   # 2 years of test data

    # Run preprocessing
    prep_data(
        csv_file='../../Solar.parquet',
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        seq_len=365,
        offset=1.0,
        out_file=os.path.join(data_dir, 'prepped.npz'),
        add_temporal=False,       # Whether to add temporal features
        add_cumulative=True       # Whether to add cumulative features
    )
