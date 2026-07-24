"""
Colorado Data Preprocessing Script - Official Batch Organization
Uses the official batch organization: all batches of one segment before moving to next segment
"""

import pandas as pd
import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import datetime
import os
import sys







def load_colorado_data(csv_path):
    """Load Colorado CSV data and convert to xarray dataset"""
    df = pd.read_csv(csv_path)
    
    # Convert Time to datetime
    df['Time'] = pd.to_datetime(df['Time'])
    
    # Get unique basins
    basins = df['basin'].unique()
    print(f"Found {len(basins)} basins: {basins}")
    
    # Create xarray dataset with proper dimensions
    # Pivot the dataframe to have basins as one dimension and time as another
    df_pivot = df.pivot(index='Time', columns='basin')
    
    # Get time and basin coordinates
    times = df_pivot.index.values
    # Use the basin order from pivot to ensure consistency
    basin_names = df_pivot['inflow'].columns.values
    
    # Create data arrays for each variable
    data_vars = {}
    
    # Dynamic features (time-varying)
    dynamic_vars = ['inflow', 'precipitation', 'temperature', 'daylight_duration_s', 
                   'solar_radiation_W_m2', 'snow_water_equivalent_kg_m2', 
                   'temp_max_C', 'temp_min_C', 'vapor_pressure_Pa']
    
    for var in dynamic_vars:
        data_vars[var] = xr.DataArray(
            df_pivot[var].values,
            dims=['date', 'seg_id_nat'],
            coords={'date': times, 'seg_id_nat': basin_names}
        )
    
    # Static features (constant for each basin)
    static_vars = ['latitude', 'longitude', 'elevation']
    for var in static_vars:
        # Get the first value for each basin (since they're constant)
        static_values = df.groupby('basin')[var].first().reindex(basin_names).values
        # Broadcast to all times
        broadcasted = np.tile(static_values[np.newaxis, :], (len(times), 1))
        data_vars[var] = xr.DataArray(
            broadcasted,
            dims=['date', 'seg_id_nat'],
            coords={'date': times, 'seg_id_nat': basin_names}
        )
    
    # Create dataset
    ds = xr.Dataset(data_vars)
    
    return ds


def create_distance_matrix(ds, spatial_idx_name='seg_id_nat'):
    """Create distance matrix based on spatial coordinates"""
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




def convert_batch_reshape(data_array, seq_len=365, offset=1.0, fill_batch=False, fill_nan=True, fill_time=False):
    """
    Modified version: Directly create time-aligned batches
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
    
    # Organize data directly by time window
    sample_idx = 0
    for time_idx in range(n_time_windows):
        start_t = time_idx * step
        end_t = start_t + seq_len
        
        # For this time window, add data from all basins
        for basin_idx in range(n_basins):
            batched_data[sample_idx] = data_array[basin_idx, start_t:end_t, :]
            sample_idx += 1
    
    return batched_data


def create_ids_times_arrays(basin_names, dates, n_time_windows, n_segs, seq_len, offset):
    """
    Create IDs and times arrays simultaneously, ensuring alignment
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
    out_file='prepped.npz'
):
    """Preprocessing function with direct time-aligned data generation"""
    
    print("Loading Colorado data...")
    ds = load_colorado_data(csv_file)
    
    x_vars = ['precipitation','temperature']
    y_var = 'inflow'  # Target variable

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

    # Save raw Y values (unnormalized, for ground truth evaluation)
    print("Creating raw Y arrays for evaluation...")
    y_train_raw_arr = y_train[y_var].values.T[:, :, np.newaxis]  # [seg, time, 1]
    y_val_raw_arr = y_val[y_var].values.T[:, :, np.newaxis]
    y_test_raw_arr = y_test[y_var].values.T[:, :, np.newaxis]

    # Directly create time-aligned batches
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
    step_val = int(seq_len * 1.0)  # Validation uses offset=1.0, so step=365
    step_tst = int(seq_len * 1.0)  # Test uses offset=1.0, so step=365
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
    print(f"Training samples: {x_trn_batched.shape[0]} ({n_trn_windows} time windows × {n_segs} basins)")
    print(f"Validation samples: {x_val_batched.shape[0]} ({n_val_windows} time windows × {n_segs} basins)")
    print(f"Test samples: {x_tst_batched.shape[0]} ({n_tst_windows} time windows × {n_segs} basins)")
    print(f"Features: {len(x_vars)}")
    print(f"Sequence length: {seq_len}")
    
    # Verify data organization
    print("\nVerifying time-aligned organization:")
    print("Time window 0 (samples 0-{}):".format(n_segs-1))
    for i in range(min(n_segs, 5)):
        print(f"  Sample {i}: Basin={ids_trn[i, 0, 0]}, Start time={times_trn[i, 0, 0]}")
    
    if n_trn_windows > 1:
        print(f"\nTime window 1 (samples {n_segs}-{2*n_segs-1}):")
        for i in range(n_segs, min(n_segs + 5, 2*n_segs)):
            print(f"  Sample {i}: Basin={ids_trn[i, 0, 0]}, Start time={times_trn[i, 0, 0]}")
        
        # Verify all basins in same time window have same start time
        print("\nVerifying time alignment in first window:")
        first_window_time = times_trn[0, 0, 0]
        all_same = all(times_trn[i, 0, 0] == first_window_time for i in range(n_segs))
        print(f"  All basins in first window have same start time: {all_same}")
    
    return data_dict




if __name__ == "__main__":

    # Get current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    train_dates = ('1999-01-01', '2009-12-31')
    # val_dates = ('2010-01-01', '2010-12-31')
    # test_dates = ('2011-01-01', '2011-12-31')
    val_dates = ('1999-01-01', '2009-12-31')
    test_dates = ('1999-01-01', '2009-12-31')


    # Directly generate time-aligned data
    prep_data(
        csv_file='../../datasets/colorado/All_Reservoirs_Combined.csv',
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates,
        seq_len=365,
        offset=1.0,
        out_file=os.path.join(data_dir, 'prepped.npz')
    )