import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import argparse
import csv
from datetime import datetime
from scipy import stats
from sklearn.metrics import r2_score

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='Model')
    parser.add_argument('--partition', type=str, default='tst')
    parser.add_argument('--target_basin', type=str, default=None, help='Target basin to log metrics for')
    parser.add_argument('--metrics_log', type=str, default=None, help='Path to metrics log file')
    parser.add_argument('--use_vae_stats', action='store_true',
                        help='Use y_mean_vae/y_std_vae for denormalization (for RAW pipeline LSTM evaluation)')
    args = parser.parse_args()
    partition = args.partition
    # If metrics_log is specified, add partition to filename
    if args.metrics_log:
        args.metrics_log = args.metrics_log.replace('.csv', f'_{partition}.csv')
    # Get data file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, 'data', 'prepped.npz')
    data = np.load(data_file, allow_pickle=True)
    pred_file = os.path.join(args.pred_dir, f'{args.partition}.npy')
    y_pred = np.load(pred_file)
    output_dir = os.path.dirname(args.pred_dir)    
    # figure_dir = os.path.join(output_dir, 'figure')
    # os.makedirs(figure_dir, exist_ok=True)
    # Get basic info
    # Prefer y_raw_* as ground truth (original values, no denormalization needed)
    # Fall back to y_obs_* with denormalization if not found
    raw_key = f'y_raw_{partition}'
    obs_key = f'y_obs_{partition}'
    use_raw_observation = raw_key in data.files

    if use_raw_observation:
        print(f"Using {raw_key} as ground truth (original values, no denormalization needed)")
        observation_raw = data[raw_key]  # original values, for evaluation
    else:
        print(f"Warning: {raw_key} not found, falling back to {obs_key} with denormalization")

    observation = data[obs_key]  # normalized values (for data organization)
    basin_names = np.unique(data[f'ids_{partition}'])
    lenseg = len(basin_names)

    ###
    if args.use_vae_stats:
        y_std = data["y_std_vae"]
        y_mean = data["y_mean_vae"]
        print("args.use_vae_stats: Use y_mean_vae/y_std_vae for denormalization")
    else:
        y_std = data["y_std"]
        y_mean = data["y_mean"]

    sequence_length = data[f"x_{partition}"].shape[1]


    n_total_samples = observation.shape[0]
    n_time_windows = n_total_samples // lenseg
    print(f"Total samples: {n_total_samples}")
    print(f"Number of basins: {lenseg}")
    print(f"Number of time windows: {n_time_windows}")
    print(f"Sequence length: {sequence_length}")


    first_window_ids = data[f'ids_{partition}'][:lenseg, 0, 0]

    if len(np.unique(first_window_ids)) == lenseg:

        # Reorganize data from time-major to basin-major for visualization
        observation_by_basin = []
        observation_raw_by_basin = []
        y_pred_by_basin = []

        for basin_idx in range(lenseg):
            # Collect all time windows for this basin
            basin_obs = []
            basin_obs_raw = []
            basin_pred = []

            for time_idx in range(n_time_windows):
                sample_idx = time_idx * lenseg + basin_idx
                basin_obs.append(observation[sample_idx])
                basin_pred.append(y_pred[sample_idx])
                if use_raw_observation:
                    basin_obs_raw.append(observation_raw[sample_idx])

            observation_by_basin.append(np.concatenate(basin_obs, axis=0))
            y_pred_by_basin.append(np.concatenate(basin_pred, axis=0))
            if use_raw_observation:
                observation_raw_by_basin.append(np.concatenate(basin_obs_raw, axis=0))

        # Convert to arrays
        y_observation_concat = np.array(observation_by_basin)[:, :, np.newaxis]
        y_predict_concat = np.array(y_pred_by_basin)[:, :, np.newaxis]
        if use_raw_observation:
            y_observation_raw_concat = np.array(observation_raw_by_basin)[:, :, np.newaxis]

    else:
        print("WARNING: Data might still be basin-major organized!")
        # If data is still basin-major, use the original method
        a = observation.shape[0] // lenseg
        observation = np.reshape(observation, [lenseg, a*sequence_length, 1])
        y_observation_concat = observation
        y_predict = np.reshape(y_pred, [lenseg, a*sequence_length, 1])
        y_predict_concat = y_predict
        if use_raw_observation:
            observation_raw = np.reshape(observation_raw, [lenseg, a*sequence_length, 1])
            y_observation_raw_concat = observation_raw

    # Denormalization
    # - If y_raw_* exists: use raw values directly for observations, only denormalize predictions
    # - If y_raw_* does not exist: denormalize both observations and predictions
    if use_raw_observation:
        print("Using raw observation as ground truth (no denormalization for observation)")
        # Use raw values directly for observations
        y_observation_concat = y_observation_raw_concat
        # Only denormalize predictions
        if len(y_std.shape) == 0 or y_std.shape[0] == 1:
            print("Denormalizing predictions with global standardization")
            y_predict_concat = y_predict_concat * y_std + y_mean
        else:
            print("Denormalizing predictions with per-basin standardization")
            for i in range(lenseg):
                y_predict_concat[i,:,:] = y_predict_concat[i,:,:] * y_std[i] + y_mean[i]
    else:
        # Fallback mode: denormalize both observations and predictions
        if len(y_std.shape) == 0 or y_std.shape[0] == 1:
            print("Using global standardization for denormalization (both obs and pred)")
            y_observation_concat = y_observation_concat * y_std + y_mean
            y_predict_concat = y_predict_concat * y_std + y_mean
        else:
            print("Using per-basin standardization for denormalization (both obs and pred)")
            for i in range(lenseg):
                y_observation_concat[i,:,:] = y_observation_concat[i,:,:] * y_std[i] + y_mean[i]
                y_predict_concat[i,:,:] = y_predict_concat[i,:,:] * y_std[i] + y_mean[i]

    # Clip negative values to 0 (flow cannot be negative)
    y_predict_concat = np.maximum(y_predict_concat, 0)





    ##RMSE
    ##overall
    y_prd = np.reshape(y_predict_concat,[-1])
    y_obs = np.reshape(y_observation_concat, [-1])
    mask = np.isnan(y_obs)
    rmse = np.sqrt(np.mean(np.square(y_prd[~mask]-y_obs[~mask])))
    print(f"Overall RMSE: {rmse:.4f}")

    #each river
    rmse_each = np.zeros(lenseg)
    for i in range(lenseg):
        y1 = y_predict_concat[i,:,0]
        y2 = y_observation_concat[i,:,0]
        mask = np.isnan(y2)
        rmse_each[i] = np.sqrt(np.mean(np.square(y1[~mask]-y2[~mask])))

    rmse_each_river = pd.DataFrame({
    'basin_names': basin_names.flatten(),
    'rmse_each': rmse_each.flatten()
    })

    ##N-RMSE
    # Flatten the predicted and observed values into one-dimensional arrays
    y_prd = np.reshape(y_predict_concat, [-1])
    y_obs = np.reshape(y_observation_concat, [-1])
    # Filter out NaN values from the observed values
    mask = np.isnan(y_obs)
    # Calculate the overall RMSE
    rmse = np.sqrt(np.mean(np.square(y_prd[~mask] - y_obs[~mask])))
    # Calculate the range-based Normalized RMSE (N_RMSE)
    data_range = np.nanmax(y_obs) - np.nanmin(y_obs)  # Range of observed values
    n_rmse = rmse / data_range
    print(f"N_RMSE (Range-based): {n_rmse:.6f}")

    ##MAE
    # Flatten predicted and observed values into 1D arrays
    y_prd = np.reshape(y_predict_concat, [-1])
    y_obs = np.reshape(y_observation_concat, [-1])
    # Filter out NaN values from observations
    mask = np.isnan(y_obs)
    # Calculate MAE
    mae = np.mean(np.abs(y_prd[~mask] - y_obs[~mask]))
    print(f"MAE: {mae:.4f}")

    ##N-MAE
    # Flatten predicted and observed values into 1D arrays
    y_prd = np.reshape(y_predict_concat, [-1])
    y_obs = np.reshape(y_observation_concat, [-1])
    # Filter out NaN values from observations
    mask = np.isnan(y_obs)
    # Calculate MAE
    mae = np.mean(np.abs(y_prd[~mask] - y_obs[~mask]))
    # Calculate range-based N-MAE
    data_range = np.nanmax(y_obs) - np.nanmin(y_obs)  # range of observed values
    n_mae = mae / data_range
    print(f"N-MAE (Range-based): {n_mae:.6f}")

    ######
    ## NSE (Nash-Sutcliffe Efficiency)
    # Flatten predicted and observed values into 1D arrays
    y_prd = np.reshape(y_predict_concat, [-1])
    y_obs = np.reshape(y_observation_concat, [-1])
    # Filter out NaN values from observations
    mask = np.isnan(y_obs)
    # Calculate NSE
    obs_mean = np.mean(y_obs[~mask])
    numerator = np.sum(np.square(y_obs[~mask] - y_prd[~mask]))
    denominator = np.sum(np.square(y_obs[~mask] - obs_mean))
    nse = 1 - (numerator / denominator)
    print(f"NSE: {nse:.6f}")

    ## R² (Coefficient of Determination)
    # Using the same data
    r2 = stats.pearsonr(y_obs[~mask], y_prd[~mask])[0]**2
    print(f"R²: {r2:.6f}")

    # Or using sklearn
    r2_sklearn = r2_score(y_obs[~mask], y_prd[~mask])

    ## All metrics for each basin: NSE, R², MAE, KGE, PBIAS
    nse_each = np.zeros(lenseg)
    r2_each = np.zeros(lenseg)
    mae_each = np.zeros(lenseg)
    kge_each = np.zeros(lenseg)
    pbias_each = np.zeros(lenseg)

    for i in range(lenseg):
        y1 = y_predict_concat[i,:,0]  # predicted
        y2 = y_observation_concat[i,:,0]  # observed
        mask = np.isnan(y2)

        y_pred_valid = y1[~mask]
        y_obs_valid = y2[~mask]

        # Skip basins with insufficient valid observations (pearsonr/NSE need >= 2)
        if len(y_obs_valid) < 2:
            nse_each[i] = np.nan
            r2_each[i] = np.nan
            mae_each[i] = np.nan
            kge_each[i] = np.nan
            pbias_each[i] = np.nan
            continue

        # NSE for each basin
        obs_mean = np.mean(y_obs_valid)
        numerator = np.sum(np.square(y_obs_valid - y_pred_valid))
        denominator = np.sum(np.square(y_obs_valid - obs_mean))
        nse_each[i] = 1 - (numerator / denominator) if denominator > 0 else np.nan

        # R² for each basin
        r2_each[i] = stats.pearsonr(y_obs_valid, y_pred_valid)[0]**2

        # MAE for each basin
        mae_each[i] = np.mean(np.abs(y_obs_valid - y_pred_valid))

        r = stats.pearsonr(y_obs_valid, y_pred_valid)[0]
        alpha = np.std(y_pred_valid) / (np.std(y_obs_valid) + 1e-10)
        beta = np.mean(y_pred_valid) / (np.mean(y_obs_valid) + 1e-10)
        kge_each[i] = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

        # PBIAS for each basin (Percent Bias)
        # PBIAS = sum(obs - sim) / sum(obs) * 100
        pbias_each[i] = np.sum(y_obs_valid - y_pred_valid) / (np.sum(y_obs_valid) + 1e-10) * 100

    # Create DataFrame with all metrics
    metrics_each_river = pd.DataFrame({
        'basin_names': basin_names.flatten(),
        'rmse_each': rmse_each.flatten(),
        'mae_each': mae_each.flatten(),
        'r2_each': r2_each.flatten(),
        'kge_each': kge_each.flatten(),
        'pbias_each': pbias_each.flatten(),
        'nse_each': nse_each.flatten()
    })
    print("\nPer-basin metrics:")
    print(metrics_each_river)

    # If target_basin specified, write its metrics to log file
    if args.target_basin and args.metrics_log:
        try:
            # Parse target_basin parameter (may be single or multiple basins)
            target_basins = args.target_basin.split()

            fieldnames = ['timestamp', 'basin', 'nse', 'kge', 'rmse', 'mae', 'r2', 'pbias']

            # No-mask sentinel: 'DUMMY_BASIN' means log ALL basins (single-run, no spatial CV).
            # In K-fold mode target_basins are real IDs and skip this branch.
            if target_basins == ['DUMMY_BASIN']:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                all_basins = basin_names.flatten()
                df = pd.DataFrame({
                    'timestamp': timestamp,
                    'basin': all_basins,
                    'nse':   [f'{v:.6f}' for v in nse_each.flatten()],
                    'kge':   [f'{v:.6f}' for v in kge_each.flatten()],
                    'rmse':  [f'{v:.4f}' for v in rmse_each.flatten()],
                    'mae':   [f'{v:.4f}' for v in mae_each.flatten()],
                    'r2':    [f'{v:.6f}' for v in r2_each.flatten()],
                    'pbias': [f'{v:.2f}' for v in pbias_each.flatten()],
                })[fieldnames]  # enforce column order
                df.to_csv(args.metrics_log, index=False)
                print(f"\n[no-mask] Logged metrics for all {len(df)} basins -> {args.metrics_log}")
            else:
                file_exists = os.path.isfile(args.metrics_log)
                # Read existing data (if file exists)
                if file_exists:
                    df = pd.read_csv(args.metrics_log)
                else:
                    df = pd.DataFrame(columns=fieldnames)

                # Iterate over each target basin
                for target_basin in target_basins:
                    # Find the index of target basin
                    target_idx = None
                    for idx, basin in enumerate(basin_names.flatten()):
                        if basin == target_basin:
                            target_idx = idx
                            break

                    if target_idx is not None:
                        # Get metrics for this basin
                        basin_nse = nse_each[target_idx]
                        basin_kge = kge_each[target_idx]
                        basin_rmse = rmse_each[target_idx]
                        basin_mae = mae_each[target_idx]
                        basin_r2 = r2_each[target_idx]
                        basin_pbias = pbias_each[target_idx]

                        # Filter out old data for current basin
                        df = df[df['basin'] != target_basin]

                        # Add new data
                        new_row = pd.DataFrame([{
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'basin': target_basin,
                            'nse': f'{basin_nse:.6f}',
                            'kge': f'{basin_kge:.6f}',
                            'rmse': f'{basin_rmse:.4f}',
                            'mae': f'{basin_mae:.4f}',
                            'r2': f'{basin_r2:.6f}',
                            'pbias': f'{basin_pbias:.2f}'
                        }])
                        df = pd.concat([df, new_row], ignore_index=True)

                        print(f"\n✓ Logged metrics for basin {target_basin}:")
                        print(f"  NSE: {basin_nse:.6f}")
                        print(f"  KGE: {basin_kge:.6f}")
                        print(f"  RMSE: {basin_rmse:.4f}")
                        print(f"  MAE: {basin_mae:.4f}")
                        print(f"  R²: {basin_r2:.6f}")
                        print(f"  PBIAS: {basin_pbias:.2f}%")
                    else:
                        print(f"\n Warning: Basin {target_basin} not found in the data")

                # Write back to file
                df.to_csv(args.metrics_log, index=False)

        except Exception as e:
            print(f"\n Error logging metrics: {e}")






if __name__ == "__main__":
    main()