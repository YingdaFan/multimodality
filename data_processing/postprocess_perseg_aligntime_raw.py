import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
from datetime import datetime
from scipy import stats
from sklearn.metrics import r2_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='Model')
    parser.add_argument('--partition', type=str, default='tst')
    parser.add_argument('--target_basin', type=str, default=None, help='Target basin to log metrics for')
    parser.add_argument('--metrics_log', type=str, default=None, help='Path to metrics log file')
    args = parser.parse_args()

    partition = args.partition
    if args.metrics_log:
        args.metrics_log = args.metrics_log.replace('.csv', f'_{partition}.csv')

    # Load data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, 'data', 'prepped.npz')
    data = np.load(data_file, allow_pickle=True)

    pred_file = os.path.join(args.pred_dir, f'{args.partition}.npy')
    y_pred = np.load(pred_file)  # Already in ORIGINAL scale for RAW pipeline

    output_dir = os.path.dirname(args.pred_dir)
    figure_dir = os.path.join(output_dir, 'figure')
    os.makedirs(figure_dir, exist_ok=True)



    # Get ground truth from y_raw_*
    raw_key = f'y_raw_{partition}'
    if raw_key not in data.files:
        print(f"ERROR: {raw_key} not found in npz. RAW pipeline requires y_raw_* arrays.")
        print("Please run preprocess_perseg_aligntime_camels.py first.")
        return

    observation_raw = data[raw_key]
    basin_names = np.unique(data[f'ids_{partition}'])
    lenseg = len(basin_names)
    sequence_length = data[f"x_{partition}"].shape[1]

    n_total_samples = observation_raw.shape[0]
    n_time_windows = n_total_samples // lenseg

    print(f"Total samples: {n_total_samples}")
    print(f"Number of basins: {lenseg}")
    print(f"Number of time windows: {n_time_windows}")
    print(f"Prediction value range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")


    # Verify data organization
    first_window_ids = data[f'ids_{partition}'][:lenseg, 0, 0]
    if len(np.unique(first_window_ids)) != lenseg:
        print("WARNING: Data might not be time-major organized!")
        return

    print("Data is time-major organized (correct!)")

    # Reorganize data from time-major to basin-major
    observation_by_basin = []
    y_pred_by_basin = []

    for basin_idx in range(lenseg):
        basin_obs = []
        basin_pred = []

        for time_idx in range(n_time_windows):
            sample_idx = time_idx * lenseg + basin_idx
            basin_obs.append(observation_raw[sample_idx])
            basin_pred.append(y_pred[sample_idx])

        observation_by_basin.append(np.concatenate(basin_obs, axis=0))
        y_pred_by_basin.append(np.concatenate(basin_pred, axis=0))

    y_observation_concat = np.array(observation_by_basin)[:, :, np.newaxis]
    y_predict_concat = np.array(y_pred_by_basin)[:, :, np.newaxis]

    # No denormalization needed - both are in original scale!
    print("Using y_raw_* as ground truth (original scale)")
    print("Predictions already in original scale (RAW pipeline)")

    # Clip negative values to 0 (flow cannot be negative)
    y_predict_concat = np.maximum(y_predict_concat, 0)

    # Calculate metrics
    # Overall RMSE
    y_prd = np.reshape(y_predict_concat, [-1])
    y_obs = np.reshape(y_observation_concat, [-1])
    mask = np.isnan(y_obs)
    rmse = np.sqrt(np.mean(np.square(y_prd[~mask] - y_obs[~mask])))
    print(f"\nOverall RMSE: {rmse:.4f}")

    # Per-basin metrics
    rmse_each = np.zeros(lenseg)
    nse_each = np.zeros(lenseg)
    r2_each = np.zeros(lenseg)
    mae_each = np.zeros(lenseg)
    kge_each = np.zeros(lenseg)
    pbias_each = np.zeros(lenseg)

    for i in range(lenseg):
        y1 = y_predict_concat[i, :, 0]
        y2 = y_observation_concat[i, :, 0]
        mask = np.isnan(y2)

        y_pred_valid = y1[~mask]
        y_obs_valid = y2[~mask]

        # Skip basins with insufficient valid observations (pearsonr/NSE need >= 2)
        if len(y_obs_valid) < 2:
            rmse_each[i] = np.nan
            nse_each[i] = np.nan
            r2_each[i] = np.nan
            mae_each[i] = np.nan
            kge_each[i] = np.nan
            pbias_each[i] = np.nan
            continue

        # RMSE
        rmse_each[i] = np.sqrt(np.mean(np.square(y_pred_valid - y_obs_valid)))

        # NSE
        obs_mean = np.mean(y_obs_valid)
        numerator = np.sum(np.square(y_obs_valid - y_pred_valid))
        denominator = np.sum(np.square(y_obs_valid - obs_mean))
        nse_each[i] = 1 - (numerator / denominator) if denominator > 0 else np.nan

        # R²
        r2_each[i] = stats.pearsonr(y_obs_valid, y_pred_valid)[0] ** 2

        # MAE
        mae_each[i] = np.mean(np.abs(y_obs_valid - y_pred_valid))

        # KGE
        r = stats.pearsonr(y_obs_valid, y_pred_valid)[0]
        alpha = np.std(y_pred_valid) / (np.std(y_obs_valid) + 1e-10)
        beta = np.mean(y_pred_valid) / (np.mean(y_obs_valid) + 1e-10)
        kge_each[i] = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        # PBIAS (standard definition: positive = overestimate, negative = underestimate)
        pbias_each[i] = np.sum(y_pred_valid - y_obs_valid) / (np.sum(y_obs_valid) + 1e-10) * 100

    # Overall metrics (recreate mask after per-basin loop)
    mask = np.isnan(y_obs)
    data_range = np.nanmax(y_obs) - np.nanmin(y_obs)
    n_rmse = rmse / data_range
    mae = np.mean(np.abs(y_prd[~mask] - y_obs[~mask]))
    n_mae = mae / data_range
    obs_mean = np.mean(y_obs[~mask])
    nse = 1 - (np.sum(np.square(y_obs[~mask] - y_prd[~mask])) / np.sum(np.square(y_obs[~mask] - obs_mean)))
    r2 = stats.pearsonr(y_obs[~mask], y_prd[~mask])[0] ** 2

    print(f"N_RMSE (Range-based): {n_rmse:.6f}")
    print(f"MAE: {mae:.4f}")
    print(f"N-MAE (Range-based): {n_mae:.6f}")
    print(f"NSE: {nse:.6f}")
    print(f"R²: {r2:.6f}")

    # Print per-basin metrics
    metrics_df = pd.DataFrame({
        'basin_names': basin_names.flatten(),
        'rmse_each': rmse_each.flatten(),
        'mae_each': mae_each.flatten(),
        'r2_each': r2_each.flatten(),
        'kge_each': kge_each.flatten(),
        'pbias_each': pbias_each.flatten(),
        'nse_each': nse_each.flatten()
    })
    print("\nPer-basin metrics:")
    print(metrics_df)

    # Log target basin metrics
    if args.target_basin and args.metrics_log:
        try:
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

                if file_exists:
                    df = pd.read_csv(args.metrics_log)
                else:
                    df = pd.DataFrame(columns=fieldnames)

                for target_basin in target_basins:
                    target_idx = None
                    for idx, basin in enumerate(basin_names.flatten()):
                        if basin == target_basin:
                            target_idx = idx
                            break

                    if target_idx is not None:
                        basin_nse = nse_each[target_idx]
                        basin_kge = kge_each[target_idx]
                        basin_rmse = rmse_each[target_idx]
                        basin_mae = mae_each[target_idx]
                        basin_r2 = r2_each[target_idx]
                        basin_pbias = pbias_each[target_idx]

                        df = df[df['basin'] != target_basin]

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

                    else:
                        print(f"\n Warning: Basin {target_basin} not found in the data")

                df.to_csv(args.metrics_log, index=False)

        except Exception as e:
            print(f"\n Error logging metrics: {e}")

    # # Visualization (if target_basin specified)
    # if args.target_basin is not None:
    #     target_basins = args.target_basin.split()
    #     plot_indices = []
    #     plot_basin_names = []

    #     for target in target_basins:
    #         indices = np.where(basin_names == target)[0]
    #         if len(indices) > 0:
    #             plot_indices.append(indices[0])
    #             plot_basin_names.append(target)

    #     if len(plot_indices) > 0:
    #         n_plot = len(plot_indices)
    #         print(f"\nGenerating visualization for {n_plot} target basin(s)...")

    #         import seaborn as sns
    #         sns.set_style("white")

    #         if n_plot <= 4:
    #             cols, rows = n_plot, 1
    #             fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4))
    #         else:
    #             cols = min(5, n_plot)
    #             rows = (n_plot + cols - 1) // cols
    #             fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows))

    #         axes_flat = np.array(axes).flatten() if n_plot > 1 else [axes]

    #         for idx, i in enumerate(plot_indices):
    #             ax = axes_flat[idx]
    #             y_obs = y_observation_concat[i, :, 0]
    #             y_pred_i = y_predict_concat[i, :, 0]
    #             time_steps = np.arange(len(y_obs))

    #             ax.scatter(time_steps, y_obs, color='#dc2626', s=8, alpha=0.6,
    #                        label='Observed', edgecolors='none')
    #             ax.plot(time_steps, y_pred_i, color='#1e40af', linewidth=1.3,
    #                     label='Predicted', alpha=0.8, linestyle='--')

    #             nse_value = nse_each[i]
    #             title_color = '#059669' if nse_value > 0.75 else '#ea580c' if nse_value > 0.5 else '#dc2626'
    #             ax.set_title(f'{basin_names[i]}\n(NSE: {nse_value:.2f})',
    #                          fontsize=10, fontweight='bold', color=title_color)

    #             ax.grid(True, alpha=0.2, linestyle='--')
    #             ax.legend(loc='upper right', frameon=True, fontsize=8)

    #         for idx in range(n_plot, len(axes_flat)):
    #             axes_flat[idx].set_visible(False)

    #         fig.suptitle(f'RAW Pipeline - {partition.upper()} Set', fontsize=14, fontweight='bold')
    #         plt.tight_layout()

    #         save_path = f"{figure_dir}/basin_grid_raw_{partition}.png"
    #         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #         plt.close()
    #         print(f"Visualization saved to: {save_path}")

    # print("\nRAW Pipeline postprocess complete!")


if __name__ == "__main__":
    main()
