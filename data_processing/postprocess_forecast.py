"""
Forecasting Postprocess: 从滑动窗口预测中提取指定步，反归一化，计算指标。

滑动窗口（stride=1）产生大量重叠预测，需要从每个窗口中选取一个预测步拼成连续序列。
step_index 控制选哪一步：
    step_index=0  → 取每个窗口的第 1 个预测步（1-step-ahead forecast）
    step_index=1  → 取第 2 个预测步（2-step-ahead）
    step_index=-1 → 取最后一个预测步（pred_len-step-ahead）

Usage:
    python postprocess_forecast.py --pred_dir ../diffusion/output/pred --step_index 0
    python postprocess_forecast.py --pred_dir ../diffusion/output/pred --step_index -1
    python postprocess_forecast.py --pred_dir ../diffusion/output/pred --step_index 0 --window 168 --pred_len 72
    python postprocess_forecast.py --pred_dir ../diffusion/output/pred --decay 0 5 11 17 23
"""

import numpy as np
import pandas as pd
import os
import json
import argparse
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D


def compute_step_metrics(y_pred_all, y_raw, times, y_mean, y_std, basin_names,
                         step_index, window, pred_len, stride=1):
    """
    Extract a single step_index from sliding-window predictions, denormalize,
    and compute per-basin metrics.

    Returns
    -------
    results : list[dict]
        Per-basin metrics (basin, nse, kge, rmse, mae, r2, pbias).
    overall : dict
        Overall metrics (rmse, mae, nse).
    """
    n_basins = len(basin_names)
    n_times = y_raw.shape[1]
    max_offsets = n_times - window - pred_len + 1
    n_offsets = (max_offsets - 1) // stride + 1

    # Extract step_index from each window
    y_pred_step = y_pred_all[:, step_index, 0]
    y_pred_by_basin = y_pred_step.reshape(n_offsets, n_basins).T

    # Ground truth indices: window k predicts time (k*stride + window + step_index)
    gt_indices = np.array([k * stride + window + step_index for k in range(n_offsets)])
    y_gt_by_basin = y_raw[:, gt_indices, 0]

    # Denormalize
    y_pred_denorm = np.zeros_like(y_pred_by_basin)
    for i in range(n_basins):
        y_pred_denorm[i] = y_pred_by_basin[i] * y_std[i] + y_mean[i]
    y_pred_denorm = np.maximum(y_pred_denorm, 0)

    # Per-basin metrics
    results = []
    for i in range(n_basins):
        bid = str(basin_names[i])
        y_p = y_pred_denorm[i]
        y_g = y_gt_by_basin[i]
        mask = ~np.isnan(y_g) & ~np.isnan(y_p)
        y_p_v = y_p[mask]
        y_g_v = y_g[mask]

        if len(y_g_v) < 2:
            continue

        rmse = np.sqrt(np.mean((y_p_v - y_g_v)**2))
        mae = np.mean(np.abs(y_p_v - y_g_v))
        obs_mean = np.mean(y_g_v)
        nse = 1 - np.sum((y_g_v - y_p_v)**2) / (np.sum((y_g_v - obs_mean)**2) + 1e-10)
        r_val = stats.pearsonr(y_g_v, y_p_v)[0]
        r2 = r_val ** 2
        alpha = np.std(y_p_v) / (np.std(y_g_v) + 1e-10)
        beta = np.mean(y_p_v) / (np.mean(y_g_v) + 1e-10)
        kge = 1 - np.sqrt((r_val - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        pbias = np.sum(y_g_v - y_p_v) / (np.sum(y_g_v) + 1e-10) * 100

        results.append({
            'basin': bid, 'n_valid': len(y_g_v),
            'nse': nse, 'kge': kge, 'rmse': rmse, 'mae': mae,
            'r2': r2, 'pbias': pbias
        })

    # Overall metrics
    all_pred = np.concatenate([y_pred_denorm[i] for i in range(n_basins)])
    all_gt = np.concatenate([y_gt_by_basin[i] for i in range(n_basins)])
    mask = ~np.isnan(all_gt) & ~np.isnan(all_pred)
    y_p_v = all_pred[mask]
    y_g_v = all_gt[mask]
    obs_mean = np.mean(y_g_v)
    overall = {
        'rmse': np.sqrt(np.mean((y_p_v - y_g_v)**2)),
        'mae': np.mean(np.abs(y_p_v - y_g_v)),
        'nse': 1 - np.sum((y_g_v - y_p_v)**2) / (np.sum((y_g_v - obs_mean)**2) + 1e-10),
    }

    return results, overall


def plot_multi_horizon(y_pred_all, y_raw, times, y_mean, y_std, basin_names,
                       step_indices, window, pred_len, output_dir, partition,
                       stride=1):
    """
    Plot obs + multiple forecast-horizon pred curves on one figure per basin.

    For each basin, generates a single plot with:
    - Observed streamflow (ground truth)
    - Multiple prediction curves (e.g. pred next 1, next 6, next 12, ...)
    each corresponding to a different step_index (forecast horizon).
    """
    n_basins = len(basin_names)
    n_times = y_raw.shape[1]
    max_offsets = n_times - window - pred_len + 1
    n_offsets = (max_offsets - 1) // stride + 1

    # Extract & denormalize predictions for each step_index
    horizon_data = {}  # step_index -> {bid: (times, pred_denorm)}
    all_basin_results = []

    for si in step_indices:
        results, overall = compute_step_metrics(
            y_pred_all, y_raw, times, y_mean, y_std, basin_names,
            si, window, pred_len, stride=stride
        )
        all_basin_results.append((si, results))
        print(f"  step_index={si} ({si+1}h-ahead): "
              f"NSE={overall['nse']:.4f}  RMSE={overall['rmse']:.4f}")

        # Extract denormalized predictions per basin
        y_pred_step = y_pred_all[:, si, 0]
        y_pred_by_basin = y_pred_step.reshape(n_offsets, n_basins).T
        gt_indices = np.array([k * stride + window + si for k in range(n_offsets)])
        t_pred = times[gt_indices]

        per_basin = {}
        for i in range(n_basins):
            bid = str(basin_names[i])
            pred_denorm = y_pred_by_basin[i] * y_std[i] + y_mean[i]
            pred_denorm = np.maximum(pred_denorm, 0)
            per_basin[bid] = (t_pred, pred_denorm)
        horizon_data[si] = per_basin

    # Obs covers the full range: from window to the end of the last pred curve
    si_max = max(step_indices)
    obs_start = window
    obs_end = (n_offsets - 1) * stride + window + si_max + 1  # covers all pred curves
    obs_end = min(obs_end, n_times)        # don't exceed data length
    t_obs = times[obs_start:obs_end]
    y_obs_by_basin = {}
    for i in range(n_basins):
        bid = str(basin_names[i])
        y_obs_by_basin[bid] = y_raw[i, obs_start:obs_end, 0]

    # ---- Plot per basin ----
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'figure.dpi': 150,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.4,
        'ytick.major.width': 0.4,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
    })

    color_obs = '#1f77b4'
    pred_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    lw_obs = 0.6
    lw_pred = 0.5

    fig_dir = os.path.join(output_dir, 'figure')
    os.makedirs(fig_dir, exist_ok=True)

    for bid in sorted(y_obs_by_basin.keys()):
        t_dates = pd.to_datetime(t_obs)
        y_g = y_obs_by_basin[bid]

        fig, ax = plt.subplots(figsize=(6.75, 2.5))

        # Obs
        ax.plot(t_dates, y_g, color=color_obs,
                linewidth=lw_obs, alpha=0.9, zorder=len(step_indices) + 2,
                label='Observed')

        # Pred curves for each horizon
        for j, si in enumerate(step_indices):
            t_pred_raw, y_p = horizon_data[si][bid]
            t_pred_dates = pd.to_datetime(t_pred_raw)
            color = pred_colors[j % len(pred_colors)]
            ax.plot(t_pred_dates, y_p, color=color,
                    linewidth=lw_pred, alpha=0.8, zorder=j + 1,
                    label=f'pred next {si+1}')

        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.2, linewidth=0.3)
        ax.set_ylabel('Streamflow', fontsize=9)

        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

        ax.set_title(f'Basin {bid}  |  Multi-horizon forecast',
                     fontsize=9, fontweight='bold', pad=4)
        ax.legend(fontsize=5, loc='upper right', frameon=True, fancybox=False,
                  edgecolor='#cccccc', framealpha=0.9, borderpad=0.3,
                  handlelength=1.2, ncol=2)

        plt.tight_layout()
        fig_path = os.path.join(fig_dir, f'{partition}_{bid}_multi_horizon.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Multi-horizon plot saved: {fig_path}")

    # ---- Save decay metrics CSV ----
    rows = []
    for si, results in all_basin_results:
        for r in results:
            rows.append({'step_index': si, 'hours_ahead': si + 1, **r})
    df_decay = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, 'denorm', f'{partition}_decay_metrics.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_decay.to_csv(csv_path, index=False)
    print(f"Decay metrics CSV saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Postprocess forecasting predictions')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Directory containing prediction .npy files')
    parser.add_argument('--partition', type=str, default='tst',
                        help='Data partition (default: tst)')
    parser.add_argument('--step_index', type=int, default=0,
                        help='Which step to extract from each prediction window '
                             '(0=first, 1=second, ..., -1=last)')
    parser.add_argument('--window', type=int, default=168,
                        help='History window length (must match training)')
    parser.add_argument('--pred_len', type=int, default=18,
                        help='Prediction length (must match training)')
    parser.add_argument('--stride', type=int, default=24,
                        help='Sliding window stride (must match training)')
    parser.add_argument('--decay', type=int, nargs='+', default=[0, 5, 11, 17])
    parser.add_argument('--no_decay', action='store_true')
    parser.add_argument('--target_basin', type=str, default=None,
                        help='Target basin(s) to log metrics for (space-separated)')
    parser.add_argument('--metrics_log', type=str, default=None,
                        help='Path to metrics CSV log file')
    args = parser.parse_args()

    partition = args.partition
    step_index = args.step_index

    # Sidecar 优先：从 pred_dir/meta.json 读取训练实际用到的 stride/window/pred_len
    # 缺失时回落到 CLI flag（保留向后兼容，老的 tst.npy 仍可跑）
    meta_path = os.path.join(args.pred_dir, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as _f:
            _meta = json.load(_f)
        window = int(_meta['window'])
        pred_len = int(_meta['pred_len'])
        stride = int(_meta['stride'])
        print(f"=== Forecasting Postprocess ===")
        print(f"[meta] using sidecar {meta_path}: window={window}, pred_len={pred_len}, stride={stride}")
    else:
        window = args.window
        pred_len = args.pred_len
        stride = args.stride
        print(f"=== Forecasting Postprocess ===")
        print(f"[meta] meta.json not found at {meta_path}; falling back to CLI args")

    # Handle negative indexing
    if step_index < 0:
        step_index = pred_len + step_index
    assert 0 <= step_index < pred_len, \
        f"step_index={step_index} out of range [0, {pred_len})"

    print(f"step_index={step_index} ({step_index+1}-step-ahead forecast)")
    print(f"window={window}, pred_len={pred_len}, stride={stride}")

    # Load preprocessed data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, 'data', 'prepped.npz')
    data = np.load(data_file, allow_pickle=True)

    # Load predictions
    pred_file = os.path.join(args.pred_dir, f'{partition}.npy')
    y_pred_all = np.load(pred_file)  # (N_total, pred_len, 1)
    print(f"Predictions loaded: {pred_file}, shape: {y_pred_all.shape}")

    # Get metadata (new format: continuous arrays, 1-D shared time axis)
    y_raw = data[f'y_raw_{partition}']       # (n_basins, n_times, 1)
    times = data[f'times_{partition}']       # (n_times,) 1-D shared
    y_mean = data['y_mean']                  # (n_basins,)
    y_std = data['y_std']                    # (n_basins,)
    basin_names = data['basin_names']        # (n_basins,)
    n_basins = len(basin_names)

    n_times = y_raw.shape[1]
    max_offsets = n_times - window - pred_len + 1
    n_offsets = (max_offsets - 1) // stride + 1
    print(f"n_basins={n_basins}, n_times={n_times}, stride={stride}, n_offsets={n_offsets}")
    print(f"Expected total samples: {n_basins * n_offsets}, got: {y_pred_all.shape[0]}")
    assert y_pred_all.shape[0] == n_basins * n_offsets, \
        f"Sample count mismatch! Check window/pred_len/stride settings."

    # =========================================================
    # Decay analysis mode: compute metrics for multiple step indices
    # =========================================================
    if not args.no_decay:
        step_indices = [s if s >= 0 else pred_len + s for s in args.decay]
        step_indices = [s for s in step_indices if 0 <= s < pred_len]
        if step_indices:
            print(f"\n=== Multi-horizon Analysis: step_indices={step_indices} ===")
            output_dir = os.path.dirname(args.pred_dir)
            plot_multi_horizon(y_pred_all, y_raw, times, y_mean, y_std, basin_names,
                               step_indices, window, pred_len, output_dir, partition,
                               stride=stride)

    # =========================================================
    # Step 1: Extract step_index from each sliding window
    # =========================================================
    # Dataset ordering: basins cycle first, then time offsets
    # So predictions are: offset0_basin0, offset0_basin1, ..., offset1_basin0, ...
    y_pred_step = y_pred_all[:, step_index, 0]                              # (n_offsets * n_basins,)
    y_pred_by_basin = y_pred_step.reshape(n_offsets, n_basins).T            # (n_basins, n_offsets)

    # Ground truth indices: window k predicts time (k*stride + window + step_index)
    gt_indices = np.array([k * stride + window + step_index for k in range(n_offsets)])
    y_gt_by_basin = y_raw[:, gt_indices, 0]   # (n_basins, n_offsets)
    times_pred = times[gt_indices]             # (n_offsets,) shared

    # =========================================================
    # Step 2: Denormalize predictions (per-basin)
    # =========================================================
    print("Denormalizing with per-basin y_mean/y_std")
    y_pred_denorm = np.zeros_like(y_pred_by_basin)
    for i in range(n_basins):
        y_pred_denorm[i] = y_pred_by_basin[i] * y_std[i] + y_mean[i]
    y_pred_denorm = np.maximum(y_pred_denorm, 0)

    # =========================================================
    # Step 3: Build per-basin dicts
    # =========================================================
    pred_by_basin = {}
    gt_by_basin = {}
    times_by_basin = {}
    for i in range(n_basins):
        bid = str(basin_names[i])
        pred_by_basin[bid] = y_pred_denorm[i]
        gt_by_basin[bid] = y_gt_by_basin[i]
        times_by_basin[bid] = times_pred   # shared

    # =========================================================
    # Step 4: Compute per-basin metrics
    # =========================================================
    print(f"\n=== Per-basin metrics (step_index={step_index}, {step_index+1}-step-ahead) ===")
    results = []

    for bid in sorted(pred_by_basin.keys()):
        y_p = pred_by_basin[bid]
        y_g = gt_by_basin[bid]
        mask = ~np.isnan(y_g) & ~np.isnan(y_p)
        y_p_v = y_p[mask]
        y_g_v = y_g[mask]

        if len(y_g_v) < 2:
            print(f"  Basin {bid}: insufficient valid data, skipping")
            continue

        # RMSE
        rmse = np.sqrt(np.mean((y_p_v - y_g_v)**2))
        # MAE
        mae = np.mean(np.abs(y_p_v - y_g_v))
        # NSE
        obs_mean = np.mean(y_g_v)
        nse = 1 - np.sum((y_g_v - y_p_v)**2) / (np.sum((y_g_v - obs_mean)**2) + 1e-10)
        # R²
        r_val = stats.pearsonr(y_g_v, y_p_v)[0]
        r2 = r_val ** 2
        # KGE
        alpha = np.std(y_p_v) / (np.std(y_g_v) + 1e-10)
        beta = np.mean(y_p_v) / (np.mean(y_g_v) + 1e-10)
        kge = 1 - np.sqrt((r_val - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        # PBIAS
        pbias = np.sum(y_g_v - y_p_v) / (np.sum(y_g_v) + 1e-10) * 100

        results.append({
            'basin': bid, 'n_valid': len(y_g_v),
            'nse': nse, 'kge': kge, 'rmse': rmse, 'mae': mae,
            'r2': r2, 'pbias': pbias
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # =========================================================
    # Step 5: Overall metrics
    # =========================================================
    all_pred = np.concatenate([pred_by_basin[b] for b in sorted(pred_by_basin.keys())])
    all_gt = np.concatenate([gt_by_basin[b] for b in sorted(gt_by_basin.keys())])
    mask = ~np.isnan(all_gt) & ~np.isnan(all_pred)
    y_p_v = all_pred[mask]
    y_g_v = all_gt[mask]

    obs_mean = np.mean(y_g_v)
    rmse = np.sqrt(np.mean((y_p_v - y_g_v)**2))
    mae = np.mean(np.abs(y_p_v - y_g_v))
    nse = 1 - np.sum((y_g_v - y_p_v)**2) / (np.sum((y_g_v - obs_mean)**2) + 1e-10)

    print(f"\n=== Overall metrics ===")
    print(f"Total valid timesteps: {len(y_g_v)}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"NSE:  {nse:.6f}")

    # =========================================================
    # Step 6: Save denormalized predictions
    # =========================================================
    output_dir = os.path.dirname(args.pred_dir)
    save_dir = os.path.join(output_dir, 'denorm')
    os.makedirs(save_dir, exist_ok=True)

    for bid in sorted(pred_by_basin.keys()):
        df_out = pd.DataFrame({
            'time': times_by_basin[bid],
            'pred': pred_by_basin[bid],
            'obs': gt_by_basin[bid]
        })
        save_path = os.path.join(save_dir, f'{partition}_{bid}_step{step_index}.csv')
        df_out.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

    # =========================================================
    # Step 7: Plot pred vs obs for each basin
    # =========================================================
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'figure.dpi': 150,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.4,
        'ytick.major.width': 0.4,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
    })

    color_obs = '#1f77b4'
    color_pred = '#d62728'
    lw_plot = 0.5
    alpha_plot = 0.85

    fig_dir = os.path.join(output_dir, 'figure')
    os.makedirs(fig_dir, exist_ok=True)

    for bid in sorted(pred_by_basin.keys()):
        t_raw = times_by_basin[bid]
        t_dates = pd.to_datetime(t_raw)
        y_p = pred_by_basin[bid]
        y_g = gt_by_basin[bid]

        # Get metrics for title
        m = next((r for r in results if r['basin'] == bid), None)

        fig, ax = plt.subplots(figsize=(6.75, 2.0))

        ax.plot(t_dates, y_g, color=color_obs,
                linewidth=lw_plot, alpha=alpha_plot, zorder=3)
        ax.plot(t_dates, y_p, color=color_pred,
                linewidth=lw_plot, alpha=alpha_plot, zorder=2)

        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.2, linewidth=0.3)
        ax.set_ylabel('Streamflow', fontsize=9)

        # Date formatting
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

        # Title with metrics
        title = f'Basin {bid}  |  {step_index+1}-step-ahead'
        if m is not None:
            title += f'  |  NSE={m["nse"]:.3f}  KGE={m["kge"]:.3f}  RMSE={m["rmse"]:.2f}'
        ax.set_title(title, fontsize=9, fontweight='bold', pad=4)

        # Legend
        legend_elements = [
            Line2D([0], [0], color=color_obs, linewidth=1.5, alpha=0.9, label='Observed'),
            Line2D([0], [0], color=color_pred, linewidth=1.5, alpha=0.9, label='Prediction'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=6,
                  frameon=True, fancybox=False, edgecolor='#cccccc',
                  framealpha=0.9, borderpad=0.3, handlelength=1.2)

        plt.tight_layout()
        fig_path = os.path.join(fig_dir, f'{partition}_{bid}_step{step_index}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved: {fig_path}")

    # =========================================================
    # Step 8: f(x) diagnostic — compare Transformer-only vs Diffusion
    # =========================================================
    fx_file = os.path.join(args.pred_dir, f'{partition}_fx.npy')
    if os.path.isfile(fx_file):
        y_fx_all = np.load(fx_file)  # (N_total, pred_len, 1)
        print(f"\n=== f(x) Diagnostic: Transformer-only vs Diffusion ===")
        print(f"f(x) predictions loaded: {fx_file}, shape: {y_fx_all.shape}")

        # Extract step_index from f(x) predictions
        y_fx_step = y_fx_all[:, step_index, 0]
        y_fx_by_basin = y_fx_step.reshape(n_offsets, n_basins).T

        # Denormalize f(x) predictions
        y_fx_denorm = np.zeros_like(y_fx_by_basin)
        for i in range(n_basins):
            y_fx_denorm[i] = y_fx_by_basin[i] * y_std[i] + y_mean[i]
        y_fx_denorm = np.maximum(y_fx_denorm, 0)

        # Per-basin f(x) metrics
        fx_results = []
        for i in range(n_basins):
            bid = str(basin_names[i])
            y_p = y_fx_denorm[i]
            y_g = y_gt_by_basin[i]
            mask_v = ~np.isnan(y_g) & ~np.isnan(y_p)
            y_p_v = y_p[mask_v]
            y_g_v_b = y_g[mask_v]
            if len(y_g_v_b) < 2:
                continue
            rmse_b = np.sqrt(np.mean((y_p_v - y_g_v_b)**2))
            mae_b = np.mean(np.abs(y_p_v - y_g_v_b))
            obs_mean_b = np.mean(y_g_v_b)
            nse_b = 1 - np.sum((y_g_v_b - y_p_v)**2) / (np.sum((y_g_v_b - obs_mean_b)**2) + 1e-10)
            r_val_b = stats.pearsonr(y_g_v_b, y_p_v)[0]
            alpha_b = np.std(y_p_v) / (np.std(y_g_v_b) + 1e-10)
            beta_b = np.mean(y_p_v) / (np.mean(y_g_v_b) + 1e-10)
            kge_b = 1 - np.sqrt((r_val_b - 1)**2 + (alpha_b - 1)**2 + (beta_b - 1)**2)
            fx_results.append({'basin': bid, 'nse': nse_b, 'kge': kge_b, 'rmse': rmse_b, 'mae': mae_b})

        # Overall f(x) metrics
        all_fx = np.concatenate([y_fx_denorm[i] for i in range(n_basins)])
        mask_fx = ~np.isnan(all_gt) & ~np.isnan(all_fx)
        fx_p = all_fx[mask_fx]
        fx_g = all_gt[mask_fx]
        fx_obs_mean = np.mean(fx_g)
        fx_nse = 1 - np.sum((fx_g - fx_p)**2) / (np.sum((fx_g - fx_obs_mean)**2) + 1e-10)
        fx_rmse = np.sqrt(np.mean((fx_p - fx_g)**2))

        print(f"\n--- Comparison (step_index={step_index}) ---")
        print(f"{'':>12s}  {'NSE':>10s}  {'RMSE':>10s}")
        print(f"{'f(x) only':>12s}  {fx_nse:>10.4f}  {fx_rmse:>10.4f}")
        print(f"{'Diffusion':>12s}  {nse:>10.4f}  {rmse:>10.4f}")
        if fx_nse > nse:
            print("=> f(x) alone is BETTER — diffusion is hurting performance")
        else:
            print("=> Diffusion improves over f(x) — diffusion is helping")

        # Plot: obs vs f(x) vs diffusion for each basin
        color_fx = '#2ca02c'
        for bid in sorted(pred_by_basin.keys()):
            t_raw = times_by_basin[bid]
            t_dates = pd.to_datetime(t_raw)
            y_g_plot = gt_by_basin[bid]
            y_diff = pred_by_basin[bid]
            i_basin = list(sorted(pred_by_basin.keys())).index(bid)
            y_fx_plot = y_fx_denorm[list(basin_names.astype(str)).index(bid)]

            m_diff = next((r for r in results if r['basin'] == bid), None)
            m_fx = next((r for r in fx_results if r['basin'] == bid), None)

            fig, ax = plt.subplots(figsize=(6.75, 2.5))
            ax.plot(t_dates, y_g_plot, color=color_obs,
                    linewidth=lw_plot, alpha=alpha_plot, zorder=4)
            ax.plot(t_dates, y_fx_plot, color=color_fx,
                    linewidth=lw_plot, alpha=0.8, zorder=2)
            ax.plot(t_dates, y_diff, color=color_pred,
                    linewidth=lw_plot, alpha=0.8, zorder=3)

            ax.set_ylim(bottom=0)
            ax.grid(axis='y', alpha=0.2, linewidth=0.3)
            ax.set_ylabel('Streamflow', fontsize=9)
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

            title = f'Basin {bid}  |  f(x) vs Diffusion'
            if m_fx and m_diff:
                title += f'\nf(x): NSE={m_fx["nse"]:.3f}  |  Diff: NSE={m_diff["nse"]:.3f}'
            ax.set_title(title, fontsize=9, fontweight='bold', pad=4)

            legend_elements = [
                Line2D([0], [0], color=color_obs, linewidth=1.5, alpha=0.9, label='Observed'),
                Line2D([0], [0], color=color_fx, linewidth=1.5, alpha=0.8, label='f(x) Transformer'),
                Line2D([0], [0], color=color_pred, linewidth=1.5, alpha=0.8, label='Diffusion'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=6,
                      frameon=True, fancybox=False, edgecolor='#cccccc',
                      framealpha=0.9, borderpad=0.3, handlelength=1.2)

            plt.tight_layout()
            fig_path = os.path.join(fig_dir, f'{partition}_{bid}_fx_vs_diff.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"f(x) vs Diffusion plot saved: {fig_path}")

    # =========================================================
    # Step 9: Log metrics (optional)
    # =========================================================
    if args.target_basin and args.metrics_log:
        from datetime import datetime
        target_basins = args.target_basin.split()
        log_path = args.metrics_log.replace('.csv', f'_{partition}.csv')

        if os.path.isfile(log_path):
            df_log = pd.read_csv(log_path)
        else:
            df_log = pd.DataFrame()

        for tb in target_basins:
            match = [r for r in results if r['basin'] == tb]
            if not match:
                print(f"Warning: Basin {tb} not found")
                continue
            m = match[0]
            # Remove old entry for this basin + step_index
            if len(df_log) > 0:
                df_log = df_log[~((df_log['basin'] == tb) & (df_log['step_index'] == step_index))]
            new_row = pd.DataFrame([{
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'basin': tb, 'step_index': step_index,
                'nse': f'{m["nse"]:.6f}', 'kge': f'{m["kge"]:.6f}',
                'rmse': f'{m["rmse"]:.4f}', 'mae': f'{m["mae"]:.4f}',
                'r2': f'{m["r2"]:.6f}', 'pbias': f'{m["pbias"]:.2f}'
            }])
            df_log = pd.concat([df_log, new_row], ignore_index=True)
            print(f"Logged metrics for basin {tb} step_index={step_index}")

        df_log.to_csv(log_path, index=False)
        print(f"Metrics log: {log_path}")


if __name__ == "__main__":
    main()
