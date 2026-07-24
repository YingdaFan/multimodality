"""
Plot obs vs pred time series for selected basins (Page 7 style).
Uses hourly obs from raw data + 1-step-ahead forecasting predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import os

# ---- Config ----
YEAR = 2018
BASINS = ['03578500', '03463300', '03460000', '03576500',
          '03518500', '03601990', '03447687', '03498850']
out_dir = os.path.dirname(__file__)
denorm_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'denorm')
data_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data_processing', 'data', 'prepped.npz')

# ---- Load raw hourly data ----
npz = np.load(data_file, allow_pickle=True)
basin_names = list(npz['basin_names'].astype(str))
times_tst = pd.to_datetime(npz['times_tst'])
y_raw = npz['y_raw_tst']      # (n_basins, n_times, 1)
y_mean = npz['y_mean']
y_std = npz['y_std']

# ---- Plot each basin ----
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
})

for bid in BASINS:
    if bid not in basin_names:
        print(f"Basin {bid} not found, skipping")
        continue

    idx = basin_names.index(bid)

    # Hourly obs for the selected year
    year_mask = times_tst.year == YEAR
    t_hourly = times_tst[year_mask]
    obs_hourly = y_raw[idx, year_mask, 0]

    # Daily pred from denorm CSV
    csv_path = os.path.join(denorm_dir, f'tst_{bid}_step0.csv')
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    df_year = df[df['time'].dt.year == YEAR].copy()
    t_pred = df_year['time']
    pred_vals = df_year['pred'].values
    obs_daily = df_year['obs'].values

    # Compute metrics on matched daily points
    mask = ~np.isnan(obs_daily) & ~np.isnan(pred_vals)
    obs_v = obs_daily[mask]
    pred_v = pred_vals[mask]

    if len(obs_v) < 2:
        print(f"Basin {bid}: insufficient data for {YEAR}")
        continue

    obs_mean_val = np.mean(obs_v)
    nse = 1 - np.sum((obs_v - pred_v)**2) / (np.sum((obs_v - obs_mean_val)**2) + 1e-10)
    rmse = np.sqrt(np.mean((pred_v - obs_v)**2))
    r_val = stats.pearsonr(obs_v, pred_v)[0]
    alpha = np.std(pred_v) / (np.std(obs_v) + 1e-10)
    beta = np.mean(pred_v) / (np.mean(obs_v) + 1e-10)
    kge = 1 - np.sqrt((r_val - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7, 2.8))

    # Hourly obs (black solid)
    ax.plot(t_hourly, obs_hourly, color='black', linewidth=0.6, alpha=0.9,
            label='Obs', zorder=3)
    # Daily pred (red dashed)
    ax.plot(t_pred, pred_vals, color='red', linewidth=0.6, alpha=0.85,
            linestyle='--', label='Pred', zorder=2)

    ax.set_ylim(bottom=0)
    ax.set_ylabel('Q (cms)', fontsize=11)
    ax.set_xlim(pd.Timestamp(f'{YEAR}-01-01'), pd.Timestamp(f'{YEAR}-12-01'))

    # X-axis: monthly ticks like the teacher's plot
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # Title with basin ID
    ax.set_title(f'{bid}', fontsize=12, fontweight='bold')

    # Metrics box (top right, like teacher's)
    metrics_text = f'NSE={nse:.2f}\nKGE={kge:.2f}\nRMSE={rmse:.1f}'
    ax.text(0.98, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.9))

    # Legend
    ax.legend(loc='upper left', fontsize=8, frameon=True, ncol=2)

    ax.grid(axis='y', alpha=0.2, linewidth=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, f'Basin_{bid}_{YEAR}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}  |  NSE={nse:.3f}  KGE={kge:.3f}  RMSE={rmse:.2f}")

print(f"\nAll basin plots saved to {out_dir}")
