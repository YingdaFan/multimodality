"""
TN-only forecast analysis.

Inputs:
  - diffusion_forecast/output/pred/tst.npy           sliding-window predictions
  - diffusion_forecast/output/pred/meta.json         window / pred_len / stride
  - data_processing/data/prepped.npz                 y_raw_tst, times_tst, y_mean/std, basin_names
  - ../camelsh_tennessee.parquet                     TN basin_id list

Outputs (under output/tennessee/):
  - tn_metrics_long.csv         per-(basin, lead) metrics
  - tn_nse_wide.csv             wide NSE table (basins × lead times) for multi-step plots
  - figures/forecast_boxplot.png
  - figures/forecast_lines.png
  - figures/forecast_heatmap.png
  - figures/forecast_cdf.png
  - figures/forecast_retention.png
  - figures/timeseries_<basin>_jan2019.png   one per TN basin, 2019-01-01..01-31
"""

# --- pin CWD to imputation/ so relative paths resolve no matter where the
# --- script is launched from (the file now lives in Presentation/code/)
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])


import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================== Config ==============================
parser = argparse.ArgumentParser(description='TN-only forecast analysis')
parser.add_argument('--leads', default='7', choices=['7', '18'],
                    help="lead-time preset: '7' = 1/3/6/9/12/15/18h (sparse), "
                         "'18' = every step 1h..18h")
args = parser.parse_args()

LEAD_PRESETS = {
    '7':  [0, 2, 5, 8, 11, 14, 17],     # 1h, 3h, 6h, 9h, 12h, 15h, 18h
    '18': list(range(18)),              # every step 1h..18h
}
LEAD_STEPS  = LEAD_PRESETS[args.leads]

PRED_DIR    = 'diffusion_forecast/output/pred'
PREPPED     = 'data_processing/data/prepped.npz'
TN_PARQUET  = '../camelsh_tennessee.parquet'
OUTPUT_DIR  = f'Presentation/tennessee_{args.leads}lead'
PLOT_START  = np.datetime64('2019-01-01')
PLOT_END    = np.datetime64('2019-02-01')   # exclusive

# Basins to exclude from aggregate stats (mean/median/boxplot/heatmap/CDF/scatter).
# Data-quality issues, not model failure. Per-basin time-series still drawn for inspection.
#   03566420: trn 期只有 2 个观测且为异常点（已剔除），导致 y_mean/y_std 与测试期完全不同尺度
EXCLUDED_BASINS = {'03566420'}


# ============================== Load ==============================
with open(os.path.join(PRED_DIR, 'meta.json')) as f:
    meta = json.load(f)
window   = int(meta['window'])
pred_len = int(meta['pred_len'])
stride   = int(meta['stride'])
print(f'meta: window={window}, pred_len={pred_len}, stride={stride}')

d = np.load(PREPPED, allow_pickle=True)
y_raw       = d['y_raw_tst']        # (n_basins, n_times, 1)  already in raw units
times       = d['times_tst']        # (n_times,)
y_mean      = d['y_mean']           # (n_basins,)
y_std       = d['y_std']            # (n_basins,)
basin_names = [str(b) for b in d['basin_names']]
n_basins    = len(basin_names)
n_times     = y_raw.shape[1]

max_offsets = n_times - window - pred_len + 1
n_offsets   = (max_offsets - 1) // stride + 1

y_pred_all = np.load(os.path.join(PRED_DIR, 'tst.npy'))  # (n_offsets * n_basins, pred_len, 1)
print(f'predictions: {y_pred_all.shape}')
assert y_pred_all.shape[0] == n_basins * n_offsets, 'sample count mismatch'

# TN subset
tn_ids = sorted(set(pd.read_parquet(TN_PARQUET, columns=['basin_id'])['basin_id'].unique()))
name_to_idx = {b: i for i, b in enumerate(basin_names)}
tn_idx = [name_to_idx[b] for b in tn_ids if b in name_to_idx]
tn_basins = [basin_names[i] for i in tn_idx]
print(f'TN basins to process: {len(tn_basins)} / {len(tn_ids)} requested')

os.makedirs(OUTPUT_DIR, exist_ok=True)
fig_dir = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(fig_dir, exist_ok=True)


# ============================== Per-lead extraction + metrics ==============================
def compute_lead(step_index):
    """Return (metrics_df, t_pred, pred_dict, obs_dict) for TN basins at this lead time."""
    y_pred_step = y_pred_all[:, step_index, 0]
    y_pred_by_basin = y_pred_step.reshape(n_offsets, n_basins).T          # (n_basins, n_offsets)
    gt_indices = np.array([k * stride + window + step_index for k in range(n_offsets)])
    y_gt_by_basin = y_raw[:, gt_indices, 0]
    t_pred = times[gt_indices]

    rows = []
    pred_dict = {}
    obs_dict = {}
    for i in tn_idx:
        b = basin_names[i]
        y_p = y_pred_by_basin[i] * y_std[i] + y_mean[i]
        y_p = np.maximum(y_p, 0)
        y_g = y_gt_by_basin[i]
        pred_dict[b] = y_p
        obs_dict[b] = y_g

        mask = ~np.isnan(y_g) & ~np.isnan(y_p)
        nv = int(mask.sum())
        row = {'basin': b, 'step_index': step_index, 'hours_ahead': step_index + 1,
               'n_valid': nv}
        if nv < 2:
            row.update({'nse': np.nan, 'kge': np.nan, 'rmse': np.nan,
                        'mae': np.nan, 'r2': np.nan, 'pbias': np.nan})
        else:
            yp_v, yg_v = y_p[mask], y_g[mask]
            rmse = np.sqrt(np.mean((yp_v - yg_v) ** 2))
            mae = np.mean(np.abs(yp_v - yg_v))
            obs_mean = np.mean(yg_v)
            nse = 1 - np.sum((yg_v - yp_v) ** 2) / (np.sum((yg_v - obs_mean) ** 2) + 1e-10)
            r_val = stats.pearsonr(yg_v, yp_v)[0]
            alpha = np.std(yp_v) / (np.std(yg_v) + 1e-10)
            beta = np.mean(yp_v) / (np.mean(yg_v) + 1e-10)
            kge = 1 - np.sqrt((r_val - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
            pbias = np.sum(yg_v - yp_v) / (np.sum(yg_v) + 1e-10) * 100
            row.update({'nse': nse, 'kge': kge, 'rmse': rmse, 'mae': mae,
                        'r2': r_val ** 2, 'pbias': pbias})
        rows.append(row)
    return pd.DataFrame(rows), t_pred, pred_dict, obs_dict


all_metrics = []
horizon_data = {}
print('\n=== Per-lead metrics (TN basins only) ===')
for si in LEAD_STEPS:
    df_s, t_pred, pred_d, obs_d = compute_lead(si)
    all_metrics.append(df_s)
    horizon_data[si] = (t_pred, pred_d, obs_d)
    valid = df_s.dropna(subset=['nse'])
    print(f'  step={si} ({si+1}h-ahead): valid NSE basins = {len(valid)}/{len(df_s)}, '
          f'mean NSE = {valid["nse"].mean():.4f}, median NSE = {valid["nse"].median():.4f}')

metrics_long = pd.concat(all_metrics, ignore_index=True)
metrics_long.to_csv(os.path.join(OUTPUT_DIR, 'tn_metrics_long.csv'), index=False)
print(f'Saved {os.path.join(OUTPUT_DIR, "tn_metrics_long.csv")}')

# Wide table for multi-step plots
pivot = metrics_long.pivot(index='basin', columns='hours_ahead', values='nse')
pivot.columns = [f'{int(c)}h_ahead' for c in pivot.columns]
pivot_all = pivot.copy()
pivot_all.to_csv(os.path.join(OUTPUT_DIR, 'tn_nse_wide.csv'))
pivot_valid = pivot.dropna()    # basins with valid NSE at every lead time
print(f'TN basins with valid NSE across all {len(LEAD_STEPS)} lead times: {len(pivot_valid)}')

# Drop excluded basins (data-quality issues) from aggregate stats
excluded_present = [b for b in EXCLUDED_BASINS if b in pivot_valid.index]
if excluded_present:
    print(f'Excluding {len(excluded_present)} basin(s) from aggregate stats '
          f'(reason: training-data quality): {excluded_present}')
    pivot_valid = pivot_valid.drop(index=excluded_present)
print(f'Basins entering aggregate stats / multi-step plots: {len(pivot_valid)}')


# ============================== Multi-step plots (ported from ForecastAnalysis.txt) ==============================
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
})

df = pivot_valid.reset_index()
lead_cols = [f'{si+1}h_ahead' for si in LEAD_STEPS]
lead_labels = [f'{si+1}-hour' for si in LEAD_STEPS]
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(LEAD_STEPS)))
n = len(df)
n_leads = len(LEAD_STEPS)
_crowded = n_leads > 8        # rotate ticks / drop per-box text when many leads
_rot = 45 if _crowded else 0


def _safe_ylim(values, pad=0.05):
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    span = max(hi - lo, 0.1)
    return lo - pad * span, min(hi + pad * span, 1.05)


# Figure 1: boxplot
fig, ax = plt.subplots(figsize=(10, 7))
data = [df[c] for c in lead_cols]
bp = ax.boxplot(data, tick_labels=lead_labels, patch_artist=True, widths=0.6,
                medianprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=6, alpha=0.6))
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax.set_ylabel('NSE')
ax.set_title(f'Forecast Performance by Lead Time (n = {n} basins)')
ax.set_ylim(*_safe_ylim(df[lead_cols].values))
if _crowded:
    plt.setp(ax.get_xticklabels(), rotation=_rot, ha='right')
else:
    medians = [df[c].median() for c in lead_cols]
    top = ax.get_ylim()[1]
    for i, m in enumerate(medians):
        ax.text(i + 1, top - 0.005, f'{m:.3f}', ha='center', va='top',
                fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'forecast_boxplot.png'), dpi=150, bbox_inches='tight'); plt.close()

# Figure 2: line plot (per-basin NSE degradation)
fig, ax = plt.subplots(figsize=(12, 7))
xs = list(range(1, len(LEAD_STEPS) + 1))
for _, row in df.iterrows():
    ax.plot(xs, [row[c] for c in lead_cols], color='gray', alpha=0.4, linewidth=0.8)
means = [df[c].mean() for c in lead_cols]
medians_v = [df[c].median() for c in lead_cols]
p25 = [df[c].quantile(0.25) for c in lead_cols]
p75 = [df[c].quantile(0.75) for c in lead_cols]
ax.fill_between(xs, p25, p75, alpha=0.4, color='steelblue', label='25th–75th percentile')
ax.plot(xs, means, 'o-', color='red', linewidth=3, markersize=10, label='Mean', zorder=5)
ax.plot(xs, medians_v, 's--', color='blue', linewidth=3, markersize=10, label='Median', zorder=5)
ax.set_xlabel('Forecast Lead Time (hours)')
ax.set_ylabel('NSE')
ax.set_title(f'NSE Degradation with Forecast Lead Time (n = {n} basins)')
ax.set_xticks(xs); ax.set_xticklabels(lead_labels)
if _crowded:
    plt.setp(ax.get_xticklabels(), rotation=_rot, ha='right')
ax.set_ylim(*_safe_ylim(df[lead_cols].values))
ax.legend(loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'forecast_lines.png'), dpi=150, bbox_inches='tight'); plt.close()

# Figure 3: heatmap (basins × lead time)
fig, ax = plt.subplots(figsize=(10, 14))
df_sorted = df.copy()
df_sorted['mean_NSE'] = df_sorted[lead_cols].mean(axis=1)
df_sorted = df_sorted.sort_values('mean_NSE', ascending=False).reset_index(drop=True)
heat = df_sorted[lead_cols].values
im = ax.imshow(heat, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=1.0)
cbar = plt.colorbar(im, ax=ax, shrink=0.8); cbar.set_label('NSE')
ax.set_ylabel('Basin (sorted by mean NSE)')
ax.set_title(f'Forecast Performance Heatmap (n = {n} basins)')
ax.set_xticks(range(len(LEAD_STEPS))); ax.set_xticklabels(lead_labels)
if _crowded:
    plt.setp(ax.get_xticklabels(), rotation=_rot, ha='right')
ax.set_yticks(range(0, n, max(1, n // 15)))
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'forecast_heatmap.png'), dpi=150, bbox_inches='tight'); plt.close()

# Figure 4: CDF
fig, ax = plt.subplots(figsize=(10, 7))
for c, lbl, col in zip(lead_cols, lead_labels, colors):
    data = df[c].sort_values()
    cdf = np.arange(1, len(data) + 1) / len(data)
    ax.plot(data, cdf, label=lbl, color=col, linewidth=2.5)
ax.set_xlabel('NSE'); ax.set_ylabel('Cumulative Probability')
ax.set_title(f'CDF of NSE by Forecast Lead Time (n = {n} basins)')
xmin, xmax = _safe_ylim(df[lead_cols].values)
ax.set_xlim(xmin, min(xmax, 1.0))
ax.set_ylim(0, 1)
ax.legend(loc='upper left', ncol=2 if _crowded else 1, fontsize=9 if _crowded else 12)
ax.grid(True, alpha=0.3)
for t in [0.7, 0.9]:
    ax.axvline(t, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(t + 0.005, 0.05, f'NSE={t}', fontsize=11, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'forecast_cdf.png'), dpi=150, bbox_inches='tight'); plt.close()

# Figure 5: skill retention (1h vs last lead)
fig, ax = plt.subplots(figsize=(9, 8))
last = lead_cols[-1]
retention = df[last] / df[lead_cols[0]]
sc = ax.scatter(df[lead_cols[0]], df[last], c=retention, cmap='RdYlGn',
                s=80, alpha=0.75, edgecolor='white', linewidth=0.5,
                vmin=0.85, vmax=1.0)
cbar = plt.colorbar(sc, ax=ax); cbar.set_label(f'Skill Retention ({lead_labels[-1]}/{lead_labels[0]})')
lo = min(df[lead_cols[0]].min(), df[last].min())
ax.plot([lo, 1.0], [lo, 1.0], 'k--', linewidth=2, label='1:1 line')
ax.set_xlabel(f'{lead_labels[0]} NSE'); ax.set_ylabel(f'{lead_labels[-1]} NSE')
ax.set_title(f'Skill Retention: {lead_labels[0]} vs {lead_labels[-1]} Forecast (n = {n} basins)')
ax.set_xlim(lo - 0.02, 1.02); ax.set_ylim(lo - 0.02, 1.02)
ax.text(0.05, 0.95, f'Mean retention: {retention.mean():.3f}', transform=ax.transAxes,
        fontsize=13, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'forecast_retention.png'), dpi=150, bbox_inches='tight'); plt.close()

print('\nMulti-step figures saved:')
for f in ['forecast_boxplot.png', 'forecast_lines.png', 'forecast_heatmap.png',
          'forecast_cdf.png', 'forecast_retention.png']:
    print(f'  {os.path.join(fig_dir, f)}')


# ============================== Per-basin Jan-2019 time series ==============================
# Plot obs (full Jan 2019) + 1h-ahead prediction curve within Jan 2019.
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
})

t_pred0, pred0, _ = horizon_data[0]
obs_mask = (times >= PLOT_START) & (times < PLOT_END)
pred_mask = (t_pred0 >= PLOT_START) & (t_pred0 < PLOT_END)
t_obs_jan = pd.to_datetime(times[obs_mask])
t_pred_jan = pd.to_datetime(t_pred0[pred_mask])

print(f'\nGenerating per-basin Jan-2019 time series for {len(tn_basins)} TN basins...')
metrics_by_basin = {r['basin']: r for r in all_metrics[0].to_dict(orient='records')}  # 1h-ahead

for b in tn_basins:
    i = name_to_idx[b]
    y_obs_jan = y_raw[i, obs_mask, 0]
    y_pred_jan = pred0[b][pred_mask]

    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(t_obs_jan, y_obs_jan, 'o', color='#1f77b4', markersize=2.5,
            alpha=0.75, label='Observed')
    ax.plot(t_pred_jan, y_pred_jan, '-', color='#d62728',
            linewidth=1.0, alpha=0.9, label='1h-ahead prediction')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Streamflow')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    m = metrics_by_basin.get(b, {})
    nse_v = m.get('nse', np.nan); kge_v = m.get('kge', np.nan); rmse_v = m.get('rmse', np.nan)
    title = f'Basin {b}  |  Jan 2019  |  1h-ahead'
    if np.isfinite(nse_v):
        title += f'  |  NSE={nse_v:.3f}  KGE={kge_v:.3f}  RMSE={rmse_v:.2f}'
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linewidth=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'timeseries_{b}_jan2019.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

print(f'Per-basin time series saved to {fig_dir}/timeseries_<basin>_jan2019.png')


# ============================== Summary ==============================
print('\n' + '=' * 70)
print('DETAILED STATISTICS (TN basins, valid across all lead times)')
print('=' * 70)
print(f'\n{"Lead Time":<12} {"Mean":>8} {"Median":>8} {"Std":>8} {"Min":>8} {"Max":>8}')
print('-' * 60)
for c, lbl in zip(lead_cols, lead_labels):
    data = df[c]
    print(f'{lbl:<12} {data.mean():>8.3f} {data.median():>8.3f} {data.std():>8.3f} '
          f'{data.min():>8.3f} {data.max():>8.3f}')

for thr in (0.7, 0.9, 0.95):
    print(f'\nBasins with NSE ≥ {thr}:')
    for c, lbl in zip(lead_cols, lead_labels):
        cnt = (df[c] >= thr).sum()
        print(f'  {lbl}: {cnt}/{n} ({100*cnt/n:.1f}%)')

print('\nAll outputs in:', OUTPUT_DIR)
