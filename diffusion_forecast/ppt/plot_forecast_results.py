"""
Multi-step Streamflow Forecasting Performance Visualization
============================================================
Adapted from teacher's ForecastMultistep.txt for NsDiff model.
Lead times: 1h, 6h, 12h, 18h ahead (vs teacher's 1h, 2h, 3h, 4h).
Data source: tst_decay_metrics.csv (108 basins).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 14

# ---- Load and pivot data ----
csv_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'denorm', 'tst_decay_metrics.csv')
out_dir = os.path.dirname(__file__)

raw = pd.read_csv(csv_path)

# Pivot: each row = one basin, columns = NSE at each lead time
df = raw.pivot_table(index='basin', columns='hours_ahead', values='nse').reset_index()

# Identify lead times from data
lead_hours = sorted(raw['hours_ahead'].unique())  # e.g. [1, 6, 12, 18]
lead_times = [f'{h}h_ahead' for h in lead_hours]
lead_labels = [f'{h}-hour' for h in lead_hours]

# Rename columns to match teacher's convention
df.columns = ['basin'] + lead_times
n_basins = len(df)

colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(lead_hours)))

# ---- Determine axis limits ----
all_nse = df[lead_times].values.flatten()
all_nse_valid = all_nse[~np.isnan(all_nse)]
ymin = max(np.floor(all_nse_valid.min() * 10) / 10 - 0.1, -0.5)
ymin = min(ymin, 0.0)  # at most start from 0

print("=" * 70)
print("MULTI-STEP FORECASTING PERFORMANCE SUMMARY")
print("=" * 70)
print(f"Number of basins: {n_basins}")
print(f"\nNSE Statistics by Lead Time:")
for lt, label in zip(lead_times, lead_labels):
    data = df[lt].dropna()
    print(f"  {label}: Mean={data.mean():.3f}, Median={data.median():.3f}, "
          f"Min={data.min():.3f}, Max={data.max():.3f}")

# =============================================================================
# Figure 1: Box Plot - NSE Distribution by Lead Time
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 7))

data_to_plot = [df[lt].dropna() for lt in lead_times]

bp = ax.boxplot(data_to_plot, tick_labels=lead_labels, patch_artist=True, widths=0.6,
                medianprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=6, alpha=0.6))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('NSE', fontsize=18)
ax.set_title(f'Forecast Performance by Lead Time (n = {n_basins} basins)', fontsize=20)
ax.set_ylim(ymin, 1.05)

# Add median values on plot
medians = [df[lt].median() for lt in lead_times]
for i, med in enumerate(medians):
    ax.text(i + 1, 1.02, f'{med:.3f}', ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'Forecast_Boxplot.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure 1: Box plot saved")

# =============================================================================
# Figure 2: Line Plot - NSE Degradation with Lead Time (All Basins)
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

x_pos = list(range(1, len(lead_hours) + 1))

# Plot individual basin lines (light gray)
for idx, row in df.iterrows():
    nse_values = [row[lt] for lt in lead_times]
    ax.plot(x_pos, nse_values, color='gray', alpha=0.5, linewidth=0.8)

# Plot mean and median
mean_values = [df[lt].mean() for lt in lead_times]
median_values = [df[lt].median() for lt in lead_times]
percentile_25 = [df[lt].quantile(0.25) for lt in lead_times]
percentile_75 = [df[lt].quantile(0.75) for lt in lead_times]

ax.fill_between(x_pos, percentile_25, percentile_75, alpha=0.5, color='steelblue',
                label='25th-75th percentile')
ax.plot(x_pos, mean_values, 'o-', color='red', linewidth=3, markersize=10,
        label='Mean', zorder=5)
ax.plot(x_pos, median_values, 's--', color='blue', linewidth=3, markersize=10,
        label='Median', zorder=5)

ax.set_xlabel('Forecast Lead Time (hours)', fontsize=18)
ax.set_ylabel('NSE', fontsize=18)
ax.set_title(f'NSE Degradation with Forecast Lead Time (n = {n_basins} basins)', fontsize=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(lead_labels)
ax.set_ylim(ymin, 1.05)
ax.legend(loc='lower left', fontsize=14)

# Add text annotations for mean/median values
for i, (m, md) in enumerate(zip(mean_values, median_values)):
    ax.annotate(f'Mean: {m:.3f}', xy=(x_pos[i], m),
                xytext=(x_pos[i] - 0.1, m - 0.04), fontsize=11)
    ax.annotate(f'Med: {md:.3f}', xy=(x_pos[i], md),
                xytext=(x_pos[i] - 0.1, 1.02), fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'Forecast_lines.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2: NSE degradation line plot saved")

# =============================================================================
# Figure 3: Heatmap - Basin x Lead Time Performance
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 14))

# Sort basins by mean NSE across all lead times
df['mean_NSE'] = df[lead_times].mean(axis=1)
df_sorted = df.sort_values('mean_NSE', ascending=False).reset_index(drop=True)

heatmap_data = df_sorted[lead_times].values

im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=max(ymin, 0), vmax=1.0)
cbar = plt.colorbar(im, ax=ax, label='NSE', shrink=0.8)
cbar.ax.tick_params(labelsize=16)
cbar.set_label('NSE', fontsize=16)

ax.set_ylabel('Basin (sorted by mean NSE)', fontsize=18)
ax.set_title(f'Forecast Performance Heatmap (n = {n_basins} basins)', fontsize=20)
ax.set_xticks(range(len(lead_labels)))
ax.set_xticklabels(lead_labels)
ax.set_yticks(range(0, len(df_sorted), 10))

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'Forecast_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3: Heatmap saved")

# =============================================================================
# Figure 4: CDF Comparison by Lead Time
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 7))

for lt, label, color in zip(lead_times, lead_labels, colors):
    data = df[lt].dropna().sort_values()
    cdf = np.arange(1, len(data) + 1) / len(data)
    ax.plot(data, cdf, label=label, color=color, linewidth=2.5)

ax.set_xlabel('NSE', fontsize=18)
ax.set_ylabel('Cumulative Probability', fontsize=18)
ax.set_title(f'CDF of NSE by Forecast Lead Time (n = {n_basins} basins)', fontsize=20)
ax.set_xlim(max(ymin, 0), 1.0)
ax.set_ylim(0., 1.0)
ax.legend(loc='upper left', fontsize=14)
ax.grid(True, alpha=0.3)

# Add threshold lines
for thresh in [0.5, 0.7]:
    ax.axvline(x=thresh, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(thresh + 0.01, 0.65, f'NSE={thresh}', fontsize=12, color='gray')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'Forecast_CDF.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4: CDF comparison saved")

# =============================================================================
# Figure 5: Scatter Plot - 1h vs last lead time NSE (Skill Retention)
# =============================================================================
first_lt = lead_times[0]
last_lt = lead_times[-1]
first_label = lead_labels[0]
last_label = lead_labels[-1]

fig, ax = plt.subplots(figsize=(9, 8))

skill_retention = df[last_lt] / df[first_lt]
scatter = ax.scatter(df[first_lt], df[last_lt], c=skill_retention,
                     cmap='RdYlGn', s=80, alpha=0.7, edgecolor='white', linewidth=0.5,
                     vmin=0.3, vmax=1.0)

cbar = plt.colorbar(scatter, ax=ax, label=f'Skill Retention ({last_label}/{first_label})')
cbar.ax.tick_params(labelsize=14)
cbar.set_label(f'Skill Retention ({last_label}/{first_label})', fontsize=14)

ax.plot([-0.5, 1], [-0.5, 1], 'k--', linewidth=2, label='1:1 line')
ax.set_xlabel(f'{first_label} Ahead NSE', fontsize=18)
ax.set_ylabel(f'{last_label} Ahead NSE', fontsize=18)
ax.set_title(f'Skill Retention: {first_label} vs {last_label} Forecast (n = {n_basins} basins)',
             fontsize=18)
ax.set_xlim(ymin, 1.02)
ax.set_ylim(ymin, 1.02)

mean_retention = skill_retention.mean()
ax.text(0.05, 0.95, f'Mean skill retention: {mean_retention:.3f}',
        transform=ax.transAxes, fontsize=14, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'Forecast_retention.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5: Skill retention scatter plot saved")

# =============================================================================
# Summary Statistics
# =============================================================================
print("\n" + "=" * 70)
print("DETAILED STATISTICS")
print("=" * 70)

print("\nNSE Performance Metrics:")
print("-" * 60)
print(f"{'Lead Time':<12} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 60)
for lt, label in zip(lead_times, lead_labels):
    data = df[lt].dropna()
    print(f"{label:<12} {data.mean():>8.3f} {data.median():>8.3f} {data.std():>8.3f} "
          f"{data.min():>8.3f} {data.max():>8.3f}")

print(f"\n\nBasins with NSE >= 0.9 at each lead time:")
for lt, label in zip(lead_times, lead_labels):
    count = (df[lt] >= 0.9).sum()
    pct = 100 * count / n_basins
    print(f"  {label}: {count}/{n_basins} ({pct:.1f}%)")

print(f"\n\nBasins with NSE >= 0.7 at each lead time:")
for lt, label in zip(lead_times, lead_labels):
    count = (df[lt] >= 0.7).sum()
    pct = 100 * count / n_basins
    print(f"  {label}: {count}/{n_basins} ({pct:.1f}%)")

print(f"\n\nMean Skill Retention ({last_label}/{first_label}): {skill_retention.mean():.3f}")
print(f"Median Skill Retention ({last_label}/{first_label}): {skill_retention.median():.3f}")

print("\n" + "=" * 70)
print(f"All figures saved to {out_dir}")
print("=" * 70)
