"""
Compose a single page-9-style summary figure mimicking InflowForecast.pdf.

Reads:
  - output/tennessee/tn_nse_wide.csv   (basins × lead times, NSE)

Outputs:
  - output/tennessee/figures/page9_summary.png   single combined figure
"""

# --- pin CWD to imputation/ so relative paths resolve no matter where the
# --- script is launched from (the file now lives in Presentation/code/)
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])


import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import slide_style as ss


# ============================== Config ==============================
parser = argparse.ArgumentParser(description='Page-9-style summary figure')
parser.add_argument('--leads', default='7', choices=['7', '18'],
                    help="which analyze_tennessee.py run to summarize")
args = parser.parse_args()

WIDE_CSV   = f'Presentation/tennessee_{args.leads}lead/tn_nse_wide.csv'
OUT_FIG    = f'Presentation/tennessee_{args.leads}lead/figures/page9_summary.png'

# Match analyze_tennessee.py exclusion (data-quality issues, not model failure)
EXCLUDED_BASINS = {'03566420'}


# ============================== Load ==============================
df = pd.read_csv(WIDE_CSV, dtype={'basin': str})
lead_cols = [c for c in df.columns if c.endswith('h_ahead')]
lead_cols.sort(key=lambda c: int(c.split('h_')[0]))
lead_labels = [f'{int(c.split("h_")[0])}-hour' for c in lead_cols]

# Filter: must have valid NSE for ALL lead times + drop excluded
df = df.dropna(subset=lead_cols)
df = df[~df['basin'].isin(EXCLUDED_BASINS)].reset_index(drop=True)
n = len(df)
n_leads = len(lead_cols)
print(f'Basins entering page-9 summary: {n}  ({n_leads} lead times: {lead_labels})')

# Data-driven lower bound: 0.6 normally, extends lower if longer leads dip below
_dmin = float(np.nanmin(df[lead_cols].values))
Y_LO = min(0.6, np.floor(_dmin * 10) / 10)
Y_HI = 1.02

_crowded = n_leads > 8        # rotate ticks / thin out text labels when many leads
_rot = 45 if _crowded else 0


# ============================== Figure ==============================
ss.apply_rc()
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
})

# Green palette inspired by PDF (dark→light green), adaptive to lead count
lead_colors = plt.cm.YlGn_r(np.linspace(0.12, 0.82, n_leads))

fig = plt.figure(figsize=(22, 11))
gs = gridspec.GridSpec(
    nrows=2, ncols=3,
    width_ratios=[1.0, 1.0, 0.95],
    height_ratios=[1.0, 1.0],
    figure=fig,
    left=0.045, right=0.965, top=0.87, bottom=0.09,
    hspace=0.30, wspace=0.20,
)

ax_box  = fig.add_subplot(gs[0, 0])
ax_cdf  = fig.add_subplot(gs[0, 1])
ax_heat = fig.add_subplot(gs[:, 2])     # spans both rows
ax_deg  = fig.add_subplot(gs[1, :2])    # spans bottom two cols


# ----- 1) Boxplot: NSE by lead time --------------------------------
data_box = [df[c].values for c in lead_cols]
bp = ax_box.boxplot(
    data_box, tick_labels=lead_labels, patch_artist=True, widths=0.55,
    medianprops=dict(color='black', linewidth=1.6),
    whiskerprops=dict(linewidth=0.8),
    capprops=dict(linewidth=0.8),
    flierprops=dict(marker='o', markerfacecolor='dimgray',
                    markeredgecolor='dimgray', markersize=4, alpha=0.6),
)
for patch, c in zip(bp['boxes'], lead_colors):
    patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor('black')

ax_box.set_ylabel('NSE')
ax_box.set_title(f'Forecast Performance by Lead Time (n = {n} basins)')
ax_box.set_ylim(Y_LO, Y_HI)
ax_box.grid(axis='y', alpha=0.3, linewidth=0.4)
if _crowded:
    plt.setp(ax_box.get_xticklabels(), rotation=_rot, ha='right')

# Median labels above each box (skip when too crowded to read)
if not _crowded:
    _box_label_fs = 11 if n_leads <= 5 else 9
    for i, c in enumerate(lead_cols, start=1):
        m = df[c].median()
        ax_box.text(i, Y_HI - 0.012, f'{m:.3f}', ha='center', va='bottom',
                    fontsize=_box_label_fs, fontweight='bold')


# ----- 2) CDF -----------------------------------------------------
for c, lbl, col in zip(lead_cols, lead_labels, lead_colors):
    data = np.sort(df[c].values)
    cdf  = np.arange(1, len(data) + 1) / len(data)
    ax_cdf.plot(data, cdf, label=lbl, color=col, linewidth=2.5)

ax_cdf.set_xlabel('NSE')
ax_cdf.set_ylabel('Cumulative Probability')
ax_cdf.set_title(f'CDF of NSE by Forecast Lead Time (n = {n} basins)')
ax_cdf.set_xlim(Y_LO, 1.0)
ax_cdf.set_ylim(0, 1.0)
ax_cdf.grid(True, alpha=0.3, linewidth=0.4)
ax_cdf.legend(loc='upper left', ncol=2 if n_leads > 5 else 1)
for t in (0.7, 0.9):
    ax_cdf.axvline(t, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
    ax_cdf.text(t + 0.005, 0.5, f'NSE={t}', fontsize=10, color='gray',
                rotation=90, va='center')


# ----- 3) Heatmap (right column, spans both rows) ------------------
df_sorted = df.copy()
df_sorted['mean_NSE'] = df_sorted[lead_cols].mean(axis=1)
df_sorted = df_sorted.sort_values('mean_NSE', ascending=False).reset_index(drop=True)
heat = df_sorted[lead_cols].values

im = ax_heat.imshow(heat, aspect='auto', cmap='RdYlGn', vmin=0.5, vmax=1.0)
cbar = plt.colorbar(im, ax=ax_heat, shrink=0.85, pad=0.02)
cbar.set_label('NSE')
ax_heat.set_ylabel('Basin (sorted by mean NSE)')
ax_heat.set_title(f'Forecast Performance Heatmap (n = {n} basins)')
ax_heat.set_xticks(range(len(lead_cols)))
ax_heat.set_xticklabels(lead_labels)
if _crowded:
    plt.setp(ax_heat.get_xticklabels(), rotation=_rot, ha='right')
ax_heat.set_yticks(range(0, n, max(1, n // 12)))


# ----- 4) NSE Degradation (bottom, spans 2 cols) -------------------
xs = list(range(1, len(lead_cols) + 1))

# Individual basin trajectories (gray)
for _, row in df.iterrows():
    ax_deg.plot(xs, [row[c] for c in lead_cols],
                color='gray', alpha=0.35, linewidth=0.6)

means    = [df[c].mean()        for c in lead_cols]
medians  = [df[c].median()      for c in lead_cols]
p25      = [df[c].quantile(0.25) for c in lead_cols]
p75      = [df[c].quantile(0.75) for c in lead_cols]

ax_deg.fill_between(xs, p25, p75, alpha=0.35, color='steelblue',
                    label='25th–75th percentile')
ax_deg.plot(xs, means,   'o-',  color='red',  linewidth=2.5, markersize=9,
            label='Mean',   zorder=5)
ax_deg.plot(xs, medians, 's--', color='blue', linewidth=2.5, markersize=9,
            label='Median', zorder=5)

ax_deg.set_xlabel('Forecast Lead Time (hours)')
ax_deg.set_ylabel('NSE')
ax_deg.set_title(f'NSE Degradation with Forecast Lead Time (n = {n} basins)')
ax_deg.set_xticks(xs)
ax_deg.set_xticklabels(lead_labels)
ax_deg.set_ylim(Y_LO, Y_HI)
ax_deg.grid(True, alpha=0.3, linewidth=0.4)
ax_deg.legend(loc='lower left')
if _crowded:
    plt.setp(ax_deg.get_xticklabels(), rotation=_rot, ha='right')

# Annotate Med/Mean: every lead when sparse, else only first + last to avoid clutter
_deg_label_fs = 10 if n_leads <= 5 else 8
_span = Y_HI - Y_LO
_annot_idx = range(n_leads) if not _crowded else (0, n_leads - 1)
for i in _annot_idx:
    m, md = means[i], medians[i]
    ax_deg.annotate(f'Med: {md:.3f}', xy=(i + 1, md),
                    xytext=(i + 1, Y_HI - 0.015 * _span / 0.42), ha='center',
                    fontsize=_deg_label_fs, color='blue')
    ax_deg.annotate(f'Mean: {m:.3f}', xy=(i + 1, m),
                    xytext=(i + 1.04, m - 0.06 * _span), ha='left',
                    fontsize=_deg_label_fs, color='red')


# --- deck-style chrome: title banner + page badge (no callout — charts fill the slide) ---
ss.add_title_banner(
    fig, f'Global FutureTST Streamflow Forecast (1–{lead_labels[-1]})',
    fontsize=21)

os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
plt.savefig(OUT_FIG, dpi=200)
plt.close()
print(f'Saved: {OUT_FIG}')


# ============================== Stats summary ==============================
print('\n' + '=' * 60)
print(f'PAGE-9 SUMMARY STATS (n = {n} basins)')
print('=' * 60)
print(f'{"Lead Time":<10} {"Mean":>8} {"Median":>8} {"Std":>8} {"Min":>8} {"Max":>8}')
print('-' * 56)
for c, lbl in zip(lead_cols, lead_labels):
    s = df[c]
    print(f'{lbl:<10} {s.mean():>8.3f} {s.median():>8.3f} {s.std():>8.3f} '
          f'{s.min():>8.3f} {s.max():>8.3f}')

for thr in (0.7, 0.9, 0.95):
    print(f'\nBasins with NSE ≥ {thr}:')
    for c, lbl in zip(lead_cols, lead_labels):
        cnt = (df[c] >= thr).sum()
        print(f'  {lbl}: {cnt}/{n} ({100*cnt/n:.1f}%)')
