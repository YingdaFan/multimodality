"""
Deck-styled individual versions of the three per-lead-time result panels that
also appear inside page9_summary.png — broken out one-per-slide so each can
be enlarged and discussed on its own page (slides 7 / 8 / 9 of the deck).

Reads:  output/tennessee_18lead/tn_nse_wide.csv

Outputs:
  output/tennessee_18lead/figures/forecast_boxplot.png   (slide 7)
  output/tennessee_18lead/figures/forecast_cdf.png       (slide 8)
  output/tennessee_18lead/figures/forecast_lines.png     (slide 9)

Same green palette (YlGn_r) and data-driven y-range as page9_summary so the
four slides feel like one coherent block.
"""

# --- pin CWD to imputation/ so relative paths resolve no matter where the
# --- script is launched from (the file now lives in Presentation/code/)
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import slide_style as ss


WIDE_CSV = 'Presentation/tennessee_18lead/tn_nse_wide.csv'
OUT_DIR  = 'Presentation/tennessee_18lead/figures'
EXCLUDED = {'03566420'}

os.makedirs(OUT_DIR, exist_ok=True)
ss.apply_rc()
plt.rcParams.update({
    'figure.dpi':       150,
    'font.size':        12,
    'axes.titlesize':   14,
    'axes.labelsize':   13,
    'xtick.labelsize':  11,
    'ytick.labelsize':  11,
    'legend.fontsize':  11,
})


# ============================== Load ==============================
df = pd.read_csv(WIDE_CSV, dtype={'basin': str})
lead_cols   = [c for c in df.columns if c.endswith('h_ahead')]
lead_cols.sort(key=lambda c: int(c.split('h_')[0]))
lead_labels = [f'{int(c.split("h_")[0])}-hour' for c in lead_cols]
df = df.dropna(subset=lead_cols)
df = df[~df['basin'].isin(EXCLUDED)].reset_index(drop=True)
n        = len(df)
n_leads  = len(lead_cols)
_dmin    = float(np.nanmin(df[lead_cols].values))
Y_LO     = min(0.6, np.floor(_dmin * 10) / 10)
Y_HI     = 1.05

print(f'Basins: {n}    Leads: {n_leads}')

lead_colors = plt.cm.YlGn_r(np.linspace(0.12, 0.82, n_leads))
_crowded = n_leads > 8


# ============================== 1) Boxplot (slide 7) ==============================
fig = plt.figure(figsize=(16, 8.5))
ax  = fig.add_axes([0.07, 0.13, 0.90, 0.70])

bp = ax.boxplot(
    [df[c].values for c in lead_cols],
    tick_labels=lead_labels, patch_artist=True, widths=0.6,
    medianprops=dict(color='black', linewidth=1.8),
    whiskerprops=dict(linewidth=0.8),
    capprops=dict(linewidth=0.8),
    flierprops=dict(marker='o', markerfacecolor='dimgray',
                    markeredgecolor='dimgray', markersize=4, alpha=0.6),
)
for patch, c in zip(bp['boxes'], lead_colors):
    patch.set_facecolor(c); patch.set_alpha(0.88); patch.set_edgecolor('black')

ax.set_ylabel('NSE')
ax.set_ylim(Y_LO, Y_HI)
ax.grid(axis='y', alpha=0.3, linewidth=0.4)
if _crowded:
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Median label above each box — hung just under the upper spine so the
# numbers don't collide with it (va='top' keeps top of text inside the axes).
for i, c in enumerate(lead_cols, start=1):
    m = df[c].median()
    ax.text(i, Y_HI - 0.005, f'{m:.3f}', ha='center', va='top',
            fontsize=9, fontweight='bold')

ss.add_title_banner(
    fig, f'NSE by Lead Time — Box Plot  (n = {n} TRB basins)',
    fontsize=20)
plt.savefig(os.path.join(OUT_DIR, 'forecast_boxplot.png'), dpi=200)
plt.close()
print('Saved forecast_boxplot.png')


# ============================== 2) CDF (slide 8) ==============================
fig = plt.figure(figsize=(16, 8.5))
ax  = fig.add_axes([0.07, 0.13, 0.90, 0.70])

for c, lbl, col in zip(lead_cols, lead_labels, lead_colors):
    data = np.sort(df[c].values)
    cdf  = np.arange(1, len(data) + 1) / len(data)
    ax.plot(data, cdf, label=lbl, color=col, linewidth=2.5)

ax.set_xlabel('NSE')
ax.set_ylabel('Cumulative Probability')
ax.set_xlim(Y_LO, 1.0)
ax.set_ylim(0, 1.0)
ax.grid(True, alpha=0.3, linewidth=0.4)
ax.legend(loc='upper left', ncol=2 if n_leads > 5 else 1)
for t in (0.7, 0.9):
    ax.axvline(t, color='gray', linestyle=':', linewidth=1.3, alpha=0.7)
    ax.text(t + 0.005, 0.5, f'NSE = {t}', fontsize=10, color='gray',
            rotation=90, va='center')

ss.add_title_banner(
    fig, f'NSE by Lead Time — Cumulative Distribution  (n = {n} TRB basins)',
    fontsize=20)
plt.savefig(os.path.join(OUT_DIR, 'forecast_cdf.png'), dpi=200)
plt.close()
print('Saved forecast_cdf.png')


# ============================== 3) Lines / degradation (slide 9) ==============================
fig = plt.figure(figsize=(16, 8.5))
ax  = fig.add_axes([0.07, 0.13, 0.90, 0.70])

xs = list(range(1, len(lead_cols) + 1))

# individual basin trajectories (gray, faint)
for _, row in df.iterrows():
    ax.plot(xs, [row[c] for c in lead_cols], color='gray',
            alpha=0.30, linewidth=0.6)

means   = [df[c].mean()         for c in lead_cols]
medians = [df[c].median()       for c in lead_cols]
p25     = [df[c].quantile(0.25) for c in lead_cols]
p75     = [df[c].quantile(0.75) for c in lead_cols]

ax.fill_between(xs, p25, p75, alpha=0.32, color='steelblue',
                label='25th–75th percentile')
ax.plot(xs, means,   'o-',  color='red',  linewidth=2.4, markersize=9,
        label='Mean',   zorder=5)
ax.plot(xs, medians, 's--', color='blue', linewidth=2.4, markersize=9,
        label='Median', zorder=5)

ax.set_xlabel('Forecast Lead Time (hours)')
ax.set_ylabel('NSE')
ax.set_xticks(xs)
ax.set_xticklabels(lead_labels)
ax.set_ylim(Y_LO, Y_HI)
ax.grid(True, alpha=0.3, linewidth=0.4)
ax.legend(loc='lower left')
if _crowded:
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# annotate first + last lead so the absolute numbers are clear.
# Leftmost: Med value (0.996) is right at the y-ceiling, so the label has
# to sit *beside* the marker, not above it.  Rightmost: there's plenty of
# headroom above the markers, keep the labels at top.
for i in (0, n_leads - 1):
    md, mn = medians[i], means[i]
    span = Y_HI - Y_LO
    if i == 0:
        # Med — above the next two markers (sits in the gap to Y_HI)
        ax.text(i + 1 + 0.30, md + 0.012, f'Med: {md:.3f}',
                ha='left', va='bottom',
                fontsize=10, color='blue', fontweight='bold')
        # Mean — between mean and median lines, well below the median markers
        ax.text(i + 1 + 0.30, mn - 0.025 * span, f'Mean: {mn:.3f}',
                ha='left', va='top',
                fontsize=10, color='red', fontweight='bold')
    else:
        ax.text(i + 1, Y_HI - 0.04 * span, f'Med: {md:.3f}',
                ha='center', va='top',
                fontsize=10, color='blue', fontweight='bold')
        ax.text(i + 1 - 0.25, mn - 0.04 * span, f'Mean: {mn:.3f}',
                ha='right', va='top',
                fontsize=10, color='red', fontweight='bold')

ss.add_title_banner(
    fig, f'NSE Degradation with Forecast Lead Time  (n = {n} TRB basins)',
    fontsize=20)
plt.savefig(os.path.join(OUT_DIR, 'forecast_lines.png'), dpi=200)
plt.close()
print('Saved forecast_lines.png')
