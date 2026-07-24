"""
Generate the 4 'setup' figures for the presentation:
  1. location_map.png        — 130 TN gauges, colored by mean NSE, sized by area
  2. data_split_timeline.png — train / val / test temporal split
  3. sliding_window.png      — 168h history -> 18h forecast, stride 24h
  4. pipeline_flowchart.png  — preprocess -> train -> postprocess -> analysis

Outputs go to output/presentation/.
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
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import slide_style as ss


TN_PARQUET   = '../camelsh_tennessee.parquet'
METRICS_CSV  = 'Presentation/tennessee_18lead/tn_metrics_long.csv'
OUT_DIR      = 'Presentation/presentation'
EXCLUDED     = {'03566420'}

os.makedirs(OUT_DIR, exist_ok=True)
ss.apply_rc()
plt.rcParams.update({'figure.dpi': 150})


# ============================================================
# 1) Location map
# ============================================================
geo = (pd.read_parquet(TN_PARQUET,
                       columns=['basin_id', 'latitude', 'longitude', 'area_sqkm'])
       .groupby('basin_id').first())

met = pd.read_csv(METRICS_CSV, dtype={'basin': str})
mean_nse = met.groupby('basin')['nse'].mean()          # per-basin mean over all leads
geo['mean_nse'] = geo.index.map(mean_nse)

proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(11, 7.5))
ax = fig.add_axes([0.06, 0.07, 0.78, 0.77], projection=proj)

# map extent: pad the gauge bounding box a bit
lon_pad = 0.6
lat_pad = 0.4
ax.set_extent([geo['longitude'].min() - lon_pad, geo['longitude'].max() + lon_pad,
               geo['latitude'].min() - lat_pad, geo['latitude'].max() + lat_pad],
              crs=proj)

# basemap layers
ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#f3f1e9', zorder=0)
ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#dbeef5', zorder=0)
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='gray',
               linewidth=0.7, zorder=1)
ax.add_feature(cfeature.RIVERS.with_scale('10m'), edgecolor='#4a90d9',
               linewidth=0.6, alpha=0.7, zorder=1)
ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='#dbeef5',
               edgecolor='#4a90d9', linewidth=0.4, zorder=1)
ax.add_feature(cfeature.BORDERS.with_scale('10m'), linewidth=0.8, zorder=1)

sizes = 25 + 220 * np.sqrt(geo['area_sqkm'] / geo['area_sqkm'].max())

# basins with a valid metric
has = geo['mean_nse'].notna()
sc = ax.scatter(geo.loc[has, 'longitude'], geo.loc[has, 'latitude'],
                s=sizes[has], c=geo.loc[has, 'mean_nse'], cmap='RdYlGn',
                vmin=0.5, vmax=1.0, edgecolor='black', linewidth=0.5,
                alpha=0.9, transform=proj, zorder=3)
# basins without a valid metric (no test data / excluded)
ax.scatter(geo.loc[~has, 'longitude'], geo.loc[~has, 'latitude'],
           s=40, marker='x', color='black', linewidth=1.2,
           label='no test data / excluded', transform=proj, zorder=4)

cbar = plt.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label('Mean NSE (over 1–18 h leads)', fontsize=12)

gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.4)
gl.top_labels = gl.right_labels = False

# marker-size legend
for a in [100, 1000, 5000]:
    ax.scatter([], [], s=25 + 220 * np.sqrt(a / geo['area_sqkm'].max()),
               c='lightgray', edgecolor='black', linewidth=0.5,
               label=f'{a:,} km²')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='lower left', fontsize=9, title='Drainage area',
          labelspacing=1.0, borderpad=0.8, framealpha=0.9)

ss.add_title_banner(fig, 'Study Area — 130 USGS Gauges in the Tennessee River Basin',
                    fontsize=17)
plt.savefig(os.path.join(OUT_DIR, 'location_map.png'), dpi=200)
plt.close()
print('Saved location_map.png')


# ============================================================
# 2) Data split timeline
# ============================================================
splits = [
    ('Validation', '1995-01-01', '1997-01-01', '#f0ad4e'),
    ('Training',   '1997-01-01', '2019-01-01', '#5bc0de'),
    ('Testing',    '2019-01-01', '2023-01-01', '#5cb85c'),
]

fig = plt.figure(figsize=(13, 3.8))
ax = fig.add_axes([0.06, 0.30, 0.90, 0.50])
for name, s, e, color in splits:
    s_d, e_d = pd.Timestamp(s), pd.Timestamp(e)
    ax.barh(0, (e_d - s_d).days, left=s_d, height=0.5,
            color=color, edgecolor='black', linewidth=0.8, alpha=0.9)
    yrs = (e_d - s_d).days / 365.25
    ax.text(s_d + (e_d - s_d) / 2, 0, f'{name}\n{s[:4]}–{int(e[:4])-1}  ({yrs:.0f} yr)',
            ha='center', va='center', fontsize=11, fontweight='bold')

ax.set_ylim(-0.6, 0.6)
ax.set_yticks([])
ax.set_xlim(pd.Timestamp('1994-06-01'), pd.Timestamp('2023-07-01'))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
for spine in ['left', 'right', 'top']:
    ax.spines[spine].set_visible(False)

ss.add_title_banner(fig, 'Temporal Data Split — CAMELS-H Hourly Streamflow',
                    fontsize=17, height=0.16)
plt.savefig(os.path.join(OUT_DIR, 'data_split_timeline.png'), dpi=200)
plt.close()
print('Saved data_split_timeline.png')


# ============================================================
# 3) Sliding window diagram
# ============================================================
WINDOW, PRED, STRIDE = 168, 18, 24
N_ROWS  = 3
ROW_H   = 0.95
ROW_GAP = 0.55
TOTAL_HOURS = WINDOW + PRED + (N_ROWS - 1) * STRIDE       # = 234

fig = plt.figure(figsize=(15, 3.6))
ax  = fig.add_axes([0.05, 0.04, 0.92, 0.80])

HIST_FILL  = '#5bc0de'        # match forecast_anatomy.png
HIST_INNER = '#0a5e7a'
FCST_FILL  = '#d9534f'
FCST_INNER = '#8b1a1a'

for k in range(N_ROWS):
    y = -k * (ROW_H + ROW_GAP)
    x0 = k * STRIDE
    # history
    ax.add_patch(Rectangle((x0, y), WINDOW, ROW_H, facecolor=HIST_FILL,
                           edgecolor='black', linewidth=0.8, alpha=0.85))
    # forecast
    ax.add_patch(Rectangle((x0 + WINDOW, y), PRED, ROW_H, facecolor=FCST_FILL,
                           edgecolor='black', linewidth=0.8, alpha=0.85))
    # window-label on the left
    ax.text(-6, y + ROW_H / 2, f'window {k + 1}',
            ha='right', va='center', fontsize=10)

    # On the first row only: rich annotations inside the boxes
    if k == 0:
        ax.text(x0 + WINDOW / 2, y + ROW_H * 0.65,
                f'History — {WINDOW} h  (= 7 days)',
                ha='center', va='center', fontsize=11.5, fontweight='bold',
                color=HIST_INNER)
        ax.text(x0 + WINDOW / 2, y + ROW_H * 0.28,
                'past Q  +  11 dynamic met  +  9 rolling-cumul  +  24 static',
                ha='center', va='center', fontsize=9.5, color=HIST_INNER,
                style='italic')
        ax.text(x0 + WINDOW + PRED / 2, y + ROW_H * 0.65,
                f'{PRED} h\nforecast',
                ha='center', va='center', fontsize=10.5, fontweight='bold',
                color='white')
        ax.text(x0 + WINDOW + PRED / 2, y + ROW_H * 0.22,
                'Q only',
                ha='center', va='center', fontsize=8.5, color='white',
                style='italic')

    # Per-row stride arrow between this row and the row above
    if k > 0:
        prev_x = (k - 1) * STRIDE
        y_top  = y + ROW_H + 0.05
        ax.annotate('', xy=(prev_x + STRIDE + 1, y_top),
                    xytext=(prev_x - 1, y_top),
                    arrowprops=dict(arrowstyle='-|>', color='#444', lw=1.3,
                                    mutation_scale=14))
        if k == 1:
            ax.text(prev_x + STRIDE / 2, y_top + 0.18,
                    f'+{STRIDE} h', ha='center', va='bottom',
                    fontsize=9, color='#444', style='italic')

ax.set_xlim(-30, TOTAL_HOURS + 12)
ax.set_ylim(-(N_ROWS - 1) * (ROW_H + ROW_GAP) - 0.10, ROW_H + 0.08)
ax.set_yticks([])
ax.set_xticks([])
for s in ('left', 'right', 'top', 'bottom'):
    ax.spines[s].set_visible(False)

ss.add_title_banner(fig,
                    'Sliding-Window Forecasting — 168 h history → 18 h forecast, stride 24 h',
                    fontsize=16)
plt.savefig(os.path.join(OUT_DIR, 'sliding_window.png'), dpi=200)
plt.close()
print('Saved sliding_window.png')


# ============================================================
# 4) Pipeline flowchart
# ============================================================
fig = plt.figure(figsize=(15, 4.5))
ax = fig.add_axes([0.02, 0.08, 0.96, 0.68])
boxes = [
    ('1. Preprocess',
     'CAMELS-H parquet\n11 met + 24 static vars\nsliding windows, normalize\n→ prepped.npz'),
    ('2. Train  FutureTST',
     '168 h history → 18 h forecast\ntrain 1997–2018\nval 1995–1996'),
    ('3. Postprocess',
     'denormalize predictions\nper-basin metrics\n(NSE/KGE/RMSE/…)'),
    ('4. TN Analysis',
     '104 TN basins\nmulti-lead figures\nboxplot/CDF/heatmap/…'),
]
# deck-style: use the slide-style content-block palette (blue / green / peach / sage)
colors = [ss.COLORS['prediction'], ss.COLORS['forecasting'],
          ss.COLORS['inputs'],     ss.COLORS['callout']]
n = len(boxes)
bw, gap = 3.0, 0.9
for i, ((title, body), col) in enumerate(zip(boxes, colors)):
    x = i * (bw + gap)
    ax.add_patch(FancyBboxPatch((x, 0), bw, 2.0,
                                boxstyle='round,pad=0.08', facecolor=col,
                                edgecolor='#666666', linewidth=1.0, alpha=0.95))
    ax.text(x + bw / 2, 1.65, title, ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(x + bw / 2, 0.85, body, ha='center', va='center', fontsize=8.2)
    if i < n - 1:
        ax.add_patch(FancyArrowPatch((x + bw, 1.0), (x + bw + gap, 1.0),
                                     arrowstyle='-|>', mutation_scale=22,
                                     color='#4e7a2f', lw=1.8))

ax.set_xlim(-0.4, n * (bw + gap) - gap + 0.4)
ax.set_ylim(-0.3, 2.4)
ax.axis('off')

ss.add_title_banner(fig, 'Forecasting Pipeline', fontsize=20, height=0.14)
plt.savefig(os.path.join(OUT_DIR, 'pipeline_flowchart.png'), dpi=200)
plt.close()
print('Saved pipeline_flowchart.png')

print(f'\nAll 4 presentation figures in {OUT_DIR}/')
