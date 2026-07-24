"""
Observation-data-quality vs prediction-quality comparison.

Target variable: Q_camelsh_obs_norm  (the model's Y).
Per basin we measure two kinds of "data quality":
  - train coverage : fraction of training hours (1997-2018) with observed Q
                     -> how well the model could LEARN that basin
  - test  coverage : fraction of test hours (2019-2022) with observed Q
                     -> how reliably NSE can be MEASURED for that basin
NSE depends on both, so the two must be looked at together.

Outputs (to output/presentation/):
  1. train_coverage_map.png
  2. test_coverage_map.png
  3. coverage_vs_nse.png   — 3 maps (train cov | test cov | NSE) + 2 scatters
  + tn_target_coverage.csv
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
from matplotlib.patches import FancyBboxPatch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import slide_style as ss


TN_PARQUET  = '../camelsh_tennessee.parquet'
METRICS_CSV = 'Presentation/tennessee_18lead/tn_metrics_long.csv'
OUT_DIR     = 'Presentation/presentation'
TRAIN_START = np.datetime64('1997-01-01')
TEST_START  = np.datetime64('2019-01-01')
EXCLUDED    = {'03566420'}          # data-quality outlier, not a coverage problem

os.makedirs(OUT_DIR, exist_ok=True)
ss.apply_rc()
plt.rcParams.update({'figure.dpi': 150})


# ---------------------------------------------------------------
# Data: per-basin geo, train/test target coverage, mean NSE
# ---------------------------------------------------------------
geo = (pd.read_parquet(TN_PARQUET,
                       columns=['basin_id', 'latitude', 'longitude', 'area_sqkm'])
       .groupby('basin_id').first())

obs = pd.read_parquet(TN_PARQUET, columns=['Time', 'basin_id', 'Q_camelsh_obs_norm'])
obs['Time'] = pd.to_datetime(obs['Time'])
t = obs['Time'].values
notna = obs['Q_camelsh_obs_norm'].notna().values
bid = obs['basin_id'].values


def coverage(mask):
    s = pd.Series(notna[mask], index=bid[mask])
    return s.groupby(level=0).mean()


geo['train_cov'] = coverage((t >= TRAIN_START) & (t < TEST_START))
geo['test_cov']  = coverage(t >= TEST_START)

met = pd.read_csv(METRICS_CSV, dtype={'basin': str})
geo['mean_nse'] = geo.index.map(met.groupby('basin')['nse'].mean())

print(f'basins: {len(geo)}')
print(f'  train coverage:  med {geo["train_cov"].median():.3f}  '
      f'min {geo["train_cov"].min():.3f}')
print(f'  test  coverage:  med {geo["test_cov"].median():.3f}  '
      f'min {geo["test_cov"].min():.3f}')

geo.to_csv(os.path.join(OUT_DIR, 'tn_target_coverage.csv'))


# ---------------------------------------------------------------
# Shared cartopy basemap
# ---------------------------------------------------------------
PROJ = ccrs.PlateCarree()


def draw_basemap(ax):
    ax.set_extent([geo['longitude'].min() - 0.6, geo['longitude'].max() + 0.6,
                   geo['latitude'].min() - 0.4, geo['latitude'].max() + 0.4],
                  crs=PROJ)
    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='#f3f1e9', zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#dbeef5', zorder=0)
    ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='gray',
                   linewidth=0.7, zorder=1)
    ax.add_feature(cfeature.RIVERS.with_scale('10m'), edgecolor='#4a90d9',
                   linewidth=0.6, alpha=0.7, zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale('10m'), facecolor='#dbeef5',
                   edgecolor='#4a90d9', linewidth=0.4, zorder=1)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.4)
    gl.top_labels = gl.right_labels = False
    return gl


SIZES = 25 + 220 * np.sqrt(geo['area_sqkm'] / geo['area_sqkm'].max())


def plot_metric(ax, values, cmap, vmin, vmax, na_label):
    """Scatter basins coloured by `values`; NaN basins drawn as black x.
    Returns the mappable; the caller adds a height-matched colorbar."""
    has = values.notna()
    sc = ax.scatter(geo.loc[has, 'longitude'], geo.loc[has, 'latitude'],
                    s=SIZES[has], c=values[has], cmap=cmap, vmin=vmin, vmax=vmax,
                    edgecolor='black', linewidth=0.5, alpha=0.9,
                    transform=PROJ, zorder=3)
    if (~has).any():
        ax.scatter(geo.loc[~has, 'longitude'], geo.loc[~has, 'latitude'],
                   s=40, marker='x', color='black', linewidth=1.2,
                   label=na_label, transform=PROJ, zorder=4)
        ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    return sc


def add_matched_colorbar(fig, ax, sc, label, fontsize=11, width=0.020, gap=0.010):
    """Add a colorbar whose height matches the *drawn* map, not the axes box.

    A cartopy GeoAxes shrinks inside its slot to keep its lon/lat aspect, so a
    normal colorbar (sized to the axes box) ends up taller than the map. We
    force a draw, read the settled map position, and size the colorbar to it.
    """
    fig.canvas.draw()
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + gap, pos.y0, width, pos.height])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(label, fontsize=fontsize)
    return cbar


# ---------------------------------------------------------------
# 1-3) Standalone maps — deck-styled (green title banner)
# ---------------------------------------------------------------
COV_LABEL = 'Target coverage — fraction of hours with observed Q'
for col, fname, title, cbar_label, vlo, vhi, na in [
    ('train_cov', 'train_coverage_map.png',
     'Training-Period Target Coverage (1997–2018) — Tennessee River Basin',
     COV_LABEL, 0.0, 1.0, 'no basin geometry'),
    ('test_cov', 'test_coverage_map.png',
     'Test-Period Target Coverage (2019–2022) — Tennessee River Basin',
     COV_LABEL, 0.0, 1.0, 'no basin geometry'),
    ('mean_nse', 'nse_map.png',
     'Prediction Quality — Mean NSE (1–18 h leads), Tennessee River Basin',
     'Mean NSE (1–18 h leads)', 0.5, 1.0, 'no test data'),
]:
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_axes([0.055, 0.07, 0.84, 0.74], projection=PROJ)
    draw_basemap(ax)
    sc = plot_metric(ax, geo[col], 'RdYlGn', vlo, vhi, na)
    add_matched_colorbar(fig, ax, sc, cbar_label, fontsize=12, width=0.024)
    ss.add_title_banner(fig, title, fontsize=16)
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=200)
    plt.close()
    print(f'Saved {fname}')


# ---------------------------------------------------------------
# 3) Combined: 3 maps + 2 scatters + takeaway panel
#
# Manual placement (no gridspec): every panel gets the same column
# width and the same colorbar treatment, and the top row height is
# set to exactly the map's lon/lat aspect so the maps fill their
# slots instead of floating with whitespace above/below.
# ---------------------------------------------------------------
FIG_W, FIG_H = 22, 11
fig = plt.figure(figsize=(FIG_W, FIG_H))

# --- layout geometry (figure fractions) ---
L, R          = 0.045, 0.995      # outer left / right
TOP, BOT      = 0.865, 0.075      # usable band below title banner / above bottom
COL_GAP       = 0.052             # gap between columns
CBAR_GAP      = 0.007             # plot -> colorbar gap
CBAR_W        = 0.011             # colorbar width
TITLE_PAD     = 0.052             # space above each plot for its title
ROW_GAP       = 0.050             # gap between the two rows

# every column is followed by a COL_GAP — including the last one, so the
# third colorbar's tick labels + axis label have room and don't run off-figure
W_col  = (R - L) / 3 - COL_GAP
PLOT_W = W_col - CBAR_GAP - CBAR_W
COL_X  = [L + i * (W_col + COL_GAP) for i in range(3)]

# The maps carry the most information, so the top row is given MORE height
# than the bottom row. TN is a wide, short basin, so a true-aspect map can
# never be tall — the maps are stretched vertically (set_aspect('auto') in
# map_axes) to fill the taller slot, trading geographic aspect for legibility.
MAP_H  = 0.37
SCAT_H = (TOP - BOT) - (2 * TITLE_PAD + MAP_H + ROW_GAP)
TOP_Y0 = TOP - TITLE_PAD - MAP_H
BOT_Y0 = BOT


def map_axes(col):
    ax = fig.add_axes([COL_X[col], TOP_Y0, PLOT_W, MAP_H], projection=PROJ)
    return ax


def scatter_axes(col):
    return fig.add_axes([COL_X[col], BOT_Y0, PLOT_W, SCAT_H])


def col_cbar(col, y0, h, sc, label):
    cax = fig.add_axes([COL_X[col] + PLOT_W + CBAR_GAP, y0, CBAR_W, h])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(label, fontsize=10)
    return cbar


# ----- top row: three maps (stretched to fill the taller slot) -----
ax1 = map_axes(0)
draw_basemap(ax1)
ax1.set_aspect('auto')
sc1 = plot_metric(ax1, geo['train_cov'], 'RdYlGn', 0.0, 1.0, 'no geometry')
ax1.set_title('(a) Training data quality\nTarget coverage 1997–2018',
              fontsize=12, fontweight='bold')
col_cbar(0, TOP_Y0, MAP_H, sc1, 'Train coverage')

ax2 = map_axes(1)
draw_basemap(ax2)
ax2.set_aspect('auto')
sc2 = plot_metric(ax2, geo['test_cov'], 'RdYlGn', 0.0, 1.0, 'no geometry')
ax2.set_title('(b) Test data quality\nTarget coverage 2019–2022',
              fontsize=12, fontweight='bold')
col_cbar(1, TOP_Y0, MAP_H, sc2, 'Test coverage')

ax3 = map_axes(2)
draw_basemap(ax3)
ax3.set_aspect('auto')
sc3 = plot_metric(ax3, geo['mean_nse'], 'RdYlGn', 0.5, 1.0, 'no test data')
ax3.set_title('(c) Prediction quality\nMean NSE',
              fontsize=12, fontweight='bold')
col_cbar(2, TOP_Y0, MAP_H, sc3, 'Mean NSE (1–18 h leads)')

# ----- bottom row: two scatters + takeaway panel -----
j = geo[['train_cov', 'test_cov', 'mean_nse']].dropna().copy()
j = j[~j.index.isin(EXCLUDED)]          # drop data-quality outlier (NSE ~ -46)

ax4 = scatter_axes(0)
sc4 = ax4.scatter(j['train_cov'], j['mean_nse'], s=55, c=j['test_cov'],
                  cmap='viridis', vmin=0, vmax=1, edgecolor='black',
                  linewidth=0.5, alpha=0.9, zorder=3)
r4 = j['train_cov'].corr(j['mean_nse'])
col_cbar(0, BOT_Y0, SCAT_H, sc4, 'Test coverage')
ax4.set_xlabel('Training-period target coverage', fontsize=11)
ax4.set_ylabel('Mean NSE (1–18 h leads)', fontsize=11)
ax4.set_title(f'(d) NSE vs TRAINING coverage   (r = {r4:.2f})',
              fontsize=12, fontweight='bold')

ax5 = scatter_axes(1)
sc5 = ax5.scatter(j['test_cov'], j['mean_nse'], s=55, c=j['train_cov'],
                  cmap='viridis', vmin=0, vmax=1, edgecolor='black',
                  linewidth=0.5, alpha=0.9, zorder=3)
r5 = j['test_cov'].corr(j['mean_nse'])
col_cbar(1, BOT_Y0, SCAT_H, sc5, 'Train coverage')
ax5.set_xlabel('Test-period target coverage', fontsize=11)
ax5.set_ylabel('Mean NSE (1–18 h leads)', fontsize=11)
ax5.set_title(f'(e) NSE vs TEST coverage   (r = {r5:.2f})',
              fontsize=12, fontweight='bold')

for ax in (ax4, ax5):
    ax.axhline(0.7, color='gray', ls=':', lw=1.2)
    ax.set_ylim(0.55, 1.02)
    ax.set_xlim(-0.03, 1.03)
    ax.grid(True, alpha=0.3, linewidth=0.4)

# takeaway panel — deck-style sage-green callout spanning the third column
ax6 = fig.add_axes([COL_X[2], BOT_Y0, W_col, SCAT_H])
ax6.axis('off')
fw, fh = fig.get_size_inches()
# ~0.20-inch corner radius on this ~5.8 x 3.5-inch callout
_callout_radius = 0.04
ax6.add_patch(FancyBboxPatch(
    (0.012, 0.012), 0.976, 0.976, transform=ax6.transAxes,
    boxstyle=f'round,pad=0,rounding_size={_callout_radius}',
    mutation_aspect=(W_col * fw) / (SCAT_H * fh),
    facecolor=ss.COLORS['callout'], edgecolor=ss.COLORS['callout_edge'],
    linewidth=1.6, zorder=0))
txt = (
    '►  Reading the comparison\n'
    f'    (n = {len(j)} basins, 03566420 excluded)\n\n'
    f'•  NSE vs TRAIN coverage:  r = {r4:.2f}\n'
    '   Real positive link — basins the model could not\n'
    '   learn (sparse training Q) tend to score worse.\n\n'
    f'•  NSE vs TEST coverage:  r = {r5:.2f}\n'
    '   Almost none — once a basin has test data, its\n'
    '   amount barely moves NSE (it just adds noise).\n\n'
    '•  Well-covered basins can still score mediocre\n'
    '   (e.g. 03597590) — coverage is a risk factor,\n'
    '   not the whole story.'
)
ax6.text(0.07, 0.5, txt, transform=ax6.transAxes, va='center', ha='left',
         fontsize=11.5, linespacing=1.45, color='#1a1a1a', zorder=1)

ss.add_title_banner(
    fig, 'Target Data Quality (train & test) vs Prediction Quality — TN Basins',
    fontsize=21)

plt.savefig(os.path.join(OUT_DIR, 'coverage_vs_nse.png'), dpi=200)
plt.close()
print('Saved coverage_vs_nse.png')

print(f'\nDone. Figures + tn_target_coverage.csv in {OUT_DIR}/')
