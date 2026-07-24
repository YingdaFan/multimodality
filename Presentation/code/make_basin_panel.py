"""
Multi-basin time-series panel inspired by InflowForecast.pdf page 7.

Layout:
  - Central TN map highlighting 8 selected basins
  - 8 surrounding time-series panels (obs vs 1h-ahead pred, calendar year 2019)
  - Connecting lines from each panel to its basin on the map
  - Each panel shows basin ID + area in title and NSE/KGE/RMSE box

Outputs:
  output/presentation/basin_panel.png
"""

# --- pin CWD to imputation/ so relative paths resolve no matter where the
# --- script is launched from (the file now lives in Presentation/code/)
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])


import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import ConnectionPatch

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import slide_style as ss


# ============================== Config ==============================
PRED_DIR    = 'diffusion_forecast/output/pred'
PREPPED     = 'data_processing/data/prepped.npz'
TN_PARQUET  = '../camelsh_tennessee.parquet'
METRICS_CSV = 'Presentation/tennessee_18lead/tn_metrics_long.csv'
COVERAGE_CSV = 'Presentation/presentation/tn_target_coverage.csv'
OUT_PNG     = 'Presentation/presentation/basin_panel.png'

LEAD_STEP   = 17                         # 17 = 18h-ahead (steps 0..17 → 1..18 h)
LEAD_HOUR   = LEAD_STEP + 1
MIN_TEST_COV = 0.9                       # only basins with ≥90% target coverage in test
PLOT_START  = np.datetime64('2019-01-01')
PLOT_END    = np.datetime64('2020-01-01') # exclusive — full calendar 2019
EXCLUDED    = {'03566420'}


# ============================== Load metrics + select basins ==============================
ss.apply_rc()

met = pd.read_csv(METRICS_CSV, dtype={'basin': str})
mlead = met[met['hours_ahead'] == LEAD_HOUR].dropna(subset=['nse']).copy()
geo = (pd.read_parquet(TN_PARQUET,
                       columns=['basin_id', 'latitude', 'longitude', 'area_sqkm'])
       .groupby('basin_id').first())
cov = pd.read_csv(COVERAGE_CSV, dtype={'basin_id': str})[
    ['basin_id', 'test_cov']].rename(columns={'basin_id': 'basin'})

mlead = (mlead.join(geo, on='basin', how='inner')
              .merge(cov, on='basin', how='inner'))
mlead = mlead[~mlead['basin'].isin(EXCLUDED)].reset_index(drop=True)

# Restrict to basins with ample observed-Q in the test period so the
# obs vs 18h-ahead comparison is data-rich, not gappy
mlead = mlead[mlead['test_cov'] >= MIN_TEST_COV].reset_index(drop=True)
print(f'Basins with test_cov ≥ {MIN_TEST_COV}: {len(mlead)}')

# 8 basins, geometrically spaced in drainage area; take the median-NSE
# basin in each bin (high-coverage already → curves stay informative)
mlead = mlead.sort_values('area_sqkm').reset_index(drop=True)
mlead['log_area'] = np.log10(mlead['area_sqkm'])
edges = np.linspace(mlead['log_area'].min(), mlead['log_area'].max(), 9)
mlead['bin'] = pd.cut(mlead['log_area'], edges, include_lowest=True)

picks = []
for _, sub in mlead.groupby('bin', observed=True):
    if len(sub) == 0:
        continue
    sub_sorted = sub.sort_values('nse').reset_index(drop=True)
    picks.append(sub_sorted.iloc[len(sub_sorted) // 2])    # median-NSE pick
picks = pd.DataFrame(picks).reset_index(drop=True)
print('Selected basins:')
print(picks[['basin', 'area_sqkm', 'test_cov', 'nse', 'kge', 'rmse',
             'latitude', 'longitude']].to_string(index=False))

selected_ids = picks['basin'].tolist()


# ============================== Load predictions + observations ==============================
with open(os.path.join(PRED_DIR, 'meta.json')) as f:
    meta = json.load(f)
window   = int(meta['window'])
pred_len = int(meta['pred_len'])
stride   = int(meta['stride'])
print(f'meta: window={window}, pred_len={pred_len}, stride={stride}')

d = np.load(PREPPED, allow_pickle=True)
y_raw       = d['y_raw_tst']
times       = d['times_tst']
y_mean      = d['y_mean']
y_std       = d['y_std']
basin_names = [str(b) for b in d['basin_names']]
n_basins    = len(basin_names)
n_times     = y_raw.shape[1]
name_to_idx = {b: i for i, b in enumerate(basin_names)}

max_offsets = n_times - window - pred_len + 1
n_offsets   = (max_offsets - 1) // stride + 1
y_pred_all  = np.load(os.path.join(PRED_DIR, 'tst.npy'))
assert y_pred_all.shape[0] == n_basins * n_offsets, 'sample count mismatch'

y_pred_step     = y_pred_all[:, LEAD_STEP, 0]
y_pred_by_basin = y_pred_step.reshape(n_offsets, n_basins).T   # (n_basins, n_offsets)
gt_indices      = np.array([k * stride + window + LEAD_STEP for k in range(n_offsets)])
t_pred          = times[gt_indices]
y_gt_by_basin   = y_raw[:, gt_indices, 0]

# Slice all curves to the calendar-2019 window
obs_mask  = (times  >= PLOT_START) & (times  < PLOT_END)
pred_mask = (t_pred >= PLOT_START) & (t_pred < PLOT_END)
t_obs_yr  = pd.to_datetime(times[obs_mask])
t_pred_yr = pd.to_datetime(t_pred[pred_mask])
print(f'2019 obs points: {obs_mask.sum()}, 2019 pred points: {pred_mask.sum()}')


# ============================== Figure layout ==============================
plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
})

FIG_W, FIG_H = 22, 12.5
fig = plt.figure(figsize=(FIG_W, FIG_H))

# 3x3 grid of cells, manually placed in figure coords to leave room for banner
BANNER_H = 0.085
TOP      = 1 - BANNER_H - 0.025      # top of the grid area
BOTTOM   = 0.06                       # leave room for page badge
LEFT     = 0.04
RIGHT    = 0.97
GW       = RIGHT - LEFT
GH       = TOP - BOTTOM

CELL_W = GW / 3
CELL_H = GH / 3
PAD_X  = 0.012
PAD_Y  = 0.020

def cell_rect(col, row):
    """row=0 top, row=2 bottom; returns [x0, y0, w, h] in figure fractions."""
    x0 = LEFT + col * CELL_W + PAD_X
    y0 = TOP - (row + 1) * CELL_H + PAD_Y
    return [x0, y0, CELL_W - 2 * PAD_X, CELL_H - 2 * PAD_Y]


# ============================== Centre map ==============================
proj = ccrs.PlateCarree()
ax_map = fig.add_axes(cell_rect(1, 1), projection=proj)

lon_pad, lat_pad = 0.6, 0.4
ax_map.set_extent([geo['longitude'].min() - lon_pad, geo['longitude'].max() + lon_pad,
                   geo['latitude'].min()  - lat_pad, geo['latitude'].max()  + lat_pad],
                  crs=proj)
ax_map.add_feature(cfeature.LAND.with_scale('10m'),  facecolor='#f3f1e9', zorder=0)
ax_map.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='#dbeef5', zorder=0)
ax_map.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='gray', linewidth=0.6, zorder=1)
ax_map.add_feature(cfeature.RIVERS.with_scale('10m'), edgecolor='#4a90d9',
                   linewidth=0.5, alpha=0.7, zorder=1)
ax_map.add_feature(cfeature.LAKES.with_scale('10m'),  facecolor='#dbeef5',
                   edgecolor='#4a90d9', linewidth=0.4, zorder=1)

# All gauges in light grey
ax_map.scatter(geo['longitude'], geo['latitude'], s=14, c='#999999',
               edgecolor='none', alpha=0.55, transform=proj, zorder=2)

# Highlight selected basins, colour-coded by NSE
sel_lon = picks['longitude'].values
sel_lat = picks['latitude'].values
sel_nse = picks['nse'].values
sc = ax_map.scatter(sel_lon, sel_lat, s=140, c=sel_nse, cmap='RdYlGn',
                    vmin=0.7, vmax=1.0, edgecolor='black', linewidth=1.0,
                    transform=proj, zorder=4)

# Basin-id labels on the map
for _, r in picks.iterrows():
    ax_map.annotate(r['basin'], xy=(r['longitude'], r['latitude']),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=7.5, fontweight='bold', color='#222222',
                    transform=proj, zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.7))

ax_map.set_title('Tennessee River Basin', fontsize=12, fontweight='bold', pad=4)
gl = ax_map.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.35)
gl.top_labels = gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}


# ============================== 8 surrounding TS panels ==============================
# Cell positions arranged clockwise from top-left
panel_cells = [
    (0, 0), (1, 0), (2, 0),     # top row
    (2, 1),                      # right middle
    (2, 2), (1, 2), (0, 2),     # bottom row (right→left)
    (0, 1),                      # left middle
]

# Sort picks by latitude→longitude so adjacent panels point to nearby map dots,
# minimising line crossings. Top row gets northern basins, bottom row southern.
picks_assign = picks.copy()
picks_assign['col'] = [c for c, r in panel_cells]
picks_assign['row'] = [r for c, r in panel_cells]

# Order picks: top row left→right by longitude (after sorting by latitude desc),
# bottom row left→right, sides north-then-south
sort_geo = picks.copy()
sort_geo = sort_geo.sort_values('latitude', ascending=False).reset_index(drop=True)
top3  = sort_geo.iloc[:3].sort_values('longitude').reset_index(drop=True)
bot3  = sort_geo.iloc[-3:].sort_values('longitude', ascending=False).reset_index(drop=True)
mid2  = sort_geo.iloc[3:5].sort_values('latitude', ascending=False).reset_index(drop=True)
# mid2[0] is more-northern: goes to right-middle if it's east of map centre, else left
map_lon_mid = (geo['longitude'].min() + geo['longitude'].max()) / 2
if mid2.iloc[0]['longitude'] >= map_lon_mid:
    right_mid, left_mid = mid2.iloc[0], mid2.iloc[1]
else:
    left_mid, right_mid = mid2.iloc[0], mid2.iloc[1]

ordered = [
    top3.iloc[0], top3.iloc[1], top3.iloc[2],   # cells (0,0) (1,0) (2,0)
    right_mid,                                   # (2,1)
    bot3.iloc[0], bot3.iloc[1], bot3.iloc[2],   # (2,2) (1,2) (0,2)
    left_mid,                                    # (0,1)
]

basin_color = dict(zip(picks['basin'], plt.cm.RdYlGn((picks['nse'] - 0.5) / 0.5)))

panel_axes = {}
for cell, row in zip(panel_cells, ordered):
    ax = fig.add_axes(cell_rect(cell[0], cell[1]))
    b = row['basin']
    i = name_to_idx[b]

    y_obs  = y_raw[i, obs_mask, 0]
    y_pred = y_pred_by_basin[i, pred_mask] * y_std[i] + y_mean[i]
    y_pred = np.maximum(y_pred, 0)

    ax.plot(t_obs_yr,  y_obs,  '-', color='black',    linewidth=0.8, label='Obs')
    ax.plot(t_pred_yr, y_pred, '--', color='#d62728', linewidth=0.8, alpha=0.9,
            label=f'Pred ({LEAD_HOUR}h)')

    ax.set_title(f"{b}  (A = {row['area_sqkm']:.0f} km²)",
                 fontsize=10.5, fontweight='bold', pad=3)
    ax.set_ylabel('Q [mm/day]', fontsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.tick_params(axis='x', pad=1)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.25, linewidth=0.3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # metric box (upper-right)
    txt = f"NSE={row['nse']:.2f}\nKGE={row['kge']:.2f}\nRMSE={row['rmse']:.1f}"
    ax.text(0.985, 0.97, txt, transform=ax.transAxes, ha='right', va='top',
            fontsize=8, family='monospace',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='#999999', linewidth=0.6))

    panel_axes[b] = (ax, cell)


# Single legend in the lower-left corner of the map cell
ax_map.plot([], [], '-',  color='black',    linewidth=1.2, label='Obs')
ax_map.plot([], [], '--', color='#d62728', linewidth=1.2, label=f'Pred ({LEAD_HOUR}h)')
ax_map.legend(loc='lower left', fontsize=9, framealpha=0.85)


# ============================== Connecting lines (panel → map dot) ==============================
# For each panel, pick the panel edge nearest the map and connect to (lon, lat)
def panel_anchor(col, row):
    """Return axes-coord anchor on the panel side that faces the map."""
    if row == 0:                          # top row → bottom edge
        return (0.5, 0.0)
    if row == 2:                          # bottom row → top edge
        return (0.5, 1.0)
    if col == 0:                          # left middle → right edge
        return (1.0, 0.5)
    return (0.0, 0.5)                     # right middle → left edge

for b, (ax, cell) in panel_axes.items():
    r = picks[picks['basin'] == b].iloc[0]
    xy_panel = panel_anchor(*cell)
    cp = ConnectionPatch(
        xyA=xy_panel, coordsA=ax.transAxes,
        xyB=(r['longitude'], r['latitude']), coordsB=ax_map.transData,
        arrowstyle='-', color='#666666', linewidth=0.7, alpha=0.7, zorder=10,
    )
    fig.add_artist(cp)


# ============================== Deck chrome ==============================
ss.add_title_banner(
    fig,
    f'Selected Basins — Observed vs {LEAD_HOUR}-hour-ahead Forecast '
    f'(test_cov ≥ {MIN_TEST_COV:.0%}, calendar 2019)',
    fontsize=19)

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
plt.savefig(OUT_PNG, dpi=200)
plt.close()
print(f'\nSaved: {OUT_PNG}')
