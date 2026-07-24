"""
Background slide figure — Global training set vs Tennessee evaluation set.

Two-panel map:
  Left:  CONUS, all 618 CAMELS-H basins (training set) as light-grey dots,
         130 TN basins highlighted, a green rectangle outlines TN.
  Right: Zoom into the rectangle, 130 TN basins coloured by drainage area.

Output: output/presentation/global_vs_tn_map.png
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
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import slide_style as ss


GLOBAL_PARQUET = '../camelsh_global.parquet'
TN_PARQUET     = '../camelsh_tennessee.parquet'
OUT_PNG        = 'Presentation/presentation/global_vs_tn_map.png'

ss.apply_rc()
plt.rcParams.update({'figure.dpi': 150, 'font.size': 11})


# ---------- load ----------
g = (pd.read_parquet(GLOBAL_PARQUET,
                     columns=['basin_id', 'latitude', 'longitude', 'area_sqkm'])
       .groupby('basin_id').first())
t = (pd.read_parquet(TN_PARQUET,
                     columns=['basin_id', 'latitude', 'longitude', 'area_sqkm'])
       .groupby('basin_id').first())
print(f'Global basins: {len(g)}    TN basins: {len(t)}')
g['is_tn'] = g.index.isin(t.index)

# TN bounding box (with a small pad)
lon_pad, lat_pad = 0.6, 0.4
tn_xmin = t['longitude'].min() - lon_pad
tn_xmax = t['longitude'].max() + lon_pad
tn_ymin = t['latitude'].min()  - lat_pad
tn_ymax = t['latitude'].max()  + lat_pad


# ---------- figure ----------
FIG_W, FIG_H = 20, 10
fig = plt.figure(figsize=(FIG_W, FIG_H))

proj = ccrs.PlateCarree()

# Left panel: CONUS map
ax1 = fig.add_axes([0.035, 0.10, 0.55, 0.74], projection=proj)
ax1.set_extent([-125, -66, 24, 50], crs=proj)
ax1.add_feature(cfeature.LAND.with_scale('50m'),   facecolor='#f3f1e9', zorder=0)
ax1.add_feature(cfeature.OCEAN.with_scale('50m'),  facecolor='#dbeef5', zorder=0)
ax1.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.5, zorder=1)
ax1.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='#444444', linewidth=0.7, zorder=1)
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor='#444444', linewidth=0.6, zorder=1)

# Non-TN basins (training pool)
non_tn = g[~g['is_tn']]
ax1.scatter(non_tn['longitude'], non_tn['latitude'], s=22, c='#7d848a',
            edgecolor='#3b3f43', linewidth=0.25, alpha=0.8,
            transform=proj, zorder=2,
            label=f'Training pool — {len(non_tn)} non-TN basins')

# TN basins (evaluation focus) — green, larger than the training pool
ax1.scatter(t['longitude'], t['latitude'], s=32, c=ss.COLORS['banner_bot'],
            edgecolor='#2a4d18', linewidth=0.5, alpha=0.95, transform=proj, zorder=3,
            label=f'Evaluation focus — {len(t)} Tennessee River Basin gauges')

# Rectangle around TN
rect = Rectangle((tn_xmin, tn_ymin), tn_xmax - tn_xmin, tn_ymax - tn_ymin,
                 fill=False, edgecolor=ss.COLORS['banner_edge'], linewidth=2.0,
                 zorder=4, transform=proj)
ax1.add_patch(rect)

ax1.set_title(f'618 CAMELS-H Basins — Jointly Trained',
              fontsize=14, fontweight='bold', pad=6)
gl = ax1.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.4)
gl.top_labels = gl.right_labels = False
gl.xlabel_style = {'size': 9}
gl.ylabel_style = {'size': 9}

ax1.legend(loc='lower left', fontsize=10, framealpha=0.92)


# Right panel: TN zoom
ax2 = fig.add_axes([0.605, 0.10, 0.36, 0.74], projection=proj)
ax2.set_extent([tn_xmin, tn_xmax, tn_ymin, tn_ymax], crs=proj)
ax2.add_feature(cfeature.LAND.with_scale('10m'),   facecolor='#f3f1e9', zorder=0)
ax2.add_feature(cfeature.OCEAN.with_scale('10m'),  facecolor='#dbeef5', zorder=0)
ax2.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='gray', linewidth=0.7, zorder=1)
ax2.add_feature(cfeature.RIVERS.with_scale('10m'), edgecolor='#4a90d9',
                linewidth=0.6, alpha=0.7, zorder=1)
ax2.add_feature(cfeature.LAKES.with_scale('10m'),  facecolor='#dbeef5',
                edgecolor='#4a90d9', linewidth=0.4, zorder=1)

# TN basins sized by area, coloured uniformly green
sizes = 30 + 220 * np.sqrt(t['area_sqkm'] / t['area_sqkm'].max())
ax2.scatter(t['longitude'], t['latitude'], s=sizes, c=ss.COLORS['banner_bot'],
            edgecolor='#2a4d18', linewidth=0.5, alpha=0.9, transform=proj, zorder=3)

# Size-legend (3 reference sizes)
for a, lbl in [(100, '100 km²'), (1000, '1,000 km²'), (5000, '5,000 km²')]:
    ax2.scatter([], [], s=30 + 220 * np.sqrt(a / t['area_sqkm'].max()),
                c='lightgray', edgecolor='black', linewidth=0.4, label=lbl)
ax2.legend(loc='lower left', fontsize=9, title='Drainage area',
           labelspacing=1.0, borderpad=0.8, framealpha=0.92)

ax2.set_title(f'Tennessee River Basin — {len(t)} Evaluation Gauges',
              fontsize=14, fontweight='bold', pad=6)
gl = ax2.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.4)
gl.top_labels = gl.right_labels = False
gl.xlabel_style = {'size': 9}
gl.ylabel_style = {'size': 9}


# ---------- Connecting lines: rectangle corners → zoom panel corners ----------
# top-right of TN rect → top-left of right panel; bottom-right → bottom-left
for src_xy, dst_axes_xy in [
    ((tn_xmax, tn_ymax), (0.0, 1.0)),     # TN top-right → ax2 top-left
    ((tn_xmax, tn_ymin), (0.0, 0.0)),     # TN bot-right → ax2 bot-left
]:
    cp = ConnectionPatch(
        xyA=src_xy, coordsA=ax1.transData,
        xyB=dst_axes_xy, coordsB=ax2.transAxes,
        arrowstyle='-', color=ss.COLORS['banner_edge'],
        linewidth=1.2, linestyle='--', alpha=0.8, zorder=20,
    )
    fig.add_artist(cp)


# ---------- Deck chrome ----------
ss.add_title_banner(
    fig,
    'Training Setup — Global FutureTST (1 Model · 618 CONUS Basins · Regional Focus on TRB)',
    fontsize=18)

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
plt.savefig(OUT_PNG, dpi=200)
plt.close()
print(f'Saved: {OUT_PNG}')
