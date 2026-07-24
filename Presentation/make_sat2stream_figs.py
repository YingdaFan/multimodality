#!/usr/bin/env python3
"""Figures for the Sat2Stream job-talk-style deck. Palette per dataviz spec."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3e0'
plt.rcParams.update({
    'font.family': 'Liberation Sans', 'text.color': INK, 'font.size': 14,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.dpi': 220, 'savefig.bbox': 'tight',
})
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sat2stream_assets')

# ---------------------------------------------------------------- 1 · map
df = pd.read_parquet('/home/yif47/river-dl/temporal/camelsh_global_daily.parquet',
                     columns=['basin_id', 'latitude', 'longitude'])
pts = df.groupby('basin_id').first()
fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.scatter(pts.longitude, pts.latitude, s=22, c=BLUE, alpha=0.8, linewidths=0)
ax.set_aspect(1.25)
ax.axis('off')
fig.savefig(f'{OUT}/fig_map.png'); plt.close(fig)

# ---------------------------------------------------------------- 2 · cutout
import xarray as xr
ds = xr.open_dataset('/home/yif47/river-dl/temporal/CAMELSH/moisture/CAMELSH/SMAP_01646500.nc')
day = '2020-06-01'
sm = ds.sm_surface.sel(time=day).values
fig, ax = plt.subplots(figsize=(5.6, 4.6))
im = ax.imshow(sm, cmap='Blues', interpolation='nearest')
ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values():
    s_.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label('surface soil moisture (m³/m³)', fontsize=13, color=INK2)
cb.ax.tick_params(labelsize=11, colors=INK2)
cb.outline.set_visible(False)
fig.savefig(f'{OUT}/fig_cutout.png'); plt.close(fig)
ds.close()

# ---------------------------------------------------------------- 3 · ablation dumbbell
configs = [
    ('baseline\n(no satellite)',            0.530, 0.526, False),
    ('+ satellite → prior only',            0.553, 0.472, False),
    ('+ trust features only',               0.472, 0.462, False),
    ('+ satellite + trust → prior only',    0.576, 0.547, False),
    ('+ satellite + trust,\nfull conditioning  (ours)', 0.572, 0.508, True),
]
fig, ax = plt.subplots(figsize=(8.6, 4.3))
ys = np.arange(len(configs))[::-1]
for y, (name, med, mean, win) in zip(ys, configs):
    c = BLUE if win else INK2
    lw = 2.4 if win else 1.6
    ax.plot([mean, med], [y, y], color=c, linewidth=lw, zorder=2, alpha=0.85)
    ax.scatter([med], [y], s=110 if win else 80, c=c, zorder=3)
    ax.scatter([mean], [y], s=110 if win else 80, facecolors='white',
               edgecolors=c, linewidths=2, zorder=3)
    ax.text(med + 0.008, y, f'{med:.3f}', va='center', fontsize=13,
            color=INK if win else INK2, fontweight='bold' if win else 'normal')
ax.set_yticks(ys)
ax.set_yticklabels([c[0] for c in configs], fontsize=14)
ax.set_xlabel('NSE on 28 zero-shot basins', fontsize=14)
ax.set_xlim(0.40, 0.63)
for s_ in ('top', 'right', 'left'):
    ax.spines[s_].set_visible(False)
ax.tick_params(left=False, labelsize=12)
ax.grid(axis='x', color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], marker='o', color='none', markerfacecolor=INK2, markersize=9, label='median'),
    Line2D([], [], marker='o', color='none', markerfacecolor='white',
           markeredgecolor=INK2, markeredgewidth=2, markersize=9, label='mean'),
], frameon=False, fontsize=13, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2)
ax.annotate('mean dragged down by 3 crashed basins', xy=(0.470, ys[1] + 0.06),
            xytext=(0.404, ys[1] + 0.52), fontsize=12.5, color=ORANGE, style='italic',
            arrowprops=dict(arrowstyle='-', color=ORANGE, lw=1.2))
fig.savefig(f'{OUT}/fig_ablation.png'); plt.close(fig)

# ---------------------------------------------------------------- 4 · karst rescue slope
stations = {
    '01619500  (94% karst)': [0.352, -0.457, 0.220],
    '01617000  (31% karst, 4 px)': [0.452, -0.329, 0.376],
    '01616500  (32% karst)': [0.659, 0.195, 0.488],
}
colors = [BLUE, ORANGE, AQUA]
xs = [0, 1, 2]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for (name, vals), c in zip(stations.items(), colors):
    ax.plot(xs, vals, color=c, linewidth=2.4, marker='o', markersize=8, label=name)
    ax.text(2.06, vals[2], f'{vals[2]:.2f}', va='center', fontsize=13, color=c, fontweight='bold')
ax.axhline(0, color=INK2, linewidth=0.8, linestyle=':')
ax.set_xticks(xs)
ax.set_xticklabels(['baseline\n(no satellite)', '+ satellite\n(crash)', '+ satellite\n+ trust features'],
                   fontsize=13.5)
ax.set_ylabel('NSE', fontsize=14)
ax.set_xlim(-0.15, 2.45)
for s_ in ('top', 'right'):
    ax.spines[s_].set_visible(False)
ax.grid(axis='y', color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=12.5, loc='lower left')
fig.savefig(f'{OUT}/fig_rescue.png'); plt.close(fig)

# ---------------------------------------------------------------- 5 · TRB calibration effect
rows = [
    ('baseline\n(no satellite)', 0.666, 0.688, INK2),
    ('satellite prior,\ncalibrator blind to it', 0.684, 0.552, ORANGE),
    ('satellite prior,\ncalibrator sees it  (ours)', 0.678, 0.707, BLUE),
]
fig, ax = plt.subplots(figsize=(8.4, 4.0))
ys = [2, 1, 0]
for y, (name, prior, final, c) in zip(ys, rows):
    ax.annotate('', xy=(final, y), xytext=(prior, y),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=2.6, mutation_scale=22))
    ax.scatter([prior], [y], s=95, facecolors='white', edgecolors=c, linewidths=2, zorder=3)
    ax.scatter([final], [y], s=95, c=c, zorder=3)
    d = final - prior
    ax.text(max(prior, final) + 0.012, y, f'{final:.3f}  ({d:+.3f})',
            va='center', fontsize=13, color=c,
            fontweight='bold' if c == BLUE else 'normal')
ax.set_yticks(ys)
ax.set_yticklabels([r[0] for r in rows], fontsize=13.5)
ax.set_xlabel('median NSE, 101 held-out Tennessee basins', fontsize=14)
ax.set_xlim(0.52, 0.80)
for s_ in ('top', 'right', 'left'):
    ax.spines[s_].set_visible(False)
ax.tick_params(left=False, labelsize=12)
ax.grid(axis='x', color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
ax.legend(handles=[
    Line2D([], [], marker='o', color='none', markerfacecolor='white',
           markeredgecolor=INK2, markeredgewidth=2, markersize=9, label='prior (stage 1)'),
    Line2D([], [], marker='o', color='none', markerfacecolor=INK2, markersize=9,
           label='after calibration (stage 2)'),
], frameon=False, fontsize=12.5, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2)
fig.savefig(f'{OUT}/fig_trb.png'); plt.close(fig)

print('figures written to', OUT)
