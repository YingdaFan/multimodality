"""
Data Availability Atlas — replaces the schematic split-timeline figure.

Two stacked heatmaps share an x-axis (1985-01 .. 2022-12):
  - top:    618 CONUS training basins (rows sorted by record start)
  - bottom: 130 Tennessee basins (rows sorted by record start)

Cell colour = fraction of the month with valid observed Q. Light beige
means no record yet; saturated green means a complete month. Vertical
lines mark the val / train / test partition boundaries used in
preprocess_camelsh_forecast.py.

Output: output/presentation/data_atlas.png
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

import slide_style as ss


CACHE     = 'Presentation/presentation/_availability.npz'
TN_PARQ   = '../camelsh_tennessee.parquet'
OUT_PNG   = 'Presentation/presentation/data_atlas.png'

VAL_START   = pd.Timestamp('1995-01-01')
TRAIN_START = pd.Timestamp('1997-01-01')
TEST_START  = pd.Timestamp('2019-01-01')
TEST_END    = pd.Timestamp('2022-12-31')
PLOT_END    = pd.Timestamp('2022-12-31')

ss.apply_rc()
plt.rcParams.update({'figure.dpi': 150, 'font.size': 11})


# ============================== Load ==============================
z = np.load(CACHE, allow_pickle=True)
A_all  = z['A'].astype(np.float32)             # (618, 480), fraction in [0,1] or NaN
basins = np.array([str(b) for b in z['basins']])
yms    = np.array([str(y) for y in z['yms']])
ym_dt  = pd.to_datetime([y + '-01' for y in yms])

# Trim to <= 2022-12 (test_end), discard the post-2023 months that aren't used
mask_t = ym_dt <= PLOT_END
A_all  = A_all[:, mask_t]
ym_dt  = ym_dt[mask_t]
print(f'After trim: {A_all.shape}')

# TN subset
tn_ids = set(pd.read_parquet(TN_PARQ, columns=['basin_id'])['basin_id'].unique())
is_tn  = np.array([b in tn_ids for b in basins])
A_tn   = A_all[is_tn]
tn_basins = basins[is_tn]
print(f'TN basins matched: {len(tn_basins)}')


# Sort each block by "first month with ≥ 0.5 coverage"
def first_valid_idx(A):
    valid = (np.nan_to_num(A) >= 0.5)
    first = np.argmax(valid, axis=1)
    first[~valid.any(axis=1)] = A.shape[1]      # no data → bottom
    return first

order_all = np.argsort(first_valid_idx(A_all))
order_tn  = np.argsort(first_valid_idx(A_tn))
A_all = A_all[order_all]
A_tn  = A_tn[order_tn]


# ============================== Figure ==============================
FIG_W, FIG_H = 22, 11.5
fig = plt.figure(figsize=(FIG_W, FIG_H))

# colormap: very light beige → deep deck-green
cmap = LinearSegmentedColormap.from_list(
    'avail',
    [(0.00, '#f5f1e6'),     # almost no data
     (0.20, '#dceac4'),
     (0.55, '#aed080'),
     (1.00, '#4e7a2f')],
)
cmap.set_bad('#f0ece0')                          # NaN → very light beige

# x extent for imshow: numeric date range
x_lo = ym_dt[0].to_pydatetime()
x_hi = (ym_dt[-1] + pd.offsets.MonthEnd(1)).to_pydatetime()

# Two rows: top = all 618, bottom = TN 130; shared time axis
LEFT, RIGHT = 0.06, 0.96
TOP, BOT     = 0.84, 0.13
H_TOP        = 0.42         # all-618 panel height
GAP_V        = 0.07
H_BOT        = TOP - BOT - H_TOP - GAP_V - 0.05   # TN panel height (smaller)

ax_all = fig.add_axes([LEFT, TOP - H_TOP,                  RIGHT - LEFT, H_TOP])
ax_tn  = fig.add_axes([LEFT, TOP - H_TOP - GAP_V - H_BOT,  RIGHT - LEFT, H_BOT])
cax    = fig.add_axes([RIGHT + 0.005, TOP - H_TOP,        0.015,         H_TOP])

def draw_panel(ax, A, title, ylabel, show_seg_labels=False):
    im = ax.imshow(A, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                   extent=[x_lo, x_hi, A.shape[0], 0])
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(A.shape[0], 0)
    ax.set_ylabel(ylabel, fontsize=11.5)
    # Title goes *inside* the axes (upper-left) so it never collides
    # with the coloured segment bars sitting outside the top spine.
    ax.text(0.005, 0.965, title, transform=ax.transAxes,
            ha='left', va='top', fontsize=13, fontweight='bold',
            color='#222',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor='none', alpha=0.78))
    years = pd.date_range('1985', PLOT_END + pd.offsets.YearBegin(1), freq='2YS')
    ax.set_xticks(years)
    ax.set_xticklabels([y.year for y in years], fontsize=10)
    ax.tick_params(axis='y', labelsize=9)
    return im

im = draw_panel(ax_all, A_all, f'All  {A_all.shape[0]}  CAMELS-H training basins '
                              f'(rows sorted by first month with data)',
                'basins  →')
draw_panel(ax_tn, A_tn, f'Tennessee River Basin  ·  {A_tn.shape[0]}  evaluation gauges',
           'TN basins  →')


# --- Split overlay lines + labels ---
def split_overlay(ax, ymax_data):
    for x, ls in [(VAL_START, '-'), (TRAIN_START, '-'),
                  (TEST_START, '-'), (TEST_END, '-')]:
        ax.axvline(x, color='#1a1a1a', linewidth=1.4, alpha=0.85, zorder=10)
    # span shadings for clarity (drawn lightly above the data)
    # We do NOT shade over the data; instead, label segments above the panel.

# Labels above the top panel
def seg_label(ax, x0, x1, txt, color, y_off=1.04):
    xmid = x0 + (x1 - x0) / 2
    ax.text(xmid, y_off, txt, transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            color=color)
    # underline-style bar
    ax.plot([x0, x1], [y_off - 0.012, y_off - 0.012],
            transform=ax.get_xaxis_transform(),
            color=color, linewidth=3, solid_capstyle='butt', clip_on=False)

split_overlay(ax_all, A_all.shape[0])
split_overlay(ax_tn,  A_tn.shape[0])
seg_label(ax_all, pd.Timestamp('1985-01-01'), VAL_START - pd.Timedelta(days=1),
          'Before our data window', '#888')
seg_label(ax_all, VAL_START, TRAIN_START - pd.Timedelta(days=1),
          'Validation 1995–96   (2 yr)', '#cc8a1d')
seg_label(ax_all, TRAIN_START, TEST_START - pd.Timedelta(days=1),
          'Training 1997–2018   (22 yr)', '#2f6bcc')
seg_label(ax_all, TEST_START, TEST_END,
          'Testing 2019–22   (4 yr)', '#2a8c3a')


# --- Colorbar ---
cb = plt.colorbar(im, cax=cax)
cb.set_label('Monthly observed-Q availability', fontsize=11)
cb.ax.tick_params(labelsize=9)


# --- Bottom caption ---
n_all, n_months = A_all.shape

# Per-basin record length: count months where the basin had ≥ 80 % monthly
# coverage *within the training window 1997-2018*.  That is the operational
# story — how many years of training each basin actually contributes.
train_col_lo = int(np.where(ym_dt >= TRAIN_START)[0][0])
train_col_hi = int(np.where(ym_dt < TEST_START)[0][-1] + 1)
train_block  = A_all[:, train_col_lo:train_col_hi]
train_months_per_basin = (np.nan_to_num(train_block) >= 0.80).sum(axis=1)
train_years = train_months_per_basin / 12
med_yr = float(np.median(train_years))
n_full = int((train_months_per_basin >= 22 * 12 * 0.9).sum())

fig.text(0.5, 0.045,
         f'618 basins × 456 months of hourly Q observations.  '
         f'Within the training window (1997–2018, 22 yr), each basin contributes '
         f'a median of {med_yr:.0f} years of data ({n_full} basins are essentially complete).  '
         f'The 2019–2022 test span is set aside before training and is the '
         f'only data shown in any results figure.',
         ha='center', va='center', fontsize=11, color='#222',
         bbox=dict(boxstyle='round,pad=0.45', facecolor='#f5f3e8',
                   edgecolor='#aaaaaa', linewidth=0.6))


# --- Deck chrome ---
ss.add_title_banner(
    fig,
    'Data Availability Atlas — Hourly Q Coverage, 618 CONUS Basins (1985–2022)',
    fontsize=18)

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
plt.savefig(OUT_PNG, dpi=200)
plt.close()
print(f'Saved: {OUT_PNG}')
