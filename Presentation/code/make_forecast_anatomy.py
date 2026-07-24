"""
Forecast Anatomy — replaces the schematic sliding-window figure with a
real-data view of what *one* FutureTST prediction actually is.

  - Top panel:    hourly rainfall (one of 11 dynamic inputs)
  - Main panel:   observed Q (black) with the 168-h history shaded blue,
                  the 18-h forecast region shaded red, and the actual
                  model prediction overlaid as red markers.
  - Side caption: stride = 24 h → 358 forecasts/year/basin, with the
                  next window's history/forecast bands drawn in faded
                  outline to make 'sliding' literal.

Output: output/presentation/forecast_anatomy.png
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
from matplotlib.patches import Rectangle

import slide_style as ss


PRED_DIR  = 'diffusion_forecast/output/pred'
PREPPED   = 'data_processing/data/prepped.npz'
PARQUET   = '../camelsh_tennessee.parquet'
OUT_PNG   = 'Presentation/presentation/forecast_anatomy.png'

BASIN     = '03527000'      # TRB, area 2916 km², 18h NSE = 0.93 — clean example
WINDOW_K  = 47              # Feb 17 → Feb 25, 2019 — covers the Feb 24 flood event

ss.apply_rc()
plt.rcParams.update({'figure.dpi': 150, 'font.size': 11})


# ============================== Load ==============================
with open(os.path.join(PRED_DIR, 'meta.json')) as f:
    meta = json.load(f)
WIN, PLEN, STRIDE = int(meta['window']), int(meta['pred_len']), int(meta['stride'])

d            = np.load(PREPPED, allow_pickle=True)
basin_names  = [str(b) for b in d['basin_names']]
i_b          = basin_names.index(BASIN)
times        = pd.to_datetime(d['times_tst'])
y_obs        = d['y_raw_tst'][i_b, :, 0]                         # mm/day
y_mean       = float(d['y_mean'][i_b]); y_std = float(d['y_std'][i_b])

n_basins     = len(basin_names)
y_pred_all   = np.load(os.path.join(PRED_DIR, 'tst.npy'))         # (n_off*n_b, 18, 1)
n_offsets    = y_pred_all.shape[0] // n_basins

def pred_window(k):
    raw = y_pred_all[k * n_basins + i_b, :, 0] * y_std + y_mean
    return np.maximum(raw, 0.0)

# This window + the next one, to make the 'sliding' explicit
y_pred_w  = pred_window(WINDOW_K)
y_pred_w1 = pred_window(WINDOW_K + 1)

# Hour indices for the focal window
H0 = WINDOW_K * STRIDE
H1 = H0 + WIN                  # forecast start (= history end)
H2 = H1 + PLEN                 # forecast end
# Next window's hours
N0 = (WINDOW_K + 1) * STRIDE
N1 = N0 + WIN
N2 = N1 + PLEN

# Period to show:  start 1 d before the focal window, end 1 d after the next forecast
plot_start = max(H0 - 24, 0)
plot_end   = min(N2 + 24, len(times))

t_plot = times[plot_start:plot_end]

# Rainf (one of 11 dynamic met inputs) for the same span — direct from parquet
rain = (pd.read_parquet(PARQUET, columns=['Time', 'basin_id', 'Rainf'])
          .query("basin_id == @BASIN").set_index('Time').sort_index())
rain.index = pd.to_datetime(rain.index)
rain_slice = rain.loc[t_plot[0]:t_plot[-1], 'Rainf']

print(f'Basin {BASIN}  |  focal window k={WINDOW_K}')
print(f'  history:  {times[H0]}  →  {times[H1]}  ({WIN} h)')
print(f'  forecast: {times[H1]}  →  {times[H2]}  ({PLEN} h)')
print(f'  next forecast starts at: {times[N1]}  (stride = {STRIDE} h)')


# ============================== Figure ==============================
FIG_W, FIG_H = 22, 8
fig = plt.figure(figsize=(FIG_W, FIG_H))

ax_rain = fig.add_axes([0.05, 0.66, 0.93, 0.16])
ax_q    = fig.add_axes([0.05, 0.13, 0.93, 0.52])

# ---- top: rainfall ----
ax_rain.bar(rain_slice.index, rain_slice.values,
            width=pd.Timedelta(hours=1.0), align='edge',
            color='#4a90d9', edgecolor='none', alpha=0.85)
ax_rain.set_ylabel('Rainf\n[mm/h]', fontsize=11)
ax_rain.tick_params(axis='x', labelbottom=False)
ax_rain.invert_yaxis()
for s in ('top', 'right', 'bottom'):
    ax_rain.spines[s].set_visible(False)
ax_rain.grid(axis='x', alpha=0.15, linewidth=0.4)

# ---- main: Q ----
ax_q.plot(t_plot, y_obs[plot_start:plot_end], '-', color='black',
          linewidth=1.4, zorder=4, label='Observed Q')

# Focal window — history + forecast
ax_q.axvspan(times[H0], times[H1], color='#5bc0de', alpha=0.22, zorder=1,
             label='168-h history')
ax_q.axvspan(times[H1], times[H2], color='#d9534f', alpha=0.28, zorder=1,
             label='18-h forecast')

# Prediction
t_fcst = times[H1:H2]
ax_q.plot(t_fcst, y_pred_w, 'o-', color='#b00000', linewidth=1.8,
          markersize=5, alpha=0.95, zorder=6,
          label='FutureTST prediction')

# Next window — faded outline only, to make "stride = 24 h" literal
ymin, ymax = ax_q.get_ylim()
ax_q.add_patch(Rectangle((mdates.date2num(times[N0]), ymin),
                         mdates.date2num(times[N1]) - mdates.date2num(times[N0]),
                         ymax - ymin,
                         facecolor='none', edgecolor='#5bc0de',
                         linewidth=1.2, linestyle='--', alpha=0.55, zorder=2))
ax_q.add_patch(Rectangle((mdates.date2num(times[N1]), ymin),
                         mdates.date2num(times[N2]) - mdates.date2num(times[N1]),
                         ymax - ymin,
                         facecolor='none', edgecolor='#d9534f',
                         linewidth=1.2, linestyle='--', alpha=0.55, zorder=2))
ax_q.set_ylim(ymin, ymax)

# Stride annotation arrow
ymid = ymin + 0.85 * (ymax - ymin)
ax_q.annotate('', xy=(times[N1], ymid), xytext=(times[H1], ymid),
              arrowprops=dict(arrowstyle='->', color='#444', lw=1.4))
ax_q.text(times[H1] + (times[N1] - times[H1]) / 2, ymid + 0.03 * (ymax - ymin),
          f'stride = {STRIDE} h\n(next window)', ha='center', va='bottom',
          fontsize=10, color='#444', style='italic')

# Window-edge labels at the top
ax_q.text(times[H0] + (times[H1] - times[H0]) / 2, ymax * 0.96,
          '168-h history (input)', ha='center', va='top',
          fontsize=11, fontweight='bold', color='#0a5e7a')
ax_q.text(times[H1] + (times[H2] - times[H1]) / 2, ymax * 0.96,
          '18-h forecast\n(output)', ha='center', va='top',
          fontsize=10, fontweight='bold', color='#8b1a1a')

ax_q.set_ylabel('Q  [mm/day]', fontsize=12)
ax_q.set_xlabel('Time (UTC)', fontsize=11)
ax_q.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax_q.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax_q.grid(axis='y', alpha=0.25, linewidth=0.4)
ax_q.legend(loc='upper left', fontsize=11, framealpha=0.92)
for s in ('top', 'right'):
    ax_q.spines[s].set_visible(False)

# Sync x-limits of the two panels
ax_rain.set_xlim(ax_q.get_xlim())


# Caption strip — what flows in / out
fig.text(0.5, 0.04,
         'Inputs per window: 168 h × (past Q + 11 dynamic met + 9 rolling-cumulative)  +  '
         '24 static catchment attributes.   Output: 18-step ahead Q.   '
         'Stride 24 h → 358 forecasts per basin per year.',
         ha='center', va='center', fontsize=11, color='#222',
         bbox=dict(boxstyle='round,pad=0.45', facecolor='#f5f3e8',
                   edgecolor='#aaaaaa', linewidth=0.6))


# Deck chrome
ss.add_title_banner(
    fig,
    f'Forecast Anatomy — basin {BASIN}, Feb 24 2019 flood event '
    f'(168 h → 18 h, stride {STRIDE} h)',
    fontsize=18)

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
plt.savefig(OUT_PNG, dpi=200)
plt.close()
print(f'\nSaved: {OUT_PNG}')
