"""
Experiment Setting — a single deck-style slide combining
  When:   1995-96 val · 1997-2018 train · 2019-22 test
  How:    168 h history → 18 h forecast, stride 24 h

Both sub-panels live in one figure with one banner / one page badge,
so it slots directly into the deck.  (The 'Where' map is shipped as a
separate slide, global_vs_tn_map.png.)

Output: output/presentation/experiment_setting.png
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
from matplotlib.patches import Rectangle, FancyArrowPatch

import slide_style as ss


OUT_PNG = 'Presentation/presentation/experiment_setting.png'

ss.apply_rc()
plt.rcParams.update({'figure.dpi': 150, 'font.size': 11})


# ============================== Figure ==============================
FIG_W, FIG_H = 22, 7.0
fig = plt.figure(figsize=(FIG_W, FIG_H))


NAKED = os.environ.get('NAKED_HEADERS') == '1'

def section_header(text, x, y):
    if NAKED:
        return
    fig.text(x, y, text, fontsize=14, fontweight='bold', color='#2f4f1f',
             ha='left', va='bottom')


# ============================== When — temporal data split ==============================
section_header('When  ·  validation 1995–96 (2 yr)   ·   training 1997–2018 (22 yr)   '
               '·   testing 2019–22 (4 yr, hold-out)',
               x=0.045, y=0.74)

ax_split = fig.add_axes([0.045, 0.55, 0.91, 0.16])

splits = [
    ('Validation', '1995-01-01', '1997-01-01', '#f0ad4e'),
    ('Training',   '1997-01-01', '2019-01-01', '#5bc0de'),
    ('Testing',    '2019-01-01', '2023-01-01', '#5cb85c'),
]
for name, s, e, color in splits:
    s_d, e_d = pd.Timestamp(s), pd.Timestamp(e)
    ax_split.barh(0, (e_d - s_d).days, left=s_d, height=0.55,
                  color=color, edgecolor='black', linewidth=0.7, alpha=0.92)
    yrs = (e_d - s_d).days / 365.25
    ax_split.text(s_d + (e_d - s_d) / 2, 0,
                  f'{name}  {s[:4]}–{int(e[:4]) - 1}  ({yrs:.0f} yr)',
                  ha='center', va='center', fontsize=11, fontweight='bold')

ax_split.set_ylim(-0.55, 0.55)
ax_split.set_yticks([])
ax_split.set_xlim(pd.Timestamp('1994-06-01'), pd.Timestamp('2023-07-01'))
ax_split.xaxis.set_major_locator(mdates.YearLocator(2))
ax_split.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.setp(ax_split.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=9)
for spine in ['left', 'right', 'top']:
    ax_split.spines[spine].set_visible(False)


# ============================== How — sliding window ==============================
section_header('How  ·  168-hour input window  →  18-hour forecast,   '
               'stride 24 h between windows',
               x=0.045, y=0.34)

ax_win = fig.add_axes([0.05, 0.05, 0.90, 0.28])

WIN_H, PRED_H, STRIDE_H = 168, 18, 24
N_ROWS  = 3
ROW_H, ROW_GAP = 0.95, 0.55
TOTAL = WIN_H + PRED_H + (N_ROWS - 1) * STRIDE_H

HIST_FILL  = '#5bc0de'
HIST_INNER = '#0a5e7a'
FCST_FILL  = '#d9534f'

for k in range(N_ROWS):
    y = -k * (ROW_H + ROW_GAP)
    x0 = k * STRIDE_H
    ax_win.add_patch(Rectangle((x0, y), WIN_H, ROW_H, facecolor=HIST_FILL,
                               edgecolor='black', linewidth=0.7, alpha=0.85))
    ax_win.add_patch(Rectangle((x0 + WIN_H, y), PRED_H, ROW_H, facecolor=FCST_FILL,
                               edgecolor='black', linewidth=0.7, alpha=0.85))
    ax_win.text(-6, y + ROW_H / 2, f'window {k + 1}',
                ha='right', va='center', fontsize=10)

    if k == 0:
        ax_win.text(x0 + WIN_H / 2, y + ROW_H * 0.66,
                    f'History — {WIN_H} h  (= 7 days)',
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color=HIST_INNER)
        ax_win.text(x0 + WIN_H / 2, y + ROW_H * 0.30,
                    'past Q  +  11 dynamic met  +  9 rolling-cumul  +  24 static',
                    ha='center', va='center', fontsize=9, color=HIST_INNER,
                    style='italic')
        ax_win.text(x0 + WIN_H + PRED_H / 2, y + ROW_H * 0.65,
                    f'{PRED_H} h\nforecast', ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
        ax_win.text(x0 + WIN_H + PRED_H / 2, y + ROW_H * 0.22,
                    'Q only', ha='center', va='center',
                    fontsize=8, color='white', style='italic')

    if k > 0:
        prev_x = (k - 1) * STRIDE_H
        y_top = y + ROW_H + 0.05
        ax_win.annotate('', xy=(prev_x + STRIDE_H + 1, y_top),
                        xytext=(prev_x - 1, y_top),
                        arrowprops=dict(arrowstyle='-|>', color='#444',
                                        lw=1.2, mutation_scale=14))
        if k == 1:
            ax_win.text(prev_x + STRIDE_H / 2, y_top + 0.18,
                        f'+{STRIDE_H} h', ha='center', va='bottom',
                        fontsize=9, color='#444', style='italic')

ax_win.set_xlim(-30, TOTAL + 12)
ax_win.set_ylim(-(N_ROWS - 1) * (ROW_H + ROW_GAP) - 0.10, ROW_H + 0.08)
ax_win.set_yticks([])
ax_win.set_xticks([])
for s in ('left', 'right', 'top', 'bottom'):
    ax_win.spines[s].set_visible(False)


# ============================== Deck chrome ==============================
if not NAKED:
    ss.add_title_banner(fig,
                        'Experiment Setting — Temporal Split & Forecast Window',
                        fontsize=20)

if NAKED:
    out_path = OUT_PNG.replace('.png', '_naked.png')
else:
    out_path = OUT_PNG
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=200)
plt.close()
print(f'Saved: {out_path}')
