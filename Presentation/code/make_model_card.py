"""
Model Card — replaces the generic pipeline_flowchart with a substantive
methodology slide. Three deck-coloured content blocks side-by-side:

  - Inputs (peach):  11 dynamic met vars + 9 rolling-cumulative + 24 static
  - Model  (blue):   FutureTST architecture + global training setup
  - Output (green):  what comes out (per-basin 18-step prediction, metrics)

All numbers/names sourced from data_processing/preprocess_camelsh_forecast.py
and run_forecast.sh — no hand-typed placeholders.

Output: output/presentation/model_card.png
"""

# --- pin CWD to imputation/ so relative paths resolve no matter where the
# --- script is launched from (the file now lives in Presentation/code/)
import os as _os
from pathlib import Path as _Path
_os.chdir(_Path(__file__).resolve().parents[2])


import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import slide_style as ss


OUT_PNG = 'Presentation/presentation/model_card.png'

ss.apply_rc()
plt.rcParams.update({'figure.dpi': 150, 'font.size': 11})


# ============================== Content ==============================

DYNAMIC = [
    ('Rainf',       'mm/h',   'Rainfall (primary forcing)'),
    ('Tair',        'K',      '2-m air temperature'),
    ('Qair',        'kg/kg',  'Specific humidity'),
    ('PSurf',       'Pa',     'Surface pressure'),
    ('Wind_E',      'm/s',    'Zonal wind'),
    ('Wind_N',      'm/s',    'Meridional wind'),
    ('SWdown',      'W/m²',   'Downward shortwave radiation'),
    ('LWdown',      'W/m²',   'Downward longwave radiation'),
    ('PotEvap',     'mm/h',   'Potential evaporation'),
    ('CAPE',        'J/kg',   'Convective avail. pot. energy'),
    ('CRainf_frac', '–',      'Convective rainfall fraction'),
]

CUMUL = [
    ('Rainf_sum_{24,72,168} h',     'mm',  'Rolling rainfall  (1 d, 3 d, 7 d)'),
    ('Tair_avg_{24,72,168} h',      'K',   'Rolling temperature'),
    ('PotEvap_sum_{24,72,168} h',   'mm',  'Rolling potential ET'),
]

STATIC_GROUPS = [
    ('Climate (9)',
     'p_mean, pet_mean, aridity, p_seasonality,\n'
     'frac_snow, high/low_prec_freq, high/low_prec_dur'),
    ('Topography (1)',
     'area_sqkm'),
    ('HydroATLAS (14)',
     'ele_mt_sav, slp_dg_uav, ria_ha_usu, run_mm_syr, gwt_cm_sav,\n'
     'cly/slt/snd_pc_uav, kar/prm/pac_pc_use, crp/for/urb_pc_use'),
]

TRAINING = [
    ('Model',         'FutureTST  (Transformer encoder + decoder)'),
    ('Encoder',       '168 h history → latent representation'),
    ('Decoder',       'latent + future met → 18 h Q forecast'),
    ('Joint training', '618 CONUS basins  (1 model, no per-region fine-tune)'),
    ('Batch',         '618  (one full basin set per step)'),
    ('Epochs',        '200  (early-stop patience = 20)'),
    ('Learning rate', '0.001  (Adam)'),
    ('Stride',        '24 h  →  one new 18-h forecast per day '
                      '(~358 forecasts / basin / year)'),
    ('Split',         'val 1995–96 (2 y) · train 1997–2018 (22 y) ·\n'
                      'test 2019–22 (4 y)'),
    ('Imputation prior',
                       'y_imputed injected from upstream diffusion pipeline\n'
                       '(handles gaps in observed Q during training)'),
]

OUTPUTS = [
    ('Raw output',     '18-step standardised Q   per window'),
    ('Denormalized',   '× σ_basin + μ_basin  →  Q [mm/day]\n'
                       '(specific runoff = streamflow / area × 86.4,\n'
                       ' so cross-basin comparable)'),
    ('Reported',       'NSE · KGE · RMSE · MAE · R² · pbias'),
    ('Per-lead view',  '1, 2, …, 18-hour-ahead skill curves'),
    ('Evaluation set', '130 TRB gauges  (104 with valid all-lead metrics)'),
]


# ============================== Figure ==============================

FIG_W, FIG_H = 22, 12.5
fig = plt.figure(figsize=(FIG_W, FIG_H))


def block(ax, title, color, edge='#666666'):
    """Style a panel as a deck content block with a coloured title bar."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 1, 1,
                                boxstyle='round,pad=0,rounding_size=0.02',
                                mutation_aspect=ax.get_position().width
                                                / ax.get_position().height
                                                * FIG_W / FIG_H,
                                transform=ax.transAxes,
                                facecolor=color, edgecolor=edge,
                                linewidth=1.0, zorder=0))
    # title strip
    ax.add_patch(plt.Rectangle((0, 0.92), 1, 0.08,
                               transform=ax.transAxes,
                               facecolor='white', alpha=0.55, edgecolor='none',
                               zorder=1))
    ax.text(0.5, 0.96, title, ha='center', va='center',
            transform=ax.transAxes,
            fontsize=15, fontweight='bold', color='#222')


# Three columns
LEFT, RIGHT = 0.025, 0.975
TOP, BOTTOM = 0.835, 0.06
GAP = 0.012
W_COL = ((RIGHT - LEFT) - 2 * GAP) / 3

ax_in    = fig.add_axes([LEFT,                      BOTTOM, W_COL, TOP - BOTTOM])
ax_model = fig.add_axes([LEFT + (W_COL + GAP),       BOTTOM, W_COL, TOP - BOTTOM])
ax_out   = fig.add_axes([LEFT + 2 * (W_COL + GAP),   BOTTOM, W_COL, TOP - BOTTOM])

block(ax_in,    f'Inputs   ({len(DYNAMIC)} dynamic + {3 * 3} rolling + 24 static)',
      ss.COLORS['inputs'],      '#a36b3a')
block(ax_model, 'FutureTST   ·   Training Setup',
      ss.COLORS['prediction'],  '#3f6a8f')
block(ax_out,   'Outputs',
      ss.COLORS['forecasting'], '#588a4a')


# ---------------- Inputs column ----------------
y = 0.88
ax_in.text(0.05, y, '11 dynamic meteorological forcings',
           transform=ax_in.transAxes,
           fontsize=12, fontweight='bold', color='#222')
y -= 0.026
ax_in.text(0.05, y, 'ERA5-Land hourly, 1985-present',
           transform=ax_in.transAxes,
           fontsize=9.5, style='italic', color='#555')
y -= 0.026
for name, unit, desc in DYNAMIC:
    y -= 0.027
    ax_in.text(0.05, y, '►', transform=ax_in.transAxes,
               fontsize=10, color='#a36b3a')
    ax_in.text(0.085, y, f'{name}',
               transform=ax_in.transAxes,
               fontsize=10.5, fontweight='bold', color='#222',
               family='monospace')
    ax_in.text(0.36, y, f'[{unit}]', transform=ax_in.transAxes,
               fontsize=10, color='#555')
    ax_in.text(0.51, y, desc, transform=ax_in.transAxes,
               fontsize=10, color='#222')

y -= 0.045
ax_in.text(0.05, y, '9 rolling-cumulative features',
           transform=ax_in.transAxes,
           fontsize=12, fontweight='bold', color='#222')
y -= 0.024
ax_in.text(0.05, y, 'derived in preprocessing — capture antecedent state',
           transform=ax_in.transAxes,
           fontsize=9.5, style='italic', color='#555')
y -= 0.025
for name, unit, desc in CUMUL:
    y -= 0.030
    ax_in.text(0.05, y, '►', transform=ax_in.transAxes,
               fontsize=10, color='#a36b3a')
    ax_in.text(0.085, y, name, transform=ax_in.transAxes,
               fontsize=9.5, fontweight='bold', color='#222',
               family='monospace')
    ax_in.text(0.62, y, f'[{unit}]', transform=ax_in.transAxes,
               fontsize=9.5, color='#555')
    y -= 0.022
    ax_in.text(0.085, y, desc, transform=ax_in.transAxes,
               fontsize=9.5, color='#222')

y -= 0.04
ax_in.text(0.05, y, '24 static catchment attributes (per basin)',
           transform=ax_in.transAxes,
           fontsize=12, fontweight='bold', color='#222')
y -= 0.025
for grp, body in STATIC_GROUPS:
    y -= 0.028
    ax_in.text(0.05, y, '►', transform=ax_in.transAxes,
               fontsize=10, color='#a36b3a')
    ax_in.text(0.085, y, grp, transform=ax_in.transAxes,
               fontsize=10.5, fontweight='bold', color='#222')
    nlines = body.count('\n') + 1
    for line in body.split('\n'):
        y -= 0.024
        ax_in.text(0.105, y, line, transform=ax_in.transAxes,
                   fontsize=9.5, color='#222', family='monospace')


# ---------------- Model column ----------------
y = 0.88
ax_model.text(0.05, y, 'Architecture',
              transform=ax_model.transAxes,
              fontsize=12.5, fontweight='bold', color='#222')
for label, val in TRAINING[:3]:
    y -= 0.035
    ax_model.text(0.05, y, f'{label}:',
                  transform=ax_model.transAxes,
                  fontsize=10.5, fontweight='bold', color='#3f6a8f')
    ax_model.text(0.27, y, val, transform=ax_model.transAxes,
                  fontsize=10.5, color='#222')

y -= 0.05
ax_model.text(0.05, y, 'Training', transform=ax_model.transAxes,
              fontsize=12.5, fontweight='bold', color='#222')
for label, val in TRAINING[3:8]:
    y -= 0.038
    ax_model.text(0.05, y, f'{label}:',
                  transform=ax_model.transAxes,
                  fontsize=10.5, fontweight='bold', color='#3f6a8f')
    ax_model.text(0.27, y, val, transform=ax_model.transAxes,
                  fontsize=10.5, color='#222')

y -= 0.05
ax_model.text(0.05, y, 'Data', transform=ax_model.transAxes,
              fontsize=12.5, fontweight='bold', color='#222')
y -= 0.038
ax_model.text(0.05, y, f'{TRAINING[8][0]}:',
              transform=ax_model.transAxes,
              fontsize=10.5, fontweight='bold', color='#3f6a8f')
for line in TRAINING[8][1].split('\n'):
    ax_model.text(0.27, y, line, transform=ax_model.transAxes,
                  fontsize=10.5, color='#222')
    y -= 0.026
y -= 0.012
ax_model.text(0.05, y, f'{TRAINING[9][0]}:',
              transform=ax_model.transAxes,
              fontsize=10.5, fontweight='bold', color='#3f6a8f')
for line in TRAINING[9][1].split('\n'):
    ax_model.text(0.27, y, line, transform=ax_model.transAxes,
                  fontsize=10.5, color='#222')
    y -= 0.024

# Footer note
ax_model.text(0.05, 0.04,
              'Source: run_forecast.sh + preprocess_camelsh_forecast.py',
              transform=ax_model.transAxes,
              fontsize=8.5, style='italic', color='#555')


# ---------------- Output column ----------------
y = 0.90
for label, val in OUTPUTS:
    y -= 0.046
    ax_out.text(0.06, y, '►', transform=ax_out.transAxes,
                fontsize=11, color='#588a4a', va='center')
    ax_out.text(0.10, y, label, transform=ax_out.transAxes,
                fontsize=11.5, fontweight='bold', color='#222',
                va='center')
    y -= 0.034
    ax_out.text(0.10, y, val, transform=ax_out.transAxes,
                fontsize=10.5, color='#222', linespacing=1.4,
                va='top')
    # Reserve extra vertical room for each additional line in val
    y -= 0.026 * val.count('\n')

# Key results box
y -= 0.06
ax_out.add_patch(plt.Rectangle((0.05, y - 0.10), 0.90, 0.13,
                               transform=ax_out.transAxes,
                               facecolor='white', alpha=0.55,
                               edgecolor='#588a4a', linewidth=1.0))
ax_out.text(0.50, y - 0.014, 'Key Results  (TRB, 18-hour horizon)',
            transform=ax_out.transAxes,
            ha='center', fontsize=12, fontweight='bold', color='#222')
ax_out.text(0.50, y - 0.05,
            'median 1-h NSE  =  0.996      median 18-h NSE  =  0.863',
            transform=ax_out.transAxes,
            ha='center', fontsize=11, color='#222', family='monospace')
ax_out.text(0.50, y - 0.082,
            '94 % of TRB gauges keep NSE ≥ 0.7 at the 18-hour horizon',
            transform=ax_out.transAxes,
            ha='center', fontsize=10.5, color='#222')


# ---------------- Arrows between blocks ----------------
def arrow_between(ax_a, ax_b):
    """Draw a green arrow from the right edge of ax_a to the left edge of ax_b."""
    pa, pb = ax_a.get_position(), ax_b.get_position()
    y = (pa.y0 + pa.y1) / 2
    fig.patches.append(FancyArrowPatch(
        (pa.x1 + 0.0005, y), (pb.x0 - 0.0005, y),
        transform=fig.transFigure, arrowstyle='-|>',
        mutation_scale=24, lw=2.0, color=ss.COLORS['banner_edge'], zorder=950))

arrow_between(ax_in,    ax_model)
arrow_between(ax_model, ax_out)


# ---------------- Deck chrome ----------------
ss.add_title_banner(
    fig,
    'Model Card — FutureTST · 35 features × 168 h → 18 h Streamflow Forecast',
    fontsize=20)

os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
plt.savefig(OUT_PNG, dpi=200)
plt.close()
print(f'Saved: {OUT_PNG}')
