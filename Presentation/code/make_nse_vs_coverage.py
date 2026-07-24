"""
Standalone deck-style versions of the two scatter panels (d) and (e)
from coverage_vs_nse.png:

  - NSE vs TRAINING coverage  (Pearson r ≈ 0.29 — real positive link)
  - NSE vs TEST coverage      (Pearson r ≈ 0.05 — essentially no link
                               once a basin has *any* test data)

Outputs:
  output/presentation/nse_vs_train_cov.png
  output/presentation/nse_vs_test_cov.png
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


CSV     = 'Presentation/presentation/tn_target_coverage.csv'
OUT_DIR = 'Presentation/presentation'
EXCLUDED = {'03566420'}

ss.apply_rc()
plt.rcParams.update({
    'figure.dpi':       150,
    'font.size':        12,
    'axes.titlesize':   14,
    'axes.labelsize':   13,
    'xtick.labelsize':  11,
    'ytick.labelsize':  11,
})


df = pd.read_csv(CSV, dtype={'basin_id': str})
df = df.dropna(subset=['mean_nse'])
df = df[~df['basin_id'].isin(EXCLUDED)].reset_index(drop=True)
n  = len(df)
print(f'Plot points: {n} basins')


def make_panel(x_col, c_col, xlabel, c_label, title, out_path,
               cmap_name='viridis'):
    fig = plt.figure(figsize=(11, 7))
    ax  = fig.add_axes([0.10, 0.13, 0.78, 0.68])

    r = float(df[[x_col, 'mean_nse']].corr().iloc[0, 1])
    sc = ax.scatter(df[x_col], df['mean_nse'],
                    c=df[c_col], cmap=cmap_name, vmin=0, vmax=1,
                    s=70, edgecolor='black', linewidth=0.5, alpha=0.9)
    cb = plt.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label(c_label)

    ax.axhline(0.7, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Mean NSE (1–18 h leads)')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(min(0.55, df['mean_nse'].min() - 0.02), 1.02)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    ax.text(0.02, 0.96, f'Pearson  r = {r:.2f}\nn = {n} basins',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=12, fontweight='bold', color='#222',
            bbox=dict(boxstyle='round,pad=0.35',
                      facecolor='white', edgecolor='#888', linewidth=0.6))

    ss.add_title_banner(fig, title, fontsize=18)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f'Saved {out_path}  (r = {r:.3f})')


make_panel(
    x_col='train_cov', c_col='test_cov',
    xlabel='Training-period target coverage  (fraction of 1997–2018 hours with obs Q)',
    c_label='Test-period target coverage',
    title='NSE vs Training-period Target Coverage — TN Basins',
    out_path=os.path.join(OUT_DIR, 'nse_vs_train_cov.png'),
    cmap_name='viridis',
)

make_panel(
    x_col='test_cov', c_col='train_cov',
    xlabel='Test-period target coverage  (fraction of 2019–2022 hours with obs Q)',
    c_label='Training-period target coverage',
    title='NSE vs Test-period Target Coverage — TN Basins',
    out_path=os.path.join(OUT_DIR, 'nse_vs_test_cov.png'),
    cmap_name='viridis',
)
