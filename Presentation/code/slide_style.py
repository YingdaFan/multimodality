"""
Shared 'deck' styling — echoes the SAFER-Hydro / InflowForecast slide design
so every exported figure looks like it belongs to the same presentation.

Design language lifted from InflowForecast.pdf:
  - green gradient rounded title banner with a soft drop shadow
  - sage-green takeaway callout box (the '❖' boxes in the deck)
  - circular page-number badge
  - semantic content colours (blue=prediction, green=forecasting, peach=inputs)
  - Liberation Sans throughout

Usage:
    import slide_style as ss
    ss.apply_rc()
    ss.add_title_banner(fig, 'Slide title')
    ss.add_callout(fig, 'one-sentence takeaway', rect=[x0, y0, w, h])
    ss.add_page_badge(fig, 9)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.colors import LinearSegmentedColormap

FONT = 'Liberation Sans'

COLORS = {
    'banner_top':   '#dcebc4',   # banner gradient — light (top)
    'banner_mid':   '#aed080',
    'banner_bot':   '#8cbd5c',   # banner gradient — darker (bottom)
    'banner_edge':  '#4e7a2f',
    'title_text':   '#1a1a1a',
    'prediction':   '#d4e6f1',   # light blue
    'forecasting':  '#e2efd9',   # light green
    'inputs':       '#fce4d6',   # peach
    'callout':      '#aecaa1',   # sage-green takeaway box
    'callout_edge': '#7da66a',
    'badge':        '#ededed',
    'badge_edge':   '#b9b9b9',
}


def apply_rc():
    """Set the shared matplotlib defaults (call once per figure script)."""
    plt.rcParams.update({
        'font.family':      FONT,
        'font.size':        11,
        'axes.titleweight': 'bold',
        'axes.edgecolor':   '#444444',
        'axes.linewidth':   0.9,
    })


def _round(fig, radius_in, coord_w_in, coord_h_in):
    """rounding_size + mutation_aspect for ~circular `radius_in`-inch corners.

    coord_w_in / coord_h_in = inches spanned by one unit of the patch's
    transform (fig width/height for transFigure; axes width/height for
    transAxes). FancyBboxPatch squeezes y by mutation_aspect, rounds, then
    stretches back, so circular corners need mutation_aspect = w/h and the
    corner radius works out to rounding_size * coord_w_in.
    """
    return radius_in / coord_w_in, coord_w_in / coord_h_in


def add_title_banner(fig, text, height=0.092, side_pad=0.012, top_pad=0.016,
                     fontsize=20):
    """Green gradient rounded title banner across the top of the figure."""
    x0, w = side_pad, 1 - 2 * side_pad
    h = height
    y0 = 1 - top_pad - h
    fw, fh = fig.get_size_inches()

    # soft drop shadow (drawn in figure coords)
    rs_f, ma_f = _round(fig, 0.13, fw, fh)
    fig.patches.append(FancyBboxPatch(
        (x0 + 0.0025, y0 - 0.006), w, h,
        boxstyle=f'round,pad=0,rounding_size={rs_f}', mutation_aspect=ma_f,
        transform=fig.transFigure, facecolor='#00000018', edgecolor='none',
        zorder=900))

    # banner axes carrying a vertical green gradient, clipped to a rounded box
    bax = fig.add_axes([x0, y0, w, h], zorder=901)
    bax.set_xlim(0, 1)
    bax.set_ylim(0, 1)
    bax.axis('off')
    cmap = LinearSegmentedColormap.from_list(
        'banner', [COLORS['banner_bot'], COLORS['banner_mid'],
                   COLORS['banner_top']])
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    im = bax.imshow(grad, aspect='auto', cmap=cmap, extent=[0, 1, 0, 1],
                    origin='lower', zorder=0)
    rs_a, ma_a = _round(fig, 0.13, w * fw, h * fh)
    box = FancyBboxPatch((0.0, 0.0), 1, 1,
                         boxstyle=f'round,pad=0,rounding_size={rs_a}',
                         mutation_aspect=ma_a, transform=bax.transAxes,
                         facecolor='none', edgecolor=COLORS['banner_edge'],
                         linewidth=2.2, zorder=2)
    bax.add_patch(box)
    im.set_clip_path(box)
    bax.text(0.5, 0.5, text, transform=bax.transAxes, ha='center', va='center',
             fontsize=fontsize, fontweight='bold', color=COLORS['title_text'],
             zorder=3)
    return bax


def add_callout(fig, text, rect, fontsize=12, ha='left', radius_in=0.16):
    """Sage-green takeaway box. rect = [x0, y0, w, h] in figure fractions."""
    x0, y0, w, h = rect
    fw, fh = fig.get_size_inches()
    rs, ma = _round(fig, radius_in, w * fw, h * fh)
    fig.patches.append(FancyBboxPatch(
        (x0, y0), w, h, boxstyle=f'round,pad=0,rounding_size={rs}',
        mutation_aspect=ma, transform=fig.transFigure,
        facecolor=COLORS['callout'], edgecolor=COLORS['callout_edge'],
        linewidth=1.6, zorder=900))
    tx = x0 + 0.018 if ha == 'left' else x0 + w / 2
    fig.text(tx, y0 + h / 2, text, ha=ha, va='center', fontsize=fontsize,
             color='#1a1a1a', linespacing=1.5, zorder=901)


def add_page_badge(fig, n, fontsize=12):
    """Small circular page-number badge in the bottom-right corner."""
    fw, fh = fig.get_size_inches()
    d = 0.34 / fh                       # badge diameter ~0.34 in, in fig fracs
    ax = fig.add_axes([1 - d * (fh / fw) - 0.012, 0.012,
                       d * (fh / fw), d], zorder=950)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.add_patch(Circle((0.5, 0.5), 0.5, facecolor=COLORS['badge'],
                        edgecolor=COLORS['badge_edge'], linewidth=1.0))
    ax.text(0.5, 0.5, str(n), ha='center', va='center', fontsize=fontsize,
            color='#333333')
