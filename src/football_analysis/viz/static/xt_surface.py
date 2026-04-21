"""xT grid surface — visualise the learned value-of-possession grid."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from mplsoccer import Pitch

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M
from football_analysis.analytics.possession_value.xt import XTGrid
from football_analysis.viz.theme import DEFAULT_THEME, Theme


def plot_xt_surface(grid: XTGrid, title: str | None = None, theme: Theme | None = None) -> Figure:
    """Render the xT grid as a heatmap overlaid on a pitch."""
    t = theme or DEFAULT_THEME
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=PITCH_LENGTH_M,
        pitch_width=PITCH_WIDTH_M,
        line_color=t.pitch_line,
    )
    fig, ax = pitch.draw(figsize=(10, 7))

    # Centres of each grid cell
    xs = np.linspace(PITCH_LENGTH_M / grid.cols / 2, PITCH_LENGTH_M - PITCH_LENGTH_M / grid.cols / 2, grid.cols)
    ys = np.linspace(PITCH_WIDTH_M / grid.rows / 2, PITCH_WIDTH_M - PITCH_WIDTH_M / grid.rows / 2, grid.rows)

    xx, yy = np.meshgrid(xs, ys)
    ax.contourf(xx, yy, grid.values, levels=20, cmap=t.xt_cmap, alpha=0.7, zorder=0.5)

    ax.set_title(title or f"Expected Threat surface ({grid.rows}x{grid.cols})")
    return fig
