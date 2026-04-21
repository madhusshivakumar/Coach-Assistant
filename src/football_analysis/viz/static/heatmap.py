"""Player heatmap — 2D KDE over on-ball action start positions."""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure
from mplsoccer import Pitch

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M
from football_analysis.viz.theme import DEFAULT_THEME, Theme


def plot_player_heatmap(
    events: pd.DataFrame,
    player_id: str,
    title: str | None = None,
    theme: Theme | None = None,
) -> Figure:
    """Render a KDE heatmap of a single player's on-ball action start locations."""
    t = theme or DEFAULT_THEME
    subset = events[(events["player_id"] == player_id) & events["start_x"].notna() & events["start_y"].notna()]

    pitch = Pitch(
        pitch_type="custom",
        pitch_length=PITCH_LENGTH_M,
        pitch_width=PITCH_WIDTH_M,
        line_color=t.pitch_line,
    )
    fig, ax = pitch.draw(figsize=(10, 7))

    if len(subset) >= 5:
        pitch.kdeplot(
            subset["start_x"],
            subset["start_y"],
            ax=ax,
            fill=True,
            levels=12,
            cmap=t.heat_cmap,
            alpha=0.7,
        )
    else:
        pitch.scatter(subset["start_x"], subset["start_y"], ax=ax, s=60, color=t.home)

    ax.set_title(title or f"Heatmap — player {player_id} ({len(subset)} actions)")
    return fig  # type: ignore[no-any-return]
