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
    player_names: dict[str, str] | None = None,
    team_names: dict[str, str] | None = None,
) -> Figure:
    """Render a KDE heatmap of a single player's on-ball action start locations.

    If ``player_names`` is supplied the title uses the resolved name; otherwise the id.
    The player's average on-ball position is overlaid as a white X marker.
    """
    t = theme or DEFAULT_THEME
    subset = events[(events["player_id"] == player_id) & events["start_x"].notna() & events["start_y"].notna()]

    pitch = Pitch(
        pitch_type="custom",
        pitch_length=PITCH_LENGTH_M,
        pitch_width=PITCH_WIDTH_M,
        line_color=t.pitch_line,
    )
    fig, ax = pitch.draw(figsize=(11, 7.2))

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
    elif not subset.empty:
        pitch.scatter(subset["start_x"], subset["start_y"], ax=ax, s=60, color=t.home)

    if not subset.empty:
        mx = float(subset["start_x"].mean())
        my = float(subset["start_y"].mean())
        pitch.scatter(
            [mx],
            [my],
            ax=ax,
            s=240,
            color="white",
            edgecolors="black",
            linewidth=1.5,
            marker="X",
            zorder=5,
        )
        ax.annotate(
            "avg pos",
            xy=(mx, my),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color="black",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.85},
        )

    if title is None:
        name = (player_names or {}).get(str(player_id), f"player {player_id}")
        team_id = subset["team_id"].mode().iloc[0] if "team_id" in subset.columns and not subset.empty else None
        team = (team_names or {}).get(str(team_id), "") if team_id is not None else ""
        suffix = f" — {team}" if team else ""
        title = f"Heatmap — {name}{suffix}  ({len(subset)} on-ball actions)"
    ax.set_title(title)
    return fig  # type: ignore[no-any-return]
