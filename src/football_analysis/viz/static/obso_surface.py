"""Static OBSO surface renderer (mplsoccer).

Shows the OBSO heatmap (hot = dangerous cell for the attacking team), all 22
players coloured by team, and the ball. Useful for eyeballing whether a
particular frame has recognisable off-ball threat — e.g. a forward making a
run into the box while the ball is in midfield.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from mplsoccer import Pitch

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M
from football_analysis.analytics.pitch_control.obso import OBSOFrame
from football_analysis.viz.theme import DEFAULT_THEME, Theme


def plot_obso_frame(
    tracking: pd.DataFrame,
    frame_id: int,
    attacking_team_id: str,
    defending_team_id: str,
    surface: OBSOFrame,
    title: str | None = None,
    theme: Theme | None = None,
    team_names: dict[str, str] | None = None,
) -> Figure:
    """Render the OBSO heatmap for one frame."""
    t = theme or DEFAULT_THEME
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=PITCH_LENGTH_M,
        pitch_width=PITCH_WIDTH_M,
        line_color=t.pitch_line,
    )
    fig, ax = pitch.draw(figsize=(11.5, 7.4))

    xx, yy = np.meshgrid(surface.xs, surface.ys)
    cs = ax.contourf(xx, yy, surface.obso, levels=20, cmap="inferno", alpha=0.7, zorder=0.5)
    cbar = fig.colorbar(cs, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("OBSO (goal probability units)", rotation=90, labelpad=12)

    frame = tracking[tracking["frame_id"] == frame_id]
    atk = frame[(frame["team_id"] == attacking_team_id) & (~frame["is_ball"]) & frame["visible"]]
    dfd = frame[(frame["team_id"] == defending_team_id) & (~frame["is_ball"]) & frame["visible"]]
    if not atk.empty:
        pitch.scatter(atk["x"], atk["y"], ax=ax, s=260, color=t.home, edgecolors="black", zorder=3)
    if not dfd.empty:
        pitch.scatter(dfd["x"], dfd["y"], ax=ax, s=260, color=t.away, edgecolors="black", zorder=3)
    ball = frame[frame["is_ball"] & frame["visible"]]
    if not ball.empty:
        pitch.scatter(
            ball["x"],
            ball["y"],
            ax=ax,
            s=140,
            color="white",
            edgecolors="black",
            linewidth=1.5,
            marker="o",
            zorder=5,
        )

    atk_name = (team_names or {}).get(attacking_team_id, attacking_team_id)
    def_name = (team_names or {}).get(defending_team_id, defending_team_id)
    ax.set_title(title or f"OBSO — {atk_name} attacking (blue) vs {def_name} (red) — frame {frame_id}")
    return fig  # type: ignore[no-any-return]
