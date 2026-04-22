"""Single-frame pitch-control renderer (mplsoccer; static PNG output).

Draws the pitch, overlays the pitch-control heatmap (blue = home, red = away),
and scatters the 22 players + ball for the selected frame. This is the
Phase-2 proof-of-life; a Plotly-animated version lands in a later slice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from mplsoccer import Pitch

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M
from football_analysis.analytics.pitch_control.spearman import (
    PitchControlFrame,
    compute_frame_from_tracking,
)
from football_analysis.viz.theme import DEFAULT_THEME, Theme


def plot_frame(
    tracking: pd.DataFrame,
    frame_id: int,
    home_team_id: str,
    away_team_id: str,
    control: PitchControlFrame | None = None,
    title: str | None = None,
    theme: Theme | None = None,
    team_names: dict[str, str] | None = None,
) -> Figure:
    """Render a single tracking frame with the pitch-control surface underneath."""
    t = theme or DEFAULT_THEME
    if control is None:
        control = compute_frame_from_tracking(tracking, frame_id, home_team_id, away_team_id)

    frame = tracking[tracking["frame_id"] == frame_id]
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=PITCH_LENGTH_M,
        pitch_width=PITCH_WIDTH_M,
        line_color=t.pitch_line,
    )
    fig, ax = pitch.draw(figsize=(11.5, 7.4))

    # Heatmap — red-white-blue diverging so 1.0 = home (blue), 0.0 = away (red).
    xx, yy = np.meshgrid(control.xs, control.ys)
    cs = ax.contourf(xx, yy, control.home_control, levels=20, cmap="RdBu", alpha=0.55, zorder=0.5)
    cbar = fig.colorbar(cs, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("P(home controls)  — 0 = away, 1 = home", rotation=90, labelpad=12)

    # Players
    home = frame[(frame["team_id"] == home_team_id) & (~frame["is_ball"]) & frame["visible"]]
    away = frame[(frame["team_id"] == away_team_id) & (~frame["is_ball"]) & frame["visible"]]
    if not home.empty:
        pitch.scatter(home["x"], home["y"], ax=ax, s=260, color=t.home, edgecolors="black", zorder=3)
    if not away.empty:
        pitch.scatter(away["x"], away["y"], ax=ax, s=260, color=t.away, edgecolors="black", zorder=3)

    # Ball
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

    # Title
    if title is None:
        h_name = (team_names or {}).get(home_team_id, home_team_id)
        a_name = (team_names or {}).get(away_team_id, away_team_id)
        title = f"Pitch control — {h_name} (blue) vs {a_name} (red) — frame {frame_id}"
    ax.set_title(title)
    return fig  # type: ignore[no-any-return]
