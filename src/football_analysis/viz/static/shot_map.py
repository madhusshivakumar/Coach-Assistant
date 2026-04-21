"""Shot map — one dot per shot, star for goals. mplsoccer-based."""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure
from mplsoccer import Pitch

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M
from football_analysis.viz.theme import DEFAULT_THEME, Theme


def plot_shot_map(
    events: pd.DataFrame,
    team_id: str | None = None,
    title: str | None = None,
    theme: Theme | None = None,
) -> Figure:
    """Render a shot map from a SPADL events DataFrame.

    Filters to `action_type == 'shot'`; if `team_id` given, scopes to that team.
    Goals are stars; other shots are filled dots.
    """
    t = theme or DEFAULT_THEME
    shots = events[events["action_type"] == "shot"]
    if team_id is not None:
        shots = shots[shots["team_id"] == team_id]

    pitch = Pitch(
        pitch_type="custom",
        pitch_length=PITCH_LENGTH_M,
        pitch_width=PITCH_WIDTH_M,
        line_color=t.pitch_line,
    )
    fig, ax = pitch.draw(figsize=(10, 7))

    goals = shots[shots["result"] == "success"]
    others = shots[shots["result"] != "success"]

    if not others.empty:
        pitch.scatter(
            others["start_x"], others["start_y"], ax=ax,
            s=80, color=t.away, alpha=0.75, edgecolors="black", label="shot",
        )
    if not goals.empty:
        pitch.scatter(
            goals["start_x"], goals["start_y"], ax=ax,
            s=160, color=t.goal, edgecolors="black", linewidth=1.5, marker="*", label="goal",
        )

    n_goals = len(goals)
    n_shots = len(shots)
    ax.set_title(title or f"Shots ({n_shots} total, {n_goals} goals)")
    if n_shots > 0:
        ax.legend(loc="upper left")
    return fig
