"""Shot map with team colours, goal annotations, and a match-score header."""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure
from mplsoccer import Pitch

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M
from football_analysis.viz.theme import DEFAULT_THEME, Theme


def _match_header(
    shots: pd.DataFrame,
    team_names: dict[str, str] | None,
    home_team_id: str | None,
) -> str:
    """Compose `{home_name} {home_goals} – {away_goals} {away_name}`.

    Excludes the penalty shootout (StatsBomb period 5) so the header reports the
    match result, not the combined total of regulation + shootout conversions.
    Returns empty string when team metadata is missing.
    """
    if not team_names or home_team_id is None:
        return ""
    ids = list(team_names.keys())
    away_ids = [t for t in ids if t != home_team_id]
    if not away_ids:
        return ""
    away_team_id = away_ids[0]
    in_regulation = shots["period"] != 5 if "period" in shots.columns else True
    regulation_goals = shots[(shots["result"] == "success") & in_regulation]
    goals_home = int((regulation_goals["team_id"] == home_team_id).sum())
    goals_away = int((regulation_goals["team_id"] == away_team_id).sum())
    return (
        f"{team_names.get(home_team_id, home_team_id)} {goals_home} – "
        f"{goals_away} {team_names.get(away_team_id, away_team_id)}"
    )


def plot_shot_map(
    events: pd.DataFrame,
    team_id: str | None = None,
    title: str | None = None,
    theme: Theme | None = None,
    home_team_id: str | None = None,
    team_names: dict[str, str] | None = None,
    player_names: dict[str, str] | None = None,
) -> Figure:
    """Render a shot map from a SPADL events DataFrame.

    - Dots are coloured by team when `home_team_id` is supplied.
    - Goals are drawn as gold stars ringed in their team's colour; scorer name is annotated.
    - Title defaults to `Shots — {home} {goals_h} – {goals_a} {away}` when names are supplied.
    """
    t = theme or DEFAULT_THEME
    shots = events[events["action_type"] == "shot"].copy()
    if team_id is not None:
        shots = shots[shots["team_id"] == team_id]
    # Draw only regulation + extra-time shots; penalty shootouts clutter the pitch
    if "period" in shots.columns:
        shots = shots[shots["period"] != 5]

    pitch = Pitch(
        pitch_type="custom",
        pitch_length=PITCH_LENGTH_M,
        pitch_width=PITCH_WIDTH_M,
        line_color=t.pitch_line,
    )
    fig, ax = pitch.draw(figsize=(11, 7.2))

    for tid in sorted(shots["team_id"].dropna().unique()):
        sub = shots[shots["team_id"] == tid]
        color = t.team_color(str(tid), home_team_id)
        label = (team_names or {}).get(str(tid), str(tid))
        goals = sub[sub["result"] == "success"]
        others = sub[sub["result"] != "success"]
        if not others.empty:
            pitch.scatter(
                others["start_x"],
                others["start_y"],
                ax=ax,
                s=90,
                color=color,
                alpha=0.7,
                edgecolors="black",
                label=f"{label} shot",
                zorder=3,
            )
        for _, g in goals.iterrows():
            pitch.scatter(
                [g["start_x"]],
                [g["start_y"]],
                ax=ax,
                s=240,
                color=t.goal,
                edgecolors=color,
                linewidth=2.0,
                marker="*",
                zorder=5,
                label=f"{label} goal",
            )
            scorer = (player_names or {}).get(str(g.get("player_id")), "")
            if scorer:
                ax.annotate(
                    scorer,
                    xy=(g["start_x"], g["start_y"]),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "lw": 0.8, "alpha": 0.9},
                    zorder=6,
                )

    header = _match_header(shots, team_names, home_team_id)
    if title is None:
        title = f"Shots — {header}" if header else f"Shots ({len(shots)} total)"
    ax.set_title(title)

    if not shots.empty:
        handles, labels = ax.get_legend_handles_labels()
        seen: dict[str, int] = {}
        for i, lab in enumerate(labels):
            seen.setdefault(lab, i)
        ax.legend([handles[i] for i in seen.values()], list(seen.keys()), loc="upper left", fontsize=9)
    return fig  # type: ignore[no-any-return]
