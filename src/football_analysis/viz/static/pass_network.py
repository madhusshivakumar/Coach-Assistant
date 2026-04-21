"""Pass network — nodes = average on-ball position per player, edges = pass volume."""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure
from mplsoccer import Pitch

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M
from football_analysis.viz.theme import DEFAULT_THEME, Theme


def plot_pass_network(
    events: pd.DataFrame,
    team_id: str,
    min_passes_edge: int = 3,
    title: str | None = None,
    theme: Theme | None = None,
) -> Figure:
    """Render a pass network for a single team.

    Node i position = mean (start_x, start_y) of on-ball actions involving player i.
    Edge (i, j) weight = number of successful passes from i to j (we proxy `receiver` via
    the player of the next on-ball action by the same team).
    """
    t = theme or DEFAULT_THEME
    team_events = events[events["team_id"] == team_id].copy().reset_index(drop=True)
    if team_events.empty:
        return _empty_pitch("No passes", theme=t)

    # Node positions: average start location per player across all their on-ball actions
    pos = (
        team_events.dropna(subset=["start_x", "start_y", "player_id"])
        .groupby("player_id")[["start_x", "start_y"]]
        .mean()
    )
    # Touch counts for node sizing
    touch_count = team_events.dropna(subset=["player_id"]).groupby("player_id").size()

    # Build edges via successive same-team actions
    passes = team_events[
        (team_events["action_type"] == "pass") & (team_events["result"] == "success") & team_events["player_id"].notna()
    ].copy()
    if passes.empty:
        return _empty_pitch(f"No successful passes — team {team_id}", theme=t)

    # Sort chronologically; receiver = next non-pass event by same team immediately after
    all_team_sorted = team_events.sort_values(["period", "time_seconds"]).reset_index(drop=True)
    player_by_idx = all_team_sorted["player_id"].tolist()
    idx_map = {row_id: i for i, row_id in enumerate(all_team_sorted.index)}  # identity but kept for clarity
    del idx_map  # not needed — we use iloc order
    edges: dict[tuple[str, str], int] = {}
    for i, row in all_team_sorted.iterrows():
        if row["action_type"] != "pass" or row["result"] != "success":
            continue
        if i + 1 >= len(all_team_sorted):
            continue
        passer = row["player_id"]
        receiver = player_by_idx[i + 1]
        if pd.isna(passer) or pd.isna(receiver) or passer == receiver:
            continue
        edges[(passer, receiver)] = edges.get((passer, receiver), 0) + 1

    pitch = Pitch(
        pitch_type="custom",
        pitch_length=PITCH_LENGTH_M,
        pitch_width=PITCH_WIDTH_M,
        line_color=t.pitch_line,
    )
    fig, ax = pitch.draw(figsize=(10, 7))

    # Draw edges
    for (a, b), w in edges.items():
        if w < min_passes_edge:
            continue
        if a not in pos.index or b not in pos.index:
            continue
        pitch.lines(
            pos.loc[a, "start_x"],
            pos.loc[a, "start_y"],
            pos.loc[b, "start_x"],
            pos.loc[b, "start_y"],
            ax=ax,
            color=t.home,
            lw=max(0.5, w / 3.0),
            alpha=0.6,
        )

    # Draw nodes. Node area scales with touches but is clamped so a 60-touch midfielder
    # doesn't swallow the pitch on a full 90-minute sample.
    if not pos.empty:
        tc = touch_count.reindex(pos.index).fillna(1)
        max_touches = max(1, tc.max())
        sizes = [150 + 450 * (tc[p] / max_touches) for p in pos.index]
        pitch.scatter(
            pos["start_x"],
            pos["start_y"],
            ax=ax,
            s=sizes,
            color=t.home,
            edgecolors="black",
            zorder=3,
            alpha=0.85,
        )
        for p in pos.index:
            ax.annotate(
                str(p)[-4:],
                xy=(pos.loc[p, "start_x"], pos.loc[p, "start_y"]),
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
                zorder=4,
            )

    ax.set_title(title or f"Pass network — team {team_id} ({sum(edges.values())} edges)")
    return fig  # type: ignore[no-any-return]


def _empty_pitch(msg: str, theme: Theme) -> Figure:
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=PITCH_LENGTH_M,
        pitch_width=PITCH_WIDTH_M,
        line_color=theme.pitch_line,
    )
    fig, ax = pitch.draw(figsize=(10, 7))
    ax.set_title(msg)
    return fig  # type: ignore[no-any-return]
