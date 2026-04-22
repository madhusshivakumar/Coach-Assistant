"""Pass network with team colours, player names, and directional arrows."""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure
from mplsoccer import Pitch

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M
from football_analysis.viz.theme import DEFAULT_THEME, Theme

_COMPOUND_PREFIXES = {
    "di",
    "de",
    "del",
    "della",
    "la",
    "le",
    "van",
    "von",
    "da",
    "das",
    "do",
    "dos",
    "mc",
    "mac",
    "al",
    "el",
    "bin",
    "ben",
    "ter",
    "ten",
}


def _short_name(full: str) -> str:
    """Collapse a player name to something that fits on a node label.

    Handles compound surnames like "Di María", "De Paul", "Van Dijk" — if the
    second-to-last token is a known compound prefix, glue it onto the last token.
    Otherwise just take the last token. Capped at 14 chars.
    """
    parts = [p for p in full.strip().split() if p]
    if not parts:
        return full[:14]
    if len(parts) >= 2 and parts[-2].lower().rstrip(".") in _COMPOUND_PREFIXES:
        return f"{parts[-2]} {parts[-1]}"[:14]
    return parts[-1][:14]


def plot_pass_network(
    events: pd.DataFrame,
    team_id: str,
    min_passes_edge: int = 3,
    min_touches_node: int = 10,
    title: str | None = None,
    theme: Theme | None = None,
    home_team_id: str | None = None,
    team_names: dict[str, str] | None = None,
    player_names: dict[str, str] | None = None,
) -> Figure:
    """Render a pass network for a single team.

    Node i position = mean (start_x, start_y) of on-ball actions involving player i.
    Edge (i -> j) = number of successful passes where the immediate next same-team action
    is attributed to player j. Rendered as directional arrows with widths proportional to
    edge weight (capped so a dominant pairing doesn't overwhelm the plot).
    """
    t = theme or DEFAULT_THEME
    color = t.team_color(team_id, home_team_id)
    team_label = (team_names or {}).get(team_id, team_id)

    team_events = events[events["team_id"] == team_id].copy().reset_index(drop=True)
    if team_events.empty:
        return _empty_pitch(f"No events — {team_label}", theme=t)

    # Node positions: average start location per player across all their on-ball actions
    pos = (
        team_events.dropna(subset=["start_x", "start_y", "player_id"])
        .groupby("player_id")[["start_x", "start_y"]]
        .mean()
    )
    touch_count = team_events.dropna(subset=["player_id"]).groupby("player_id").size()

    # Keep only players with enough touches to be meaningful — drops brief subs so
    # labels don't pile up in the middle of the pitch. Auto-relax threshold if we'd
    # end up with fewer than the typical 11-start XI.
    keep_players = touch_count[touch_count >= min_touches_node].index
    if len(keep_players) < 11 and not touch_count.empty:
        keep_players = touch_count.sort_values(ascending=False).head(14).index
    pos = pos.loc[pos.index.intersection(keep_players)]

    passes = team_events[
        (team_events["action_type"] == "pass") & (team_events["result"] == "success") & team_events["player_id"].notna()
    ].copy()
    if passes.empty:
        return _empty_pitch(f"No successful passes — {team_label}", theme=t)

    # Build directed edges: receiver = next same-team event after a successful pass
    sorted_events = team_events.sort_values(["period", "time_seconds"]).reset_index(drop=True)
    player_by_idx = sorted_events["player_id"].tolist()
    edges: dict[tuple[str, str], int] = {}
    keep_set = set(pos.index)
    for i, row in sorted_events.iterrows():
        if row["action_type"] != "pass" or row["result"] != "success":
            continue
        if i + 1 >= len(sorted_events):
            continue
        passer = row["player_id"]
        receiver = player_by_idx[i + 1]
        if pd.isna(passer) or pd.isna(receiver) or passer == receiver:
            continue
        if passer not in keep_set or receiver not in keep_set:
            continue
        edges[(passer, receiver)] = edges.get((passer, receiver), 0) + 1

    pitch = Pitch(
        pitch_type="custom",
        pitch_length=PITCH_LENGTH_M,
        pitch_width=PITCH_WIDTH_M,
        line_color=t.pitch_line,
    )
    fig, ax = pitch.draw(figsize=(11, 7.2))

    # Directional arrows
    max_edge = max((w for w in edges.values()), default=1)
    for (a, b), w in edges.items():
        if w < min_passes_edge or a not in pos.index or b not in pos.index:
            continue
        width = 1.0 + 3.5 * (w / max_edge)
        pitch.arrows(
            pos.loc[a, "start_x"],
            pos.loc[a, "start_y"],
            pos.loc[b, "start_x"],
            pos.loc[b, "start_y"],
            ax=ax,
            color=color,
            width=width,
            alpha=0.55,
            headwidth=6,
            headlength=6,
            zorder=2,
        )

    # Nodes
    if not pos.empty:
        tc = touch_count.reindex(pos.index).fillna(1)
        max_touches = max(1, tc.max())
        sizes = [200 + 600 * (tc[p] / max_touches) for p in pos.index]
        pitch.scatter(
            pos["start_x"],
            pos["start_y"],
            ax=ax,
            s=sizes,
            color=color,
            edgecolors="black",
            zorder=3,
            alpha=0.9,
        )
        # Offset labels above or below alternately so adjacent nodes don't collide
        rank_by_y = pos["start_y"].rank(method="first").astype(int)
        for p in pos.index:
            full = (player_names or {}).get(str(p), str(p))
            label = _short_name(full)
            offset_y = -16 if rank_by_y[p] % 2 == 0 else 18
            va = "top" if offset_y < 0 else "bottom"
            ax.annotate(
                label,
                xy=(pos.loc[p, "start_x"], pos.loc[p, "start_y"]),
                xytext=(0, offset_y),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=8,
                color="black",
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.85},
                zorder=4,
            )

    n_edges_drawn = sum(1 for w in edges.values() if w >= min_passes_edge)
    default_title = (
        f"Pass network — {team_label}  "
        f"({len(pos)} players ≥{min_touches_node} touches, "
        f"{n_edges_drawn} edges ≥{min_passes_edge} passes)"
    )
    ax.set_title(title or default_title)
    return fig  # type: ignore[no-any-return]


def _empty_pitch(msg: str, theme: Theme) -> Figure:
    pitch = Pitch(
        pitch_type="custom",
        pitch_length=PITCH_LENGTH_M,
        pitch_width=PITCH_WIDTH_M,
        line_color=theme.pitch_line,
    )
    fig, ax = pitch.draw(figsize=(11, 7.2))
    ax.set_title(msg)
    return fig  # type: ignore[no-any-return]
