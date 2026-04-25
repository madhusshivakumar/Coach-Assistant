"""Per-snapshot state trajectory for an episode.

Slice A's state is intentionally lightweight: positions, ball, and a few team-shape
aggregates. Pitch-control + OBSO surfaces (which dominate compute) are deferred to
Slice B's attribution layer — the cheap layer ships first and stands alone.

Coordinates are kept in canonical pitch frame (105 × 68 m, home attacks +x in H1).
``ball_x_oriented`` and the ``*_oriented`` team-shape stats flip x for teams attacking
left so "forward" always means "+x_oriented" — this makes downstream features
direction-symmetric without re-computing for each team.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_analysis.analytics.episodes.segmenter import EpisodeBoundary

PITCH_LENGTH_M: float = 105.0


def _select_snapshot_frames(
    tracking: pd.DataFrame,
    episode: EpisodeBoundary,
    snapshot_hz: float,
) -> list[int]:
    """Pick snapshot frame_ids uniformly across the episode at ``snapshot_hz``.

    Falls back to the episode's only frame for instantaneous (duration_s == 0)
    episodes. De-duplicates collisions while preserving order.
    """
    if episode.duration_s <= 0:
        return [episode.start_frame]
    n_snaps = max(2, round(episode.duration_s * snapshot_hz) + 1)
    target_times = np.linspace(episode.start_time_s, episode.end_time_s, n_snaps)
    frame_times = (
        tracking[(tracking["frame_id"] >= episode.start_frame) & (tracking["frame_id"] <= episode.end_frame)][
            ["frame_id", "time_seconds"]
        ]
        .drop_duplicates(subset="frame_id")
        .sort_values("time_seconds")
        .reset_index(drop=True)
    )
    if frame_times.empty:
        return []
    seen: set[int] = set()
    out: list[int] = []
    for t in target_times:
        idx = (frame_times["time_seconds"] - t).abs().idxmin()
        f = int(frame_times.loc[idx, "frame_id"])
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _orient_x(x: float, attacking_to_right: bool) -> float:
    return x if attacking_to_right else PITCH_LENGTH_M - x


def episode_state_trajectory(
    tracking: pd.DataFrame,
    episode: EpisodeBoundary,
    home_team_id: str,
    away_team_id: str,
    snapshot_hz: float = 2.0,
    attacking_to_right: bool = True,
) -> pd.DataFrame:
    """Compute the per-snapshot state DataFrame for one episode.

    Args:
        tracking: canonical tracking DataFrame for the whole match.
        episode: ``EpisodeBoundary`` from the segmenter.
        home_team_id, away_team_id: team_id values used in tracking.
        snapshot_hz: snapshots per second (2.0 = every 0.5 s).
        attacking_to_right: orient the ``*_oriented`` features so "forward" = +x.

    Returns:
        DataFrame with one row per snapshot. Empty if the episode has no usable
        frames (e.g. all snapshots fall in tracking gaps).
    """
    defending_team = away_team_id if episode.possession_team == home_team_id else home_team_id
    snap_frames = _select_snapshot_frames(tracking, episode, snapshot_hz=snapshot_hz)

    rows: list[dict[str, float | int | None]] = []
    for frame_id in snap_frames:
        sub = tracking[tracking["frame_id"] == frame_id]
        if sub.empty:
            continue

        time_s = float(sub["time_seconds"].iloc[0])
        ball = sub[sub["is_ball"] & sub["visible"]]
        if ball.empty:
            ball_x = ball_y = ball_speed = float("nan")
        else:
            ball_x = float(ball["x"].iloc[0])
            ball_y = float(ball["y"].iloc[0])
            ball_speed = float(np.hypot(ball["vx"].iloc[0], ball["vy"].iloc[0]))

        attackers = sub[(sub["team_id"] == episode.possession_team) & ~sub["is_ball"] & sub["visible"]]
        defenders = sub[(sub["team_id"] == defending_team) & ~sub["is_ball"] & sub["visible"]]

        atk_x_oriented = (
            attackers["x"].apply(lambda x: _orient_x(float(x), attacking_to_right))
            if not attackers.empty
            else pd.Series([], dtype=float)
        )
        def_x_oriented = (
            defenders["x"].apply(lambda x: _orient_x(float(x), attacking_to_right))
            if not defenders.empty
            else pd.Series([], dtype=float)
        )

        rows.append(
            {
                "episode_id": episode.episode_id,
                "frame_id": int(frame_id),
                "time_s": time_s,
                "rel_time_s": round(time_s - episode.start_time_s, 3),
                "ball_x": ball_x,
                "ball_y": ball_y,
                "ball_x_oriented": (_orient_x(ball_x, attacking_to_right) if not np.isnan(ball_x) else float("nan")),
                "ball_speed": ball_speed,
                "attackers_visible": len(attackers),
                "defenders_visible": len(defenders),
                "attackers_mean_x_oriented": (
                    float(atk_x_oriented.mean()) if not atk_x_oriented.empty else float("nan")
                ),
                "attackers_length": (
                    float(attackers["x"].max() - attackers["x"].min()) if len(attackers) >= 2 else float("nan")
                ),
                "attackers_width": (
                    float(attackers["y"].max() - attackers["y"].min()) if len(attackers) >= 2 else float("nan")
                ),
                "defenders_mean_x_oriented": (
                    float(def_x_oriented.mean()) if not def_x_oriented.empty else float("nan")
                ),
                # Defensive-line height in oriented coords = the deepest defender's x_oriented
                # (closest to their own goal). Smaller = a deeper line.
                "defenders_line_height_oriented": (
                    float(def_x_oriented.min()) if not def_x_oriented.empty else float("nan")
                ),
                "defenders_compactness_x": (float(defenders["x"].std()) if len(defenders) >= 2 else float("nan")),
            }
        )
    return pd.DataFrame(rows)
