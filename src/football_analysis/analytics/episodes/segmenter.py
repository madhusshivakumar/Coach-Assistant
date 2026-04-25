"""Episode segmentation — split a match into possession episodes.

An episode is a contiguous run of frames where one team holds the ball. It ends at
either:

1. **Turnover** — the other team gains possession.
2. **Out-of-play** — a ``MIN_DEAD_FRAMES``-frame run of ball-not-visible (the canonical
   dead-ball signal in the Metrica tracking schema). Single-frame ball glitches must
   not break episodes; that's pure tracking noise.
3. **Match end** — the final episode just runs out.

Possession identity is taken from ``classify_frames`` (Phase 3A), which already smooths
spurious flips with a sticky-frames heuristic. Within a live (non-long-dead) run, short
``possession_team is None`` gaps are forward-filled so contested-ball blips don't
fragment otherwise-coherent episodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Minimum consecutive ball-not-visible frames to count as a dead-ball break.
# Below this threshold (default 5 frames ~= 0.2 s at 25 Hz), the gap is treated as
# tracking noise and the episode continues across it.
MIN_DEAD_FRAMES_DEFAULT: int = 5


@dataclass(frozen=True)
class EpisodeBoundary:
    """Bounds of one possession episode, before state/outcome enrichment."""

    episode_id: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    possession_team: str
    end_reason: str  # "possession_change" | "out_of_play" | "match_end"


def _ball_visibility_per_frame(tracking: pd.DataFrame) -> pd.Series:
    """For each frame_id, True iff at least one ball row is marked visible."""
    ball = tracking[tracking["is_ball"]]
    if ball.empty:
        return pd.Series(dtype=bool)
    return ball.groupby("frame_id")["visible"].any()


def segment_episodes(
    classified: pd.DataFrame,
    tracking: pd.DataFrame,
    min_dead_frames: int = MIN_DEAD_FRAMES_DEFAULT,
) -> list[EpisodeBoundary]:
    """Build the ordered list of possession episodes for a match.

    Args:
        classified: output of ``classify_frames`` — must carry ``frame_id``,
            ``time_seconds``, and ``possession_team`` (the latter may be None for
            contested frames; that's tolerated).
        tracking: canonical tracking DataFrame; only the ball's ``visible`` column
            is consumed here, to detect dead-ball windows.
        min_dead_frames: a window of >= this many consecutive ball-not-visible
            frames triggers an episode break with ``end_reason="out_of_play"``.

    Returns:
        ``EpisodeBoundary`` list ordered by ``start_frame``. Frames inside long
        dead-ball windows belong to no episode.
    """
    if classified.empty:
        return []

    df = classified.sort_values("frame_id").reset_index(drop=True).copy()

    # 1. Per-frame ball visibility.
    ball_vis = _ball_visibility_per_frame(tracking)
    df["ball_visible"] = df["frame_id"].map(ball_vis).fillna(False).astype(bool)
    df["dead"] = ~df["ball_visible"]

    # 2. Run-length encode dead-vs-alive runs; mark only LONG dead runs as breaks.
    df["dead_run"] = (df["dead"] != df["dead"].shift()).cumsum()
    dead_lengths = df[df["dead"]].groupby("dead_run").size()
    long_dead_runs = set(dead_lengths[dead_lengths >= min_dead_frames].index)
    df["is_long_dead"] = df["dead"] & df["dead_run"].isin(long_dead_runs)

    # 3. Effective possession: NaN inside long-dead windows; otherwise forward-fill
    # short None gaps WITHIN each non-long-dead run (so a brief contested-ball blip
    # doesn't fragment a coherent possession). Each non-long-dead "run" is a separate
    # ffill scope; we never carry possession across a long-dead boundary.
    df["run_group"] = (df["is_long_dead"] != df["is_long_dead"].shift()).cumsum()
    df["effective_pos"] = df.groupby("run_group", group_keys=False)["possession_team"].ffill()
    df.loc[df["is_long_dead"], "effective_pos"] = pd.NA

    # 4. Episode = each contiguous run of (effective_pos == one team).
    df["ep_change"] = (df["effective_pos"] != df["effective_pos"].shift()).cumsum()

    episodes: list[EpisodeBoundary] = []
    counter = 0
    for _, sub in df.groupby("ep_change", sort=True):
        team_value = sub["effective_pos"].iloc[0]
        if pd.isna(team_value):
            continue  # dead-ball gap or undecided run — not an episode

        # End reason: peek at the row after this run.
        last_idx = sub.index[-1]
        next_idx = last_idx + 1
        if next_idx >= len(df):
            end_reason = "match_end"
        elif df.loc[next_idx, "is_long_dead"]:
            end_reason = "out_of_play"
        else:
            end_reason = "possession_change"

        start_t = float(sub["time_seconds"].iloc[0])
        end_t = float(sub["time_seconds"].iloc[-1])
        episodes.append(
            EpisodeBoundary(
                episode_id=counter,
                start_frame=int(sub["frame_id"].iloc[0]),
                end_frame=int(sub["frame_id"].iloc[-1]),
                start_time_s=start_t,
                end_time_s=end_t,
                duration_s=round(end_t - start_t, 3),
                possession_team=str(team_value),
                end_reason=end_reason,
            )
        )
        counter += 1
    return episodes
