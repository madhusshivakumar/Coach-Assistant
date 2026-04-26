"""OBSO time-series across an episode + decisive-moment detection.

Slice B's attribution looks at the peak frame only. To answer "when did the play
start to work?" we need OBSO-max at *every* snapshot through the episode, then
the earliest frame where OBSO-max first crossed a meaningful fraction of the
peak. That frame is the **decisive moment** — a much better answer than "first
ball into the final third" for plays that begin already inside the final third.

Cost: one ``compute_obso_frame`` call per snapshot. At 2 Hz over a ~6 s episode
that's ~12 OBSO calls, each ~0.014 s on a coarse grid — sub-second per episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_analysis.analytics.episodes.contribution import (
    DEFAULT_ATTRIBUTION_COLS,
    DEFAULT_ATTRIBUTION_ROWS,
)
from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.pitch_control.obso import compute_obso_frame


@dataclass(frozen=True)
class DecisiveMoment:
    """The earliest frame where OBSO-max first crossed ``threshold_pct * peak``."""

    frame_id: int
    time_s: float
    rel_time_s: float
    obso_at_decisive: float
    peak_obso: float
    threshold_pct: float


def compute_obso_trajectory(
    record: EpisodeRecord,
    tracking: pd.DataFrame,
    home_team_id: str,
    away_team_id: str,
    rows: int = DEFAULT_ATTRIBUTION_ROWS,
    cols: int = DEFAULT_ATTRIBUTION_COLS,
) -> pd.DataFrame:
    """OBSO-max + arg-max position at every snapshot frame of the episode.

    Returns a DataFrame with one row per snapshot:
    ``frame_id, time_s, rel_time_s, obso_max, obso_argmax_x, obso_argmax_y``.
    Empty if the state_trajectory is empty.
    """
    states = record.state_trajectory
    if states.empty:
        return pd.DataFrame(columns=["frame_id", "time_s", "rel_time_s", "obso_max", "obso_argmax_x", "obso_argmax_y"])

    attacking_team = record.boundary.possession_team
    defending_team = away_team_id if attacking_team == home_team_id else home_team_id

    rows_out: list[dict[str, float | int]] = []
    for _, snap in states.iterrows():
        frame_id = int(snap["frame_id"])
        try:
            of = compute_obso_frame(
                tracking,
                frame_id,
                attacking_team_id=attacking_team,
                defending_team_id=defending_team,
                rows=rows,
                cols=cols,
            )
        except Exception:
            continue
        obso_max = float(np.max(of.obso))
        flat_idx = int(np.argmax(of.obso))
        ri, ci = np.unravel_index(flat_idx, of.obso.shape)
        rows_out.append(
            {
                "frame_id": frame_id,
                "time_s": float(snap["time_s"]),
                "rel_time_s": float(snap["rel_time_s"]),
                "obso_max": obso_max,
                "obso_argmax_x": float(of.xs[ci]),
                "obso_argmax_y": float(of.ys[ri]),
            }
        )
    return pd.DataFrame(rows_out)


def find_decisive_moment(
    obso_trajectory: pd.DataFrame,
    threshold_pct: float = 0.5,
) -> DecisiveMoment | None:
    """First frame where ``obso_max`` first crossed ``threshold_pct * peak``.

    Returns None if the trajectory is empty or peak is non-positive.
    """
    if obso_trajectory.empty:
        return None
    peak = float(obso_trajectory["obso_max"].max())
    if peak <= 0.0:
        return None
    threshold = threshold_pct * peak
    crossed = obso_trajectory[obso_trajectory["obso_max"] >= threshold]
    if crossed.empty:
        return None
    first = crossed.iloc[0]
    return DecisiveMoment(
        frame_id=int(first["frame_id"]),
        time_s=float(first["time_s"]),
        rel_time_s=float(first["rel_time_s"]),
        obso_at_decisive=float(first["obso_max"]),
        peak_obso=peak,
        threshold_pct=threshold_pct,
    )
