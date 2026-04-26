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
from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M
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
    use_gpu: bool = False,
) -> pd.DataFrame:
    """OBSO-max + arg-max position at every snapshot frame of the episode.

    Returns a DataFrame with one row per snapshot:
    ``frame_id, time_s, rel_time_s, obso_max, obso_argmax_x, obso_argmax_y``.
    Empty if the state_trajectory is empty.

    When ``use_gpu=True`` and PyTorch + CUDA are available, all snapshots in
    the episode are batched into a single device call via
    ``compute_obso_batch_torch`` — typically ~5x faster on CPU and 10-15x on
    GPU for episodes with 10+ snapshots. Falls back to the per-frame numpy
    path silently when torch isn't installed.
    """
    states = record.state_trajectory
    if states.empty:
        return pd.DataFrame(columns=["frame_id", "time_s", "rel_time_s", "obso_max", "obso_argmax_x", "obso_argmax_y"])

    attacking_team = record.boundary.possession_team
    defending_team = away_team_id if attacking_team == home_team_id else home_team_id

    if use_gpu:
        try:
            return _compute_obso_trajectory_gpu(
                states,
                tracking,
                attacking_team,
                defending_team,
                rows,
                cols,
            )
        except ImportError:
            pass  # torch not installed → fall through to numpy path

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


def _compute_obso_trajectory_gpu(
    states: pd.DataFrame,
    tracking: pd.DataFrame,
    attacking_team: str,
    defending_team: str,
    rows: int,
    cols: int,
) -> pd.DataFrame:
    """Batched OBSO across all snapshot frames of an episode."""
    # Lazy-import torch path so non-GPU users don't pay for it. ``noqa`` because
    # this is intentionally inside the function body — top-level import would
    # force torch to load even when use_gpu=False.
    from football_analysis.analytics.pitch_control.obso_gpu import (  # noqa: PLC0415
        compute_obso_batch_torch,
    )

    snap_frames = [int(f) for f in states["frame_id"].astype(int)]
    home_pos: list[np.ndarray] = []
    home_vel: list[np.ndarray] = []
    away_pos: list[np.ndarray] = []
    away_vel: list[np.ndarray] = []
    ball_xy = np.zeros((len(snap_frames), 2), dtype=np.float64)

    for i, frame_id in enumerate(snap_frames):
        sub = tracking[tracking["frame_id"] == frame_id]
        atk = sub[(sub["team_id"] == attacking_team) & ~sub["is_ball"] & sub["visible"]]
        defn = sub[(sub["team_id"] == defending_team) & ~sub["is_ball"] & sub["visible"]]
        home_pos.append(atk[["x", "y"]].to_numpy(dtype=np.float64) if not atk.empty else np.empty((0, 2)))
        home_vel.append(atk[["vx", "vy"]].to_numpy(dtype=np.float64) if not atk.empty else np.empty((0, 2)))
        away_pos.append(defn[["x", "y"]].to_numpy(dtype=np.float64) if not defn.empty else np.empty((0, 2)))
        away_vel.append(defn[["vx", "vy"]].to_numpy(dtype=np.float64) if not defn.empty else np.empty((0, 2)))
        ball = sub[sub["is_ball"] & sub["visible"]]
        if ball.empty:
            ball_xy[i] = (PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2)
        else:
            ball_xy[i] = (float(ball.iloc[0]["x"]), float(ball.iloc[0]["y"]))

    batched = compute_obso_batch_torch(
        home_positions=home_pos,
        home_velocities=home_vel,
        away_positions=away_pos,
        away_velocities=away_vel,
        ball_xy=ball_xy,
        attacking_is_home=np.ones(len(snap_frames), dtype=np.bool_),
        rows=rows,
        cols=cols,
    )

    rows_out: list[dict[str, float | int]] = []
    for i, frame_id in enumerate(snap_frames):
        obso_i = batched.obso[i]
        obso_max = float(np.max(obso_i))
        flat_idx = int(np.argmax(obso_i))
        ri, ci = np.unravel_index(flat_idx, obso_i.shape)
        rows_out.append(
            {
                "frame_id": frame_id,
                "time_s": float(states.iloc[i]["time_s"]),
                "rel_time_s": float(states.iloc[i]["rel_time_s"]),
                "obso_max": obso_max,
                "obso_argmax_x": float(batched.xs[ci]),
                "obso_argmax_y": float(batched.ys[ri]),
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
