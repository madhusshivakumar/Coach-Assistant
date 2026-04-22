"""Tracking normalisation: kloppy TrackingDataset -> canonical long-form DataFrame.

Output schema matches `football_analysis.data.validation.TrackingSchema`:
one row per (match_id, period, frame_id, player|ball), with metric 105x68 coords,
bottom-left origin, canonical home-attacks-L->R-in-H1 orientation, and computed
velocities via finite difference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M

if TYPE_CHECKING:
    from kloppy.domain.models.tracking import TrackingDataset


def _to_metric(x: float | None, y: float | None) -> tuple[float | None, float | None]:
    """Rescale Metrica's [0,1] top-left-origin coords to metric 105x68 bottom-left-origin.

    Metrica's native convention is y=0 at top, y=1 at bottom. We flip y so y=0 is at the
    bottom (canonical) and rescale both dimensions to metres.
    """
    if x is None or y is None:
        return None, None
    return float(x) * PITCH_LENGTH_M, (1.0 - float(y)) * PITCH_WIDTH_M


def tracking_dataset_to_long(dataset: TrackingDataset, match_id: str) -> pd.DataFrame:
    """Convert a kloppy `TrackingDataset` into the canonical long-form DataFrame.

    One row per (frame, entity). Entities are players OR the ball.

    Metrica's orientation already aligns with our canonical convention:
    home attacks L->R in H1 and R->L in H2 (teams swap ends at HT physically). So no
    extra x-flip is needed -- just the y-flip in `_to_metric`.

    Velocities are computed per-entity by finite difference over successive frames.
    """
    rows: list[dict[str, object]] = []
    for frame in dataset.frames:
        period = int(frame.period.id)
        frame_id = int(frame.frame_id)
        t = frame.timestamp
        time_seconds = t.total_seconds() if hasattr(t, "total_seconds") else float(t)

        # Ball row
        if frame.ball_coordinates is not None:
            bx, by = _to_metric(frame.ball_coordinates.x, frame.ball_coordinates.y)
            rows.append(
                {
                    "match_id": match_id,
                    "period": period,
                    "frame_id": frame_id,
                    "time_seconds": time_seconds,
                    "player_id": None,
                    "team_id": None,
                    "x": bx,
                    "y": by,
                    "is_ball": True,
                    "visible": bx is not None,
                }
            )

        # Player rows
        for player, pdata in frame.players_data.items():
            if pdata.coordinates is None:
                continue
            px, py = _to_metric(pdata.coordinates.x, pdata.coordinates.y)
            rows.append(
                {
                    "match_id": match_id,
                    "period": period,
                    "frame_id": frame_id,
                    "time_seconds": time_seconds,
                    "player_id": str(player.player_id),
                    "team_id": str(player.team.team_id),
                    "x": px,
                    "y": py,
                    "is_ball": False,
                    "visible": px is not None,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return _empty_with_velocities(df)
    return _attach_velocities(df)


def _empty_with_velocities(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("vx", "vy", "speed"):
        df[col] = pd.Series([], dtype="float64")
    return df


def _attach_velocities(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-entity vx, vy, speed via finite difference over frames."""
    df = df.sort_values(["match_id", "player_id", "period", "frame_id"], na_position="last").reset_index(drop=True)
    # player_id is nullable (ball); fill with 'BALL' sentinel for groupby
    df["_group_pid"] = df["player_id"].fillna("BALL")

    grouped = df.groupby(["match_id", "_group_pid", "period"], sort=False)
    dt = grouped["time_seconds"].diff()
    dx = grouped["x"].diff()
    dy = grouped["y"].diff()

    with np.errstate(divide="ignore", invalid="ignore"):
        vx = (dx / dt).astype("float64")
        vy = (dy / dt).astype("float64")
    speed = np.sqrt(vx.pow(2) + vy.pow(2))
    # First frame per group has NaN velocities -> treat as 0
    df["vx"] = vx.fillna(0.0)
    df["vy"] = vy.fillna(0.0)
    df["speed"] = speed.fillna(0.0)
    # Guard against any lingering non-finite (NaN came from dt=0 edges)
    for col in ("vx", "vy", "speed"):
        df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    df = df.drop(columns=["_group_pid"])
    return df
