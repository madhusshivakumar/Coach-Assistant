"""OBSO — Off-Ball Scoring Opportunity surface.

    OBSO(x, y) = P_home_control(x, y) * P_ball_arrives(x, y) * P_goal_from(x, y)

Values are in [0, 1] and represent the probability that **if** the attacking
team could instantaneously move the ball to (x, y), **and** that team controls
(x, y) when it arrives, **and** a shot is attempted, it results in a goal.

A player's off-ball value at a frame is the OBSO at their current pitch
location — a high number means they are standing in a genuinely dangerous
area, whether or not the ball ever reaches them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_analysis.analytics.pitch_control.scoring import goal_probability_grid
from football_analysis.analytics.pitch_control.spearman import (
    DEFAULT_COLS,
    DEFAULT_ROWS,
    PitchControlFrame,
    _grid,
    compute_frame_from_tracking,
)
from football_analysis.analytics.pitch_control.transition import (
    DEFAULT_PASS_SIGMA_M,
    ball_arrival_probability,
)


@dataclass(frozen=True)
class OBSOFrame:
    """OBSO surface plus the factor decomposition for inspection/debugging."""

    obso: np.ndarray  # shape (rows, cols); attacking team's OBSO
    control: np.ndarray  # shape (rows, cols); attacking team's pitch-control
    arrival: np.ndarray  # shape (rows, cols); ball-arrival probability
    goal: np.ndarray  # shape (rows, cols); goal probability from each cell
    xs: np.ndarray
    ys: np.ndarray


def compute_obso_frame(
    tracking: pd.DataFrame,
    frame_id: int,
    attacking_team_id: str,
    defending_team_id: str,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    pass_sigma: float = DEFAULT_PASS_SIGMA_M,
) -> OBSOFrame:
    """Compute OBSO for a single frame.

    `attacking_team_id` is the team whose off-ball value we want — the pitch-control
    surface is oriented so 1.0 = that team controls the cell.
    """
    # Pitch control: compute_frame_from_tracking returns P(home controls). Flip it
    # so we always measure P(attacking team controls).
    pc: PitchControlFrame = compute_frame_from_tracking(
        tracking,
        frame_id=frame_id,
        home_team_id=attacking_team_id,
        away_team_id=defending_team_id,
        rows=rows,
        cols=cols,
    )
    attack_control = pc.home_control

    ball = tracking[(tracking["frame_id"] == frame_id) & tracking["is_ball"] & tracking["visible"]]
    if ball.empty:
        # No ball position — assume equally likely anywhere on the pitch
        xs, ys, _ = _grid(rows, cols)
        arrival = np.ones((rows, cols), dtype=np.float64)
    else:
        xs, ys, _ = _grid(rows, cols)
        bx = float(ball.iloc[0]["x"])
        by = float(ball.iloc[0]["y"])
        arrival = ball_arrival_probability(xs, ys, bx, by, sigma=pass_sigma)

    goal_surface = goal_probability_grid(xs, ys)

    obso = attack_control * arrival * goal_surface
    return OBSOFrame(
        obso=np.asarray(obso, dtype=np.float64),
        control=attack_control,
        arrival=arrival,
        goal=goal_surface,
        xs=xs,
        ys=ys,
    )


def per_player_obso(
    tracking: pd.DataFrame,
    frame_id: int,
    attacking_team_id: str,
    defending_team_id: str,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
) -> pd.DataFrame:
    """Return each attacking player's OBSO at their current location in the frame.

    Output columns: player_id, x, y, obso, control, arrival, goal. Ranked descending
    by `obso`.
    """
    surface = compute_obso_frame(
        tracking,
        frame_id,
        attacking_team_id=attacking_team_id,
        defending_team_id=defending_team_id,
        rows=rows,
        cols=cols,
    )

    attackers = tracking[
        (tracking["frame_id"] == frame_id)
        & (tracking["team_id"] == attacking_team_id)
        & (~tracking["is_ball"])
        & (tracking["visible"])
    ].copy()
    if attackers.empty:
        return pd.DataFrame(columns=["player_id", "x", "y", "obso", "control", "arrival", "goal"])

    # Nearest-cell lookup
    col_idx = np.clip(np.searchsorted(surface.xs, attackers["x"].to_numpy()) - 1, 0, cols - 1)
    row_idx = np.clip(np.searchsorted(surface.ys, attackers["y"].to_numpy()) - 1, 0, rows - 1)

    attackers["obso"] = surface.obso[row_idx, col_idx]
    attackers["control"] = surface.control[row_idx, col_idx]
    attackers["arrival"] = surface.arrival[row_idx, col_idx]
    attackers["goal"] = surface.goal[row_idx, col_idx]

    return (
        attackers[["player_id", "x", "y", "obso", "control", "arrival", "goal"]]
        .sort_values("obso", ascending=False)
        .reset_index(drop=True)
    )
