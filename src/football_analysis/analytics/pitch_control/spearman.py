"""Pitch control — Spearman-style logistic over minimum time-to-intercept.

For each grid cell, each team's "time to arrive" is the minimum over its players.
The team with the lower time dominates, softened by a logistic around their
difference:

    P_home(x, y) = 1 / (1 + exp((t_home - t_away) / sigma))

Values in [0, 1]: 1.0 means home almost certainly controls that cell, 0.0 means
away. By construction P_home + P_away = 1 (they are complementary probabilities
on a single Bernoulli).

The grid is 104 columns x 68 rows by default (1 metre per cell) and assumes the
canonical 105 x 68 m pitch with origin at the bottom-left.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M
from football_analysis.analytics.pitch_control.motion import (
    DEFAULT_MAX_SPEED,
    DEFAULT_REACTION_TIME,
    time_to_intercept,
)

DEFAULT_SIGMA: float = 0.45  # seconds; controls sharpness of the logistic

# How many metres-per-cell for the default grid.
DEFAULT_COLS: int = 104
DEFAULT_ROWS: int = 68


@dataclass(frozen=True)
class PitchControlFrame:
    """A single-frame pitch-control surface plus the grid axes."""

    home_control: np.ndarray  # shape (rows, cols), P(home controls cell)
    xs: np.ndarray  # shape (cols,), x centre of each column
    ys: np.ndarray  # shape (rows,), y centre of each row
    t_home: np.ndarray  # shape (rows, cols), min home time-to-intercept
    t_away: np.ndarray  # shape (rows, cols), min away time-to-intercept


def _grid(
    rows: int, cols: int, pitch_length: float = PITCH_LENGTH_M, pitch_width: float = PITCH_WIDTH_M
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(xs, ys, targets)` where `targets[r, c] == (xs[c], ys[r])`."""
    xs = np.linspace(pitch_length / cols / 2, pitch_length - pitch_length / cols / 2, cols)
    ys = np.linspace(pitch_width / rows / 2, pitch_width - pitch_width / rows / 2, rows)
    xx, yy = np.meshgrid(xs, ys)
    targets = np.stack([xx, yy], axis=-1)  # (rows, cols, 2)
    return xs, ys, targets


def compute_frame(
    home_positions: np.ndarray,
    home_velocities: np.ndarray,
    away_positions: np.ndarray,
    away_velocities: np.ndarray,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    sigma: float = DEFAULT_SIGMA,
    reaction_time: float = DEFAULT_REACTION_TIME,
    max_speed: float = DEFAULT_MAX_SPEED,
) -> PitchControlFrame:
    """Compute the pitch-control surface for a single frame.

    Edge cases:
    - If a team has no players in the frame, their time-to-intercept is +inf
      across the pitch and the other team controls everything.
    - If both teams are empty the surface is 0.5 (maximum uncertainty).
    """
    xs, ys, targets = _grid(rows, cols)

    def _min_time(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        if pos.size == 0:
            return np.full((rows, cols), np.inf, dtype=np.float64)
        t = time_to_intercept(pos, vel, targets, reaction_time=reaction_time, max_speed=max_speed)
        return np.asarray(t.min(axis=0))

    t_home = _min_time(home_positions, home_velocities)
    t_away = _min_time(away_positions, away_velocities)

    # Sigmoid over the time differential. Clip to [-30, 30] standard deviations
    # to avoid numpy overflow warnings for astronomically distant cells.
    both_inf = np.isinf(t_home) & np.isinf(t_away)
    with np.errstate(over="ignore"):
        diff = np.clip((t_home - t_away) / sigma, -30.0, 30.0)
        hc: np.ndarray = 1.0 / (1.0 + np.exp(diff))
    # Saturate: if only one side has players the sigmoid of +/-inf already
    # gave 0 or 1; if both empty, fall back to 0.5.
    home_control = np.where(both_inf, 0.5, hc)

    return PitchControlFrame(
        home_control=np.asarray(home_control, dtype=np.float64),
        xs=xs,
        ys=ys,
        t_home=t_home,
        t_away=t_away,
    )


def compute_frame_from_tracking(
    tracking: pd.DataFrame,
    frame_id: int,
    home_team_id: str,
    away_team_id: str,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    sigma: float = DEFAULT_SIGMA,
    reaction_time: float = DEFAULT_REACTION_TIME,
    max_speed: float = DEFAULT_MAX_SPEED,
) -> PitchControlFrame:
    """Convenience wrapper: pull the right rows from a canonical tracking DataFrame."""
    frame = tracking[tracking["frame_id"] == frame_id]
    if frame.empty:
        raise ValueError(f"no rows for frame_id={frame_id}")

    def _stack(team_id: str) -> tuple[np.ndarray, np.ndarray]:
        sub = frame[(frame["team_id"] == team_id) & (~frame["is_ball"]) & (frame["visible"])]
        if sub.empty:
            return np.empty((0, 2)), np.empty((0, 2))
        pos = sub[["x", "y"]].to_numpy(dtype=np.float64)
        vel = sub[["vx", "vy"]].to_numpy(dtype=np.float64)
        return pos, vel

    home_pos, home_vel = _stack(home_team_id)
    away_pos, away_vel = _stack(away_team_id)
    return compute_frame(
        home_pos,
        home_vel,
        away_pos,
        away_vel,
        rows=rows,
        cols=cols,
        sigma=sigma,
        reaction_time=reaction_time,
        max_speed=max_speed,
    )
