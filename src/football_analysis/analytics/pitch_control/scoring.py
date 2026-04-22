"""Naive goal-probability surface for OBSO.

Per Spearman 2018 ("Beyond Expected Goals"), OBSO factorises as:

    OBSO(x, y) = P_control(x, y) * P_ball_arrives(x, y) * P_goal_from(x, y)

This module provides the third factor: the probability that a shot from a given
pitch location results in a goal. It's intentionally *naive* — a full xG model
needs a training corpus and features (body part, pressure, set-piece context).
For a Phase-2 proof-of-life we fit a logistic over two geometric features,
distance to goal and the viewing angle from which the goal-mouth is seen,
calibrated so the output is in a realistic xG range (~0.30 near the penalty
spot, near 0.0 at halfway).

    P_goal = sigmoid(b0 + b1 * distance + b2 * -angle)

When a full StatsBomb-trained xG model lands later, swap this out — the
interface (DataFrame or (H,W) array) is stable.
"""

from __future__ import annotations

import numpy as np

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M

# Goal geometry: centre of the goal at x=PITCH_LENGTH, y=PITCH_WIDTH/2.
# Goal posts 7.32 m apart, so the goal-mouth spans y = 30.34 .. 37.66 m.
GOAL_X: float = PITCH_LENGTH_M
GOAL_CENTER_Y: float = PITCH_WIDTH_M / 2.0
GOAL_WIDTH: float = 7.32

# Logistic coefficients, calibrated so penalty-spot distance (11 m, centre) yields
# ~0.30 goal probability, long-range (30 m) drops to ~0.03. Deliberately gentle —
# a trained xG would have a steeper distance curve.
_INTERCEPT: float = 1.2
_BETA_DIST: float = -0.12
_BETA_ANGLE: float = 1.5


def _shot_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Angle (radians) subtended by the goal-mouth from the shot location.

    Wider angle = easier shot. We use the interior angle between lines from
    (x, y) to the two goalposts.
    """
    post_left = np.array([GOAL_X, GOAL_CENTER_Y - GOAL_WIDTH / 2.0])
    post_right = np.array([GOAL_X, GOAL_CENTER_Y + GOAL_WIDTH / 2.0])

    dx_l, dy_l = post_left[0] - x, post_left[1] - y
    dx_r, dy_r = post_right[0] - x, post_right[1] - y

    dot = dx_l * dx_r + dy_l * dy_r
    norm_l = np.sqrt(dx_l**2 + dy_l**2)
    norm_r = np.sqrt(dx_r**2 + dy_r**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.clip(dot / (norm_l * norm_r + 1e-9), -1.0, 1.0)
    return np.asarray(np.arccos(cos), dtype=np.float64)


def goal_probability(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return the naive goal probability for shots taken at (x, y).

    `x`, `y` may be arbitrary broadcast-compatible arrays of metres. Output has the
    same shape and is clipped to [0, 1].
    """
    distance = np.sqrt((GOAL_X - x) ** 2 + (GOAL_CENTER_Y - y) ** 2)
    angle = _shot_angle(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    # Shots from behind the goal line (x > GOAL_X) are physically impossible — force zero.
    logit = _INTERCEPT + _BETA_DIST * distance + _BETA_ANGLE * angle
    prob = 1.0 / (1.0 + np.exp(-logit))
    prob = np.where(x > GOAL_X, 0.0, prob)
    return np.asarray(np.clip(prob, 0.0, 1.0), dtype=np.float64)


def goal_probability_grid(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Evaluate `goal_probability` on a (rows, cols) grid given cell-centre axes.

    `xs` has shape `(cols,)`, `ys` has shape `(rows,)`. Output is `(rows, cols)`.
    """
    xx, yy = np.meshgrid(xs, ys)
    return np.asarray(goal_probability(xx, yy), dtype=np.float64)
