"""Ball-transition probability: how likely is the ball to arrive at (x, y) from its
current position?

Used as the second factor in OBSO. A full Spearman model would incorporate pass
feasibility through the defender field (pitch control "passing lane"); a Phase-2
proof-of-life uses an isotropic Gaussian over distance from the current ball
position, which is sufficient to demonstrate the OBSO pipeline.

    P_arrive(x, y | ball_pos) = N(d = ||(x,y) - ball_pos||; sigma)

where sigma is chosen so realistic pass distances (~15-25 m) have meaningful
mass while 60+ m passes are rare.
"""

from __future__ import annotations

import numpy as np

DEFAULT_PASS_SIGMA_M: float = 20.0  # metres


def ball_arrival_probability(
    xs: np.ndarray,
    ys: np.ndarray,
    ball_x: float,
    ball_y: float,
    sigma: float = DEFAULT_PASS_SIGMA_M,
) -> np.ndarray:
    """Return an unnormalised (rows, cols) surface of P(ball arrives at each cell).

    `xs` has shape `(cols,)`, `ys` has shape `(rows,)`. We deliberately do NOT
    normalise across the pitch: OBSO uses the relative value per cell, not the
    global sum. Values are in [0, 1] with 1 at the ball's current location.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    xx, yy = np.meshgrid(xs, ys)
    dist2 = (xx - ball_x) ** 2 + (yy - ball_y) ** 2
    return np.asarray(np.exp(-dist2 / (2.0 * sigma**2)), dtype=np.float64)
