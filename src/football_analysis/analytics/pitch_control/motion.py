"""Motion model for pitch control.

Simplified constant-acceleration-to-max-speed intercept model, per the
LaurieOnTracking formulation:

    t_intercept(player, target) = reaction_time + ||target - (p + v * reaction_time)|| / v_max

That is: during the reaction time the player drifts with current velocity, then
thereafter moves in a straight line at max speed toward the target. Acceleration
is approximated away.

Vectorised so (N players) x (H x W grid) is computed in one numpy call.
"""

from __future__ import annotations

import numpy as np

DEFAULT_REACTION_TIME: float = 0.7  # seconds
DEFAULT_MAX_SPEED: float = 5.0  # m/s — used for players; ball uses a higher value


def time_to_intercept(
    positions: np.ndarray,
    velocities: np.ndarray,
    targets: np.ndarray,
    reaction_time: float = DEFAULT_REACTION_TIME,
    max_speed: float = DEFAULT_MAX_SPEED,
) -> np.ndarray:
    """Return the time each player needs to reach each target point.

    Args:
        positions: shape `(N, 2)` — `(x, y)` per player in metres.
        velocities: shape `(N, 2)` — `(vx, vy)` per player in m/s.
        targets: shape `(H, W, 2)` — `(x, y)` for each grid cell in metres.
        reaction_time: seconds of drift before the player redirects.
        max_speed: hard cap on straight-line running speed (m/s).

    Returns:
        Array of shape `(N, H, W)`; element `[i, h, w]` is player `i`'s time to
        reach the grid cell centred at `targets[h, w]`.
    """
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2)")
    if velocities.shape != positions.shape:
        raise ValueError("velocities must have the same shape as positions")
    if targets.ndim != 3 or targets.shape[-1] != 2:
        raise ValueError("targets must have shape (H, W, 2)")
    if max_speed <= 0.0:
        raise ValueError("max_speed must be positive")

    reaction_pos = positions + velocities * reaction_time  # (N, 2)
    diff = targets[None, :, :, :] - reaction_pos[:, None, None, :]  # (N, H, W, 2)
    dist = np.linalg.norm(diff, axis=-1)  # (N, H, W)
    result: np.ndarray = reaction_time + dist / max_speed
    return result
