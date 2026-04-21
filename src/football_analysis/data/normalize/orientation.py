"""Orientation normalisation.

Contract: after normalisation, the **home team attacks left-to-right in the first half**,
and right-to-left in the second half (flipped at halftime).

In practice this means:
- If kloppy reports a home_team that attacks in the `ATTACK_RIGHT` direction during period 1,
  no transform is needed for period 1; period 2 is flipped.
- If a provider has the opposite convention, every event/frame in period 1 is flipped and
  period 2 is identity.

This module is intentionally provider-agnostic: callers pass a boolean
`home_attacks_left_to_right_p1` determined from kloppy metadata.
"""

from __future__ import annotations

from football_analysis.analytics.pitch import Pitch, flip_horizontal


def normalise_point(
    x: float,
    y: float,
    period: int,
    home_attacks_left_to_right_p1: bool,
    pitch: Pitch | None = None,
) -> tuple[float, float]:
    """Return the canonical-orientation (x, y) for a point in a given period.

    `period` is 1, 2, or extra/penalty periods (3, 4, 5).
    - Period 1: flip iff !home_attacks_left_to_right_p1.
    - Period 2+: flip iff home_attacks_left_to_right_p1 (teams swap ends at every period break).
    """
    if period < 1:
        raise ValueError(f"period must be >= 1; got {period}")
    # Home's attacking direction alternates each period
    home_ltr_this_period = home_attacks_left_to_right_p1 if (period % 2 == 1) else (not home_attacks_left_to_right_p1)
    if home_ltr_this_period:
        return x, y
    return flip_horizontal(x, y, pitch)
