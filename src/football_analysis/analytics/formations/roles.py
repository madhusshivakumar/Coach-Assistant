"""Bialkowski-style per-frame role assignment.

Given the current positions of 10 outfielders and a reference formation template
(role slot coordinates in a normalised frame), solve the Hungarian assignment that
minimises total squared displacement between players and role slots. This produces
a (frame → {player_id: role_label}) mapping that is stable across frames even when
players swap positions within the shape.

Templates are expressed with the attacking direction pointing towards +x, so the
ordering (attacking team) maps x values onto the pitch's attacking half. Role
coordinates are given in metres on the canonical 105×68 pitch, relative to the
team's attacking goal (so "LCB" sits at small x, "ST" at large x).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M


@dataclass(frozen=True)
class FormationTemplate:
    """Named reference positions for each role slot. Coordinates in metres on 105×68."""

    name: str
    roles: tuple[str, ...]
    xs: tuple[float, ...]
    ys: tuple[float, ...]

    def as_array(self) -> np.ndarray:
        """Return role coordinates as an (n, 2) numpy array."""
        return np.stack([np.asarray(self.xs, dtype=np.float64), np.asarray(self.ys, dtype=np.float64)], axis=1)


# Canonical 4-3-3 template, attacking → +x. Numbers are chosen to sit in plausible
# average positions across a full match (not peak-attack positions).
FORMATION_4_3_3 = FormationTemplate(
    name="4-3-3",
    roles=("LB", "LCB", "RCB", "RB", "LCM", "DM", "RCM", "LW", "ST", "RW"),
    xs=(40, 40, 40, 40, 60, 55, 60, 80, 85, 80),
    ys=(14, 28, 40, 54, 22, 34, 46, 12, 34, 56),
)

# 4-4-2 with flat midfield four.
FORMATION_4_4_2 = FormationTemplate(
    name="4-4-2",
    roles=("LB", "LCB", "RCB", "RB", "LM", "LCM", "RCM", "RM", "LF", "RF"),
    xs=(40, 40, 40, 40, 65, 60, 60, 65, 85, 85),
    ys=(12, 28, 40, 56, 10, 28, 40, 58, 26, 42),
)

# 4-2-3-1 (double pivot, a number 10, lone 9).
FORMATION_4_2_3_1 = FormationTemplate(
    name="4-2-3-1",
    roles=("LB", "LCB", "RCB", "RB", "LDM", "RDM", "LAM", "CAM", "RAM", "ST"),
    xs=(40, 40, 40, 40, 58, 58, 78, 76, 78, 88),
    ys=(12, 28, 40, 56, 28, 40, 14, 34, 54, 34),
)

DEFAULT_TEMPLATES: tuple[FormationTemplate, ...] = (
    FORMATION_4_3_3,
    FORMATION_4_4_2,
    FORMATION_4_2_3_1,
)


def _orient_to_attacking_right(positions: np.ndarray, attacking_right: bool) -> np.ndarray:
    """Mirror positions so the team is always attacking towards +x."""
    if attacking_right:
        return positions
    mirrored = positions.copy()
    mirrored[:, 0] = PITCH_LENGTH_M - mirrored[:, 0]
    mirrored[:, 1] = PITCH_WIDTH_M - mirrored[:, 1]
    return mirrored


def assign_roles(
    positions: pd.DataFrame,
    template: FormationTemplate = FORMATION_4_3_3,
    attacking_right: bool = True,
) -> pd.DataFrame:
    """Assign role labels to 10 outfielders in a single frame.

    Args:
        positions: DataFrame with columns `player_id`, `x`, `y`. Exactly 10 rows.
            Rows that are clearly the GK (lowest mean x over a match) should
            already be filtered out by the caller — GKs are not role-assigned.
        template: formation template to map to.
        attacking_right: True if the team attacks towards +x in canonical coords.
            Away-team frames should pass `False` so they're flipped before matching.

    Returns:
        DataFrame with columns `player_id`, `role`, `x`, `y`, `slot_x`, `slot_y`,
        `displacement` (Euclidean distance from slot).
    """
    if len(positions) != 10:
        raise ValueError(f"assign_roles expects exactly 10 outfielders, got {len(positions)}")
    if len(template.roles) != 10:
        raise ValueError(f"template must have 10 slots, got {len(template.roles)}")

    coords = positions[["x", "y"]].to_numpy(dtype=np.float64)
    oriented = _orient_to_attacking_right(coords, attacking_right)
    slots = template.as_array()

    # Cost matrix: pairwise squared Euclidean
    diff = oriented[:, None, :] - slots[None, :, :]
    cost = np.sum(diff * diff, axis=2)
    row_idx, col_idx = linear_sum_assignment(cost)

    out = positions.iloc[row_idx].copy().reset_index(drop=True)
    out["role"] = [template.roles[j] for j in col_idx]
    out["slot_x"] = slots[col_idx, 0]
    out["slot_y"] = slots[col_idx, 1]
    out["displacement"] = np.sqrt(cost[row_idx, col_idx])
    return out[["player_id", "role", "x", "y", "slot_x", "slot_y", "displacement"]]


def best_template_for_frame(
    positions: pd.DataFrame,
    templates: tuple[FormationTemplate, ...] = DEFAULT_TEMPLATES,
    attacking_right: bool = True,
) -> tuple[FormationTemplate, float]:
    """Pick the template with lowest total displacement cost for these positions."""
    coords = _orient_to_attacking_right(positions[["x", "y"]].to_numpy(dtype=np.float64), attacking_right)
    best_tpl: FormationTemplate | None = None
    best_cost = float("inf")
    for tpl in templates:
        slots = tpl.as_array()
        diff = coords[:, None, :] - slots[None, :, :]
        cost = np.sum(diff * diff, axis=2)
        _, col_idx = linear_sum_assignment(cost)
        total = float(cost[np.arange(len(coords)), col_idx].sum())
        if total < best_cost:
            best_cost = total
            best_tpl = tpl
    assert best_tpl is not None  # templates is non-empty by type
    return best_tpl, best_cost
