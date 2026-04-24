"""Tests for Bialkowski-style per-frame role assignment."""

from __future__ import annotations

import pandas as pd
import pytest

from football_analysis.analytics.formations.roles import (
    FORMATION_4_2_3_1,
    FORMATION_4_3_3,
    FORMATION_4_4_2,
    assign_roles,
    best_template_for_frame,
)


def _positions_from_template(template, jitter: float = 1.0) -> pd.DataFrame:
    """Build outfielder positions close to the template slots with small jitter."""
    rows: list[dict] = []
    for i in range(len(template.roles)):
        rows.append(
            {
                "player_id": f"p{i}",
                "x": template.xs[i] + (jitter if i % 2 == 0 else -jitter),
                "y": template.ys[i] + (jitter if i % 3 == 0 else -jitter / 2),
            }
        )
    return pd.DataFrame(rows)


def test_assign_roles_recovers_perfect_template() -> None:
    """If players sit exactly on the template slots, the Hungarian assignment should
    label each player with their matching role."""
    tpl = FORMATION_4_3_3
    rows = [{"player_id": f"p{i}", "x": tpl.xs[i], "y": tpl.ys[i]} for i in range(10)]
    out = assign_roles(pd.DataFrame(rows), template=tpl)
    by_player = dict(zip(out["player_id"], out["role"], strict=True))
    for i, expected_role in enumerate(tpl.roles):
        assert by_player[f"p{i}"] == expected_role
    assert (out["displacement"] < 1e-9).all()


def test_assign_roles_small_perturbation_preserves_role_map() -> None:
    tpl = FORMATION_4_3_3
    positions = _positions_from_template(tpl, jitter=1.0)
    out = assign_roles(positions, template=tpl)
    assert len(out) == 10
    assert set(out["role"]) == set(tpl.roles)  # every slot assigned once
    assert (out["displacement"] <= 1.5).all()


def test_assign_roles_validates_10_players() -> None:
    with pytest.raises(ValueError, match="10 outfielders"):
        assign_roles(pd.DataFrame([{"player_id": "p1", "x": 50.0, "y": 34.0}]))


def test_assign_roles_mirrors_for_attacking_left() -> None:
    """When attacking_right=False, the positions should be mirrored before matching,
    so a team that attacks -x still gets their correct roles."""
    tpl = FORMATION_4_3_3
    # Put the team in a mirrored position (as if they attack -x on canonical pitch).
    rows: list[dict] = []
    for i, _ in enumerate(tpl.roles):
        rows.append(
            {
                "player_id": f"p{i}",
                "x": 105.0 - tpl.xs[i],
                "y": 68.0 - tpl.ys[i],
            }
        )
    out = assign_roles(pd.DataFrame(rows), template=tpl, attacking_right=False)
    by_player = dict(zip(out["player_id"], out["role"], strict=True))
    for i, expected_role in enumerate(tpl.roles):
        assert by_player[f"p{i}"] == expected_role


def test_best_template_picks_matching_shape() -> None:
    """When positions are built from a specific template, best_template_for_frame should
    prefer that template over the others."""
    for tpl in (FORMATION_4_3_3, FORMATION_4_4_2, FORMATION_4_2_3_1):
        positions = _positions_from_template(tpl, jitter=0.5)
        best, cost = best_template_for_frame(positions)
        assert best.name == tpl.name
        assert cost < 50.0  # small total displacement


def test_assign_roles_rejects_wrong_template_size() -> None:
    from football_analysis.analytics.formations.roles import FormationTemplate

    nine_slots = FormationTemplate(
        name="broken",
        roles=tuple(f"R{i}" for i in range(9)),
        xs=tuple([50.0] * 9),
        ys=tuple([34.0] * 9),
    )
    with pytest.raises(ValueError, match="10 slots"):
        assign_roles(_positions_from_template(FORMATION_4_3_3), template=nine_slots)
