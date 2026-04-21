"""Tests for team-level metrics: PPDA and field tilt."""

from __future__ import annotations

import math

import pandas as pd

from football_analysis.analytics.pitch import PITCH_LENGTH_M
from football_analysis.analytics.team.field_tilt import compute_field_tilt
from football_analysis.analytics.team.ppda import compute_ppda


def _passes(team_id: str, xs: list[float]) -> list[dict]:
    return [
        {
            "team_id": team_id, "action_type": "pass", "result": "success",
            "start_x": x, "start_y": 34.0, "end_x": x + 5, "end_y": 34.0,
        }
        for x in xs
    ]


def _def_actions(team_id: str, xs: list[float]) -> list[dict]:
    # All defensive action types are valid presses
    return [
        {
            "team_id": team_id, "action_type": "tackle", "result": "success",
            "start_x": x, "start_y": 34.0, "end_x": None, "end_y": None,
        }
        for x in xs
    ]


def test_ppda_basic_ratio() -> None:
    # Opponent plays 6 passes in their own 60% (x < 42); we make 3 defensive actions there.
    events = pd.DataFrame(
        _passes("B", [10.0, 15.0, 20.0, 25.0, 30.0, 35.0])  # all x < 42 threshold
        + _def_actions("A", [10.0, 20.0, 30.0])
    )
    assert compute_ppda(events, team_id="A", opponent_id="B") == 2.0


def test_ppda_infinite_when_no_defensive_actions() -> None:
    events = pd.DataFrame(_passes("B", [10.0, 20.0, 30.0]))
    assert math.isinf(compute_ppda(events, "A", "B"))


def test_ppda_ignores_passes_in_attacking_third() -> None:
    # Opponent passes in their attacking zone (x >= 42) — these shouldn't count
    events = pd.DataFrame(
        _passes("B", [10.0, 50.0, 60.0, 80.0])  # only the 10.0 pass counts
        + _def_actions("A", [10.0])
    )
    assert compute_ppda(events, "A", "B") == 1.0


def test_field_tilt_is_share_of_final_third_passes() -> None:
    threshold = (2.0 / 3.0) * PITCH_LENGTH_M  # 70.0
    events = pd.DataFrame(
        _passes("A", [80.0, 90.0, 95.0])  # 3 A passes in final third
        + _passes("B", [75.0])  # 1 B pass in final third
        + _passes("A", [30.0])  # A pass outside final third — ignored
    )
    assert compute_field_tilt(events, "A", "B") == 0.75
    assert compute_field_tilt(events, "B", "A") == 0.25
    assert threshold == 70.0  # regression check on constant


def test_field_tilt_returns_zero_when_no_final_third_passes() -> None:
    events = pd.DataFrame(_passes("A", [30.0, 40.0]) + _passes("B", [20.0]))
    assert compute_field_tilt(events, "A", "B") == 0.0
