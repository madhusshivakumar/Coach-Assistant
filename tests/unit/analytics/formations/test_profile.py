"""Tests for FormationProfile extraction (tracking + events sides)."""

from __future__ import annotations

import pandas as pd
import pytest

from football_analysis.analytics.formations.profile import (
    FormationProfile,
    extract_events_profile,
    extract_tracking_profile,
)


def _tracking_with_phase(team: str, phase: str, n_frames: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a tracking + classified pair for one team in one phase, with a static shape."""
    rows: list[dict] = []
    classified_rows: list[dict] = []
    for f in range(1, n_frames + 1):
        t = 0.04 * f
        # 11 outfielders in a 4-3-3-ish shape, GK deep
        positions = [
            (5, 34),  # GK
            (25, 14),
            (25, 28),
            (25, 40),
            (25, 54),  # back 4
            (50, 22),
            (50, 34),
            (50, 46),  # mid 3
            (80, 14),
            (80, 34),
            (80, 54),  # front 3
        ]
        for i, (x, y) in enumerate(positions):
            rows.append(
                {
                    "frame_id": f,
                    "period": 1,
                    "time_seconds": t,
                    "player_id": f"{team}{i}",
                    "team_id": team,
                    "x": float(x),
                    "y": float(y),
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": False,
                    "visible": True,
                }
            )
        classified_rows.append({"frame_id": f, "time_seconds": t, "possession_team": team, "phase": phase})
    return pd.DataFrame(rows), pd.DataFrame(classified_rows)


def test_extract_tracking_profile_returns_one_per_phase() -> None:
    tracking, classified = _tracking_with_phase("home", "build_up", n_frames=50)
    profiles = extract_tracking_profile(tracking, classified, "home")
    assert "build_up" in profiles
    p = profiles["build_up"]
    assert isinstance(p, FormationProfile)
    assert p.sample == "tracking"
    assert p.team_id == "home"
    assert p.n_frames == 50
    assert {"length_mean", "width_mean", "hull_area_mean"} <= p.metrics.keys()


def test_extract_tracking_profile_skips_low_signal_phase() -> None:
    """Phases with fewer than min_frames_per_phase frames should be dropped."""
    tracking, classified = _tracking_with_phase("home", "build_up", n_frames=10)
    profiles = extract_tracking_profile(tracking, classified, "home", min_frames_per_phase=25)
    assert profiles == {}


def test_extract_tracking_profile_empty_when_no_tracking_for_team() -> None:
    tracking, classified = _tracking_with_phase("home", "build_up", n_frames=50)
    # ask for a team that isn't in the tracking
    profiles = extract_tracking_profile(tracking, classified, "phantom")
    assert profiles == {}


def _enriched_events() -> pd.DataFrame:
    """Minimal enriched-event frame with xt_delta + the columns PPDA / field-tilt need."""
    return pd.DataFrame(
        [
            # Home builds up + scores
            {
                "team_id": "H",
                "action_type": "pass",
                "result": "success",
                "start_x": 30.0,
                "start_y": 34.0,
                "end_x": 60.0,
                "end_y": 34.0,
                "xt_delta": 0.05,
            },
            {
                "team_id": "H",
                "action_type": "pass",
                "result": "success",
                "start_x": 60.0,
                "start_y": 34.0,
                "end_x": 90.0,
                "end_y": 34.0,
                "xt_delta": 0.10,
            },
            {
                "team_id": "H",
                "action_type": "shot",
                "result": "success",
                "start_x": 100.0,
                "start_y": 34.0,
                "end_x": None,
                "end_y": None,
                "xt_delta": float("nan"),
            },
            # Away passes in their own half (counts towards H's PPDA opp passes)
            {
                "team_id": "A",
                "action_type": "pass",
                "result": "success",
                "start_x": 20.0,
                "start_y": 30.0,
                "end_x": 30.0,
                "end_y": 30.0,
                "xt_delta": 0.02,
            },
            # Home defensive action in opp's own 60% (presses)
            {
                "team_id": "H",
                "action_type": "tackle",
                "result": "success",
                "start_x": 25.0,
                "start_y": 34.0,
                "end_x": None,
                "end_y": None,
                "xt_delta": float("nan"),
            },
        ]
    )


def test_extract_events_profile_has_expected_metric_keys() -> None:
    events = _enriched_events()
    p = extract_events_profile(events, "H", "A")
    assert isinstance(p, FormationProfile)
    assert p.phase is None
    assert p.sample == "events"
    assert {
        "xt_generated",
        "xt_conceded",
        "ppda",
        "field_tilt",
        "shots",
        "goals",
    } == p.metrics.keys()
    # H scored 1 shot, 1 goal
    assert p.metrics["shots"] == 1.0
    assert p.metrics["goals"] == 1.0
    # H generated positive xT (clipped to non-negative sum)
    assert p.metrics["xt_generated"] > 0


def test_extract_events_profile_caps_infinite_ppda() -> None:
    """When the opponent makes zero defensive actions, raw PPDA is +inf — must be capped."""
    rows = [
        {
            "team_id": "A",
            "action_type": "pass",
            "result": "success",
            "start_x": 20.0,
            "start_y": 34.0,
            "end_x": 30.0,
            "end_y": 34.0,
            "xt_delta": 0.0,
        },
    ]
    p = extract_events_profile(pd.DataFrame(rows), team_id="H", opponent_id="A")
    # No H defensive actions → PPDA capped at PPDA_INF_CAP (50)
    assert p.metrics["ppda"] == 50.0


def test_extract_events_profile_requires_xt_delta_column() -> None:
    rows = [
        {
            "team_id": "H",
            "action_type": "pass",
            "result": "success",
            "start_x": 30.0,
            "start_y": 34.0,
            "end_x": 60.0,
            "end_y": 34.0,
        },
    ]
    with pytest.raises(ValueError, match="xt_delta"):
        extract_events_profile(pd.DataFrame(rows), "H", "A")
