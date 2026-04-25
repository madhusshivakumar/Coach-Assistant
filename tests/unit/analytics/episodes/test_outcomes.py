"""Tests for the episode-outcome classifier."""

from __future__ import annotations

import math

import pandas as pd

from football_analysis.analytics.episodes.outcomes import (
    FINAL_THIRD_X,
    PENALTY_AREA_X_MIN,
    EpisodeOutcome,
    classify_outcome,
)
from football_analysis.analytics.episodes.segmenter import EpisodeBoundary


def _tracking_with_ball_at(end_x: float, end_y: float, end_speed: float) -> pd.DataFrame:
    """Single-frame tracking with the ball at a given pos + speed."""
    rows = [
        {
            "frame_id": 1,
            "period": 1,
            "time_seconds": 0.04,
            "player_id": "ball",
            "team_id": "home",
            "x": end_x,
            "y": end_y,
            "vx": end_speed,
            "vy": 0.0,
            "is_ball": True,
            "visible": True,
        }
    ]
    return pd.DataFrame(rows)


def _ep(end_reason: str = "possession_change") -> EpisodeBoundary:
    return EpisodeBoundary(
        episode_id=0,
        start_frame=1,
        end_frame=1,
        start_time_s=0.04,
        end_time_s=0.04,
        duration_s=0.0,
        possession_team="home",
        end_reason=end_reason,
    )


def test_outcome_reached_final_third_when_ball_past_x70() -> None:
    """Ball at x=80 → reached_final_third=True (attacking right)."""
    tracking = _tracking_with_ball_at(end_x=80.0, end_y=34.0, end_speed=2.0)
    out = classify_outcome(_ep(), tracking, attacking_to_right=True)
    assert out.reached_final_third is True


def test_outcome_does_not_reach_final_third_when_ball_in_own_half() -> None:
    tracking = _tracking_with_ball_at(end_x=40.0, end_y=34.0, end_speed=2.0)
    out = classify_outcome(_ep(), tracking, attacking_to_right=True)
    assert out.reached_final_third is False


def test_outcome_orientation_flips_final_third_check() -> None:
    """Ball at x=20 means we're in the FINAL THIRD if attacking left."""
    tracking = _tracking_with_ball_at(end_x=20.0, end_y=34.0, end_speed=2.0)
    out_left = classify_outcome(_ep(), tracking, attacking_to_right=False)
    out_right = classify_outcome(_ep(), tracking, attacking_to_right=True)
    assert out_left.reached_final_third is True
    assert out_right.reached_final_third is False


def test_outcome_ended_in_box_requires_x_and_y_in_box() -> None:
    """Ball just inside the penalty area."""
    tracking = _tracking_with_ball_at(end_x=PENALTY_AREA_X_MIN + 1.0, end_y=34.0, end_speed=5.0)
    out = classify_outcome(_ep(), tracking, attacking_to_right=True)
    assert out.ended_in_box is True
    # off the side: y way outside box
    tracking2 = _tracking_with_ball_at(end_x=PENALTY_AREA_X_MIN + 1.0, end_y=2.0, end_speed=5.0)
    out2 = classify_outcome(_ep(), tracking2, attacking_to_right=True)
    assert out2.ended_in_box is False


def test_outcome_shot_like_requires_box_and_speed() -> None:
    # In box but slow -> not shot_like
    slow = _tracking_with_ball_at(end_x=95.0, end_y=34.0, end_speed=3.0)
    out_slow = classify_outcome(_ep(), slow, attacking_to_right=True)
    assert out_slow.ended_in_box is True
    assert out_slow.shot_like is False

    # In box AND fast → shot_like
    fast = _tracking_with_ball_at(end_x=95.0, end_y=34.0, end_speed=20.0)
    out_fast = classify_outcome(_ep(), fast, attacking_to_right=True)
    assert out_fast.shot_like is True


def test_outcome_with_no_visible_ball_returns_nans() -> None:
    """Episode with no visible ball frames returns NaN ball coords + False flags."""
    tracking = pd.DataFrame(
        [
            {
                "frame_id": 1,
                "period": 1,
                "time_seconds": 0.04,
                "player_id": "ball",
                "team_id": "home",
                "x": 95.0,
                "y": 34.0,
                "vx": 20.0,
                "vy": 0.0,
                "is_ball": True,
                "visible": False,
            }
        ]
    )
    out = classify_outcome(_ep(), tracking, attacking_to_right=True)
    assert math.isnan(out.end_ball_x)
    assert out.reached_final_third is False
    assert out.ended_in_box is False
    assert out.shot_like is False


def test_outcome_preserves_end_reason_and_duration() -> None:
    tracking = _tracking_with_ball_at(end_x=50.0, end_y=34.0, end_speed=1.0)
    ep = EpisodeBoundary(
        episode_id=42,
        start_frame=1,
        end_frame=10,
        start_time_s=0.04,
        end_time_s=0.4,
        duration_s=0.36,
        possession_team="home",
        end_reason="out_of_play",
    )
    out = classify_outcome(ep, tracking, attacking_to_right=True)
    assert out.end_reason == "out_of_play"
    assert math.isclose(out.duration_s, 0.36, abs_tol=1e-6)
    assert isinstance(out, EpisodeOutcome)
    assert out.episode_id == 42


def test_outcome_uses_final_third_constant() -> None:
    """Sanity-check the canonical 105 m pitch puts the final-third boundary at x=70."""
    assert FINAL_THIRD_X == 70.0
