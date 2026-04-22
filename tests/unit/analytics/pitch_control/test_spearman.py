"""Tests for the Spearman pitch-control surface."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from football_analysis.analytics.pitch import PITCH_LENGTH_M
from football_analysis.analytics.pitch_control.spearman import (
    PitchControlFrame,
    compute_frame,
    compute_frame_from_tracking,
)


def _zero_vel(n: int) -> np.ndarray:
    return np.zeros((n, 2))


def test_returns_shape_and_axes() -> None:
    home = np.array([[50.0, 34.0]])
    away = np.array([[60.0, 34.0]])
    control = compute_frame(home, _zero_vel(1), away, _zero_vel(1), rows=8, cols=16)
    assert isinstance(control, PitchControlFrame)
    assert control.home_control.shape == (8, 16)
    assert control.xs.shape == (16,)
    assert control.ys.shape == (8,)


def test_values_in_unit_interval() -> None:
    home = np.array([[30.0, 30.0], [80.0, 40.0]])
    away = np.array([[50.0, 20.0], [90.0, 50.0]])
    c = compute_frame(home, _zero_vel(2), away, _zero_vel(2))
    assert (c.home_control >= 0.0).all()
    assert (c.home_control <= 1.0).all()


def test_empty_teams_fall_back_to_half() -> None:
    """Both teams empty → P(home) = 0.5 everywhere (maximum uncertainty)."""
    empty = np.empty((0, 2))
    c = compute_frame(empty, empty, empty, empty, rows=4, cols=4)
    assert np.allclose(c.home_control, 0.5)


def test_only_home_present_dominates() -> None:
    home = np.array([[50.0, 34.0]])
    empty = np.empty((0, 2))
    c = compute_frame(home, _zero_vel(1), empty, empty, rows=4, cols=4)
    assert (c.home_control > 0.99).all()


def test_only_away_present_home_near_zero() -> None:
    away = np.array([[50.0, 34.0]])
    empty = np.empty((0, 2))
    c = compute_frame(empty, empty, away, _zero_vel(1), rows=4, cols=4)
    assert (c.home_control < 0.01).all()


def test_mirror_symmetry_sums_to_one() -> None:
    """Swap home<->away and the two control surfaces should sum to 1 per cell
    (since P_home + P_away == 1 by construction)."""
    home = np.array([[30.0, 20.0], [70.0, 50.0]])
    away = np.array([[50.0, 34.0]])
    c = compute_frame(home, _zero_vel(2), away, _zero_vel(1), rows=6, cols=10)
    # Compute the mirror by swapping inputs
    c_mirror = compute_frame(away, _zero_vel(1), home, _zero_vel(2), rows=6, cols=10)
    total = c.home_control + c_mirror.home_control
    # Should be 1.0 pointwise (within floating-point)
    assert np.allclose(total, 1.0, atol=1e-9)


def test_closer_player_wins_cell() -> None:
    """A home player standing on top of a cell should control it against a distant away player."""
    home = np.array([[10.0, 10.0]])
    away = np.array([[100.0, 60.0]])
    c = compute_frame(home, _zero_vel(1), away, _zero_vel(1), rows=34, cols=52)
    # The cell nearest (10,10): column ~ 10/105*52 = 4.95 -> 4 or 5; row ~ 10/68*34 ~ 5.
    r = 5
    col = round(10.0 / PITCH_LENGTH_M * 52)
    assert c.home_control[r, col] > 0.7


@given(
    sigma=st.floats(min_value=0.1, max_value=2.0, allow_nan=False),
)
@settings(max_examples=15, deadline=None)
def test_sigma_range_keeps_output_in_01(sigma: float) -> None:
    home = np.array([[30.0, 30.0]])
    away = np.array([[70.0, 30.0]])
    c = compute_frame(home, _zero_vel(1), away, _zero_vel(1), sigma=sigma, rows=6, cols=10)
    assert (c.home_control >= 0.0).all()
    assert (c.home_control <= 1.0).all()


def test_from_tracking_builds_frame_from_dataframe() -> None:
    tracking = pd.DataFrame(
        [
            # Home player
            {
                "frame_id": 1,
                "team_id": "home",
                "player_id": "h1",
                "x": 20.0,
                "y": 30.0,
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": False,
                "visible": True,
            },
            # Away player
            {
                "frame_id": 1,
                "team_id": "away",
                "player_id": "a1",
                "x": 80.0,
                "y": 38.0,
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": False,
                "visible": True,
            },
            # Ball — should be ignored for control computation
            {
                "frame_id": 1,
                "team_id": None,
                "player_id": None,
                "x": 50.0,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": True,
                "visible": True,
            },
            # Off-camera away player — should be ignored
            {
                "frame_id": 1,
                "team_id": "away",
                "player_id": "a2",
                "x": np.nan,
                "y": np.nan,
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": False,
                "visible": False,
            },
        ]
    )
    c = compute_frame_from_tracking(tracking, frame_id=1, home_team_id="home", away_team_id="away")
    # Home should control their own area (left), away controls right.
    assert c.home_control[:, 5].mean() > 0.6
    assert c.home_control[:, -5].mean() < 0.4


def test_from_tracking_raises_on_missing_frame() -> None:
    tracking = pd.DataFrame(
        [
            {
                "frame_id": 1,
                "team_id": "home",
                "player_id": "h1",
                "x": 20.0,
                "y": 30.0,
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": False,
                "visible": True,
            },
        ]
    )
    with pytest.raises(ValueError, match="frame_id"):
        compute_frame_from_tracking(tracking, frame_id=999, home_team_id="home", away_team_id="away")
