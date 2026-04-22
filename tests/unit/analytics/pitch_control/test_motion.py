"""Tests for the time-to-intercept motion model."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from football_analysis.analytics.pitch_control.motion import (
    DEFAULT_MAX_SPEED,
    DEFAULT_REACTION_TIME,
    time_to_intercept,
)


def _targets(rows: int = 3, cols: int = 3) -> np.ndarray:
    xs = np.linspace(0.0, 10.0, cols)
    ys = np.linspace(0.0, 10.0, rows)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx, yy], axis=-1)


def test_stationary_player_time_equals_reaction_plus_distance_over_speed() -> None:
    positions = np.array([[0.0, 0.0]])
    velocities = np.array([[0.0, 0.0]])
    targets = _targets()
    t = time_to_intercept(positions, velocities, targets, reaction_time=1.0, max_speed=2.0)
    # At target (0,0) the player is already there after their reaction time.
    assert t[0, 0, 0] == pytest.approx(1.0)
    # At (10, 10) the distance is sqrt(200), time = 1.0 + sqrt(200)/2.
    assert t[0, -1, -1] == pytest.approx(1.0 + np.sqrt(200) / 2.0)


def test_player_with_velocity_reaches_forward_target_faster_than_backward() -> None:
    positions = np.array([[0.0, 0.0]])
    velocities = np.array([[5.0, 0.0]])  # moving +x
    targets = np.array([[[10.0, 0.0]], [[-10.0, 0.0]]])  # (H=2, W=1, 2)
    t = time_to_intercept(positions, velocities, targets, reaction_time=0.5, max_speed=5.0)
    # Forward target: after reaction, player is at (2.5, 0), 7.5m away at 5m/s = 1.5s.
    # Backward target: after reaction at (2.5, 0), distance is 12.5m at 5m/s = 2.5s.
    assert t[0, 0, 0] == pytest.approx(0.5 + 7.5 / 5.0)  # forward
    assert t[0, 1, 0] == pytest.approx(0.5 + 12.5 / 5.0)  # backward
    assert t[0, 0, 0] < t[0, 1, 0]


def test_multiple_players_get_separate_times() -> None:
    positions = np.array([[0.0, 0.0], [10.0, 10.0]])
    velocities = np.array([[0.0, 0.0], [0.0, 0.0]])
    targets = _targets()
    t = time_to_intercept(positions, velocities, targets, reaction_time=0.0, max_speed=1.0)
    assert t.shape == (2, 3, 3)
    # Player 0 at (0,0) is 0 from (0,0)
    assert t[0, 0, 0] == pytest.approx(0.0)
    # Player 1 at (10,10) is 0 from (10,10)
    assert t[1, -1, -1] == pytest.approx(0.0)


def test_validates_input_shapes() -> None:
    good_targets = _targets()
    with pytest.raises(ValueError, match="positions"):
        time_to_intercept(np.array([0.0, 0.0]), np.zeros((1, 2)), good_targets)
    with pytest.raises(ValueError, match="velocities"):
        time_to_intercept(np.zeros((2, 2)), np.zeros((3, 2)), good_targets)
    with pytest.raises(ValueError, match="targets"):
        time_to_intercept(np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="max_speed"):
        time_to_intercept(np.zeros((1, 2)), np.zeros((1, 2)), good_targets, max_speed=0.0)


@given(
    px=st.floats(-50.0, 150.0, allow_nan=False, allow_infinity=False),
    py=st.floats(-50.0, 100.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50)
def test_reaction_time_is_lower_bound(px: float, py: float) -> None:
    """For any player with any velocity, time_to_intercept >= reaction_time everywhere
    except within the reaction-time drift radius (a detail we allow as an edge case).
    Here we test with zero velocity to keep the lower bound clean."""
    positions = np.array([[px, py]])
    velocities = np.zeros((1, 2))
    targets = np.array([[[px, py]]])  # target == position
    t = time_to_intercept(
        positions,
        velocities,
        targets,
        reaction_time=DEFAULT_REACTION_TIME,
        max_speed=DEFAULT_MAX_SPEED,
    )
    assert t[0, 0, 0] == pytest.approx(DEFAULT_REACTION_TIME)
