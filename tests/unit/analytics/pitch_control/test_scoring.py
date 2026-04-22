"""Tests for the naive goal-probability surface used in OBSO."""

from __future__ import annotations

import numpy as np

from football_analysis.analytics.pitch_control.scoring import (
    GOAL_CENTER_Y,
    GOAL_X,
    _shot_angle,
    goal_probability,
    goal_probability_grid,
)


def test_probability_is_in_unit_interval() -> None:
    xs = np.linspace(0.0, 105.0, 21)
    ys = np.linspace(0.0, 68.0, 14)
    grid = goal_probability_grid(xs, ys)
    assert (grid >= 0.0).all()
    assert (grid <= 1.0).all()


def test_penalty_spot_higher_than_halfway() -> None:
    """The penalty spot (94m, 34m) must have materially higher goal prob than halfway (52m, 34m)."""
    p_spot = float(goal_probability(np.array([94.0]), np.array([34.0]))[0])
    p_half = float(goal_probability(np.array([52.5]), np.array([34.0]))[0])
    assert p_spot > p_half
    assert p_spot > 0.1
    assert p_half < p_spot / 2.0


def test_central_beats_wide_at_same_distance() -> None:
    """A central shot 18m out should beat a wide one the same distance out."""
    central = float(goal_probability(np.array([GOAL_X - 18.0]), np.array([GOAL_CENTER_Y]))[0])
    wide = float(goal_probability(np.array([GOAL_X - 18.0]), np.array([GOAL_CENTER_Y + 20.0]))[0])
    assert central > wide


def test_behind_goal_line_is_zero() -> None:
    p = float(goal_probability(np.array([GOAL_X + 5.0]), np.array([GOAL_CENTER_Y]))[0])
    assert p == 0.0


def test_shot_angle_wider_at_penalty_spot() -> None:
    """Angle subtended by the goal-mouth should be wider from 11m out than from 30m out."""
    near = float(_shot_angle(np.array([GOAL_X - 11.0]), np.array([GOAL_CENTER_Y]))[0])
    far = float(_shot_angle(np.array([GOAL_X - 30.0]), np.array([GOAL_CENTER_Y]))[0])
    assert near > far


def test_grid_shape_and_values_match_pointwise() -> None:
    xs = np.array([80.0, 90.0, 100.0])
    ys = np.array([20.0, 34.0, 48.0])
    grid = goal_probability_grid(xs, ys)
    assert grid.shape == (3, 3)
    # Pointwise spot check
    direct = goal_probability(np.array([90.0]), np.array([34.0]))[0]
    assert abs(grid[1, 1] - direct) < 1e-9
