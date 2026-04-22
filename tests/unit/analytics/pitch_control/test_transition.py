"""Tests for the isotropic-gaussian ball-arrival probability used in OBSO."""

from __future__ import annotations

import numpy as np
import pytest

from football_analysis.analytics.pitch_control.transition import (
    DEFAULT_PASS_SIGMA_M,
    ball_arrival_probability,
)


def test_peaks_at_ball_position() -> None:
    xs = np.linspace(0.0, 100.0, 11)
    ys = np.linspace(0.0, 68.0, 9)
    surface = ball_arrival_probability(xs, ys, ball_x=50.0, ball_y=34.0)
    i = int(np.argmax(surface))
    # Grid cell nearest (50, 34) should be the max — and value should be 1.0 there
    assert surface.max() == pytest.approx(1.0, abs=1e-6)
    assert surface.flat[i] == surface.max()


def test_decay_symmetric_about_ball() -> None:
    xs = np.linspace(40.0, 60.0, 5)
    ys = np.linspace(29.0, 39.0, 5)
    surface = ball_arrival_probability(xs, ys, ball_x=50.0, ball_y=34.0)
    # (40, 29) and (60, 39) are both 10m * sqrt(2)/sqrt(2) = same distance on this grid
    assert surface[0, 0] == pytest.approx(surface[-1, -1], abs=1e-6)


def test_larger_sigma_spreads_mass() -> None:
    xs = np.array([50.0, 70.0])  # one centre, one 20m away
    ys = np.array([34.0])
    narrow = ball_arrival_probability(xs, ys, 50.0, 34.0, sigma=5.0)
    wide = ball_arrival_probability(xs, ys, 50.0, 34.0, sigma=30.0)
    # At 20m distance, the wide sigma preserves more mass
    assert wide[0, 1] > narrow[0, 1]


def test_sigma_must_be_positive() -> None:
    with pytest.raises(ValueError, match="sigma"):
        ball_arrival_probability(np.array([0.0]), np.array([0.0]), 0.0, 0.0, sigma=0.0)


def test_default_sigma_is_pass_range() -> None:
    """Default sigma should be in the typical-pass range (10-30 m)."""
    assert 10.0 <= DEFAULT_PASS_SIGMA_M <= 30.0
