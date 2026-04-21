"""Tests for canonical pitch geometry."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from football_analysis.analytics.pitch import (
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    Pitch,
    flip_horizontal,
    rescale_point,
)


def test_pitch_defaults() -> None:
    p = Pitch()
    assert p.length == PITCH_LENGTH_M
    assert p.width == PITCH_WIDTH_M
    assert p.center == (52.5, 34.0)


def test_in_bounds_inclusive() -> None:
    p = Pitch()
    assert p.in_bounds(0.0, 0.0)
    assert p.in_bounds(105.0, 68.0)
    assert not p.in_bounds(-0.01, 34.0)
    assert not p.in_bounds(105.01, 34.0)


def test_rescale_bottom_left_identity_for_canonical_pitch() -> None:
    # A point on a 105x68 pitch with bottom-left origin maps to itself.
    x, y = rescale_point(10.0, 20.0, 105.0, 68.0, source_origin="bottom_left")
    assert abs(x - 10.0) < 1e-9
    assert abs(y - 20.0) < 1e-9


def test_rescale_top_left_flips_y() -> None:
    # StatsBomb 120x80 with origin top-left. A point at (60, 40) is centre.
    x, y = rescale_point(60.0, 40.0, 120.0, 80.0, source_origin="top_left")
    assert abs(x - 52.5) < 1e-9
    assert abs(y - 34.0) < 1e-9  # 40 is centre y, maps to centre of 68m pitch


def test_rescale_center_origin() -> None:
    # Metrica normalised [-0.5, 0.5] on length 1 -> canonical centre
    x, y = rescale_point(0.0, 0.0, 1.0, 1.0, source_origin="center")
    assert abs(x - 52.5) < 1e-9
    assert abs(y - 34.0) < 1e-9


def test_rescale_unknown_origin_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        rescale_point(0.0, 0.0, 105.0, 68.0, source_origin="nope")


def test_flip_horizontal_is_involution() -> None:
    x, y = 12.3, 45.6
    fx, fy = flip_horizontal(x, y)
    ffx, ffy = flip_horizontal(fx, fy)
    assert abs(ffx - x) < 1e-9
    assert abs(ffy - y) < 1e-9


@given(
    x=st.floats(min_value=0.0, max_value=PITCH_LENGTH_M, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=0.0, max_value=PITCH_WIDTH_M, allow_nan=False, allow_infinity=False),
)
def test_flip_twice_roundtrips(x: float, y: float) -> None:
    fx, fy = flip_horizontal(x, y)
    ffx, ffy = flip_horizontal(fx, fy)
    assert abs(ffx - x) < 1e-9
    assert abs(ffy - y) < 1e-9
