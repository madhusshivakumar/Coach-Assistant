"""Tests for orientation normalisation (home attacks L->R in period 1)."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M
from football_analysis.data.normalize.orientation import normalise_point


def test_period1_home_ltr_identity() -> None:
    x, y = normalise_point(10.0, 20.0, period=1, home_attacks_left_to_right_p1=True)
    assert (x, y) == (10.0, 20.0)


def test_period1_home_rtl_flips() -> None:
    x, y = normalise_point(10.0, 20.0, period=1, home_attacks_left_to_right_p1=False)
    assert x == PITCH_LENGTH_M - 10.0
    assert y == PITCH_WIDTH_M - 20.0


def test_period2_home_ltr_flips() -> None:
    # Home attacked L->R in H1 means they attack R->L in H2; we flip to keep canonical.
    x, y = normalise_point(10.0, 20.0, period=2, home_attacks_left_to_right_p1=True)
    assert x == PITCH_LENGTH_M - 10.0
    assert y == PITCH_WIDTH_M - 20.0


def test_period2_home_rtl_identity() -> None:
    x, y = normalise_point(10.0, 20.0, period=2, home_attacks_left_to_right_p1=False)
    assert (x, y) == (10.0, 20.0)


def test_period_must_be_positive() -> None:
    import pytest

    with pytest.raises(ValueError):
        normalise_point(0.0, 0.0, period=0, home_attacks_left_to_right_p1=True)


@given(
    x=st.floats(min_value=0.0, max_value=PITCH_LENGTH_M, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=0.0, max_value=PITCH_WIDTH_M, allow_nan=False, allow_infinity=False),
    period=st.integers(min_value=1, max_value=5),
    p1=st.booleans(),
)
def test_normalise_is_involution_per_period(x: float, y: float, period: int, p1: bool) -> None:
    """Applying the same normalisation twice returns the original point (flip is its own inverse)."""
    a = normalise_point(x, y, period, p1)
    b = normalise_point(a[0], a[1], period, p1)
    assert abs(b[0] - x) < 1e-9
    assert abs(b[1] - y) < 1e-9
