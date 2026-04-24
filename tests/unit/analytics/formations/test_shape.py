"""Tests for team-shape / compactness metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from football_analysis.analytics.formations.shape import (
    ShapeMetrics,
    compute_shape,
    shape_time_series,
)


def _square(side: float = 20.0, origin_x: float = 20.0, origin_y: float = 20.0) -> pd.DataFrame:
    """Four corners of a square — convex hull area should be side^2."""
    return pd.DataFrame(
        [
            {"player_id": "a", "x": origin_x, "y": origin_y},
            {"player_id": "b", "x": origin_x + side, "y": origin_y},
            {"player_id": "c", "x": origin_x + side, "y": origin_y + side},
            {"player_id": "d", "x": origin_x, "y": origin_y + side},
        ]
    )


def test_square_gives_expected_length_width_hull() -> None:
    positions = _square(side=20.0)
    m = compute_shape(positions, back_line_n=2, top_line_n=2)
    assert isinstance(m, ShapeMetrics)
    assert m.length == pytest.approx(20.0, abs=1e-6)
    assert m.width == pytest.approx(20.0, abs=1e-6)
    assert m.convex_hull_area == pytest.approx(400.0, abs=1e-6)


def test_vertical_compactness_is_offensive_minus_defensive_line() -> None:
    positions = _square(side=30.0)
    m = compute_shape(positions, back_line_n=2, top_line_n=2)
    assert m.offensive_line_height - m.defensive_line_height == pytest.approx(m.vertical_compactness)
    assert m.vertical_compactness == pytest.approx(30.0, abs=1e-6)


def test_empty_positions_returns_zeros() -> None:
    m = compute_shape(pd.DataFrame(columns=["player_id", "x", "y"]))
    assert m.length == 0.0
    assert m.width == 0.0
    assert m.convex_hull_area == 0.0


def test_attacking_right_false_mirrors_positions() -> None:
    """Same physical shape should yield the same length/width regardless of
    attacking direction — mirroring is symmetry-preserving."""
    left_side = _square(side=20.0, origin_x=20.0)
    right_side = pd.DataFrame(
        [{"player_id": "a", "x": 105.0 - r["x"], "y": 68.0 - r["y"]} for _, r in left_side.iterrows()]
    )
    m_left = compute_shape(left_side, back_line_n=2, top_line_n=2, attacking_right=True)
    m_right = compute_shape(right_side, back_line_n=2, top_line_n=2, attacking_right=False)
    assert m_left.length == pytest.approx(m_right.length)
    assert m_left.width == pytest.approx(m_right.width)
    assert m_left.convex_hull_area == pytest.approx(m_right.convex_hull_area)
    # Line heights should also match after mirroring.
    assert m_left.defensive_line_height == pytest.approx(m_right.defensive_line_height)


def test_three_points_still_forms_a_hull() -> None:
    rows = pd.DataFrame(
        [
            {"player_id": "a", "x": 20.0, "y": 20.0},
            {"player_id": "b", "x": 40.0, "y": 20.0},
            {"player_id": "c", "x": 30.0, "y": 40.0},
        ]
    )
    m = compute_shape(rows, back_line_n=2, top_line_n=1)
    # Triangle area 200
    assert m.convex_hull_area == pytest.approx(200.0, abs=1e-6)


def test_shape_time_series_excludes_goalkeeper() -> None:
    """Construct a fake 11-player team where the GK is deep (x near 0) and verify
    they're excluded from the per-frame shape computation."""
    rows: list[dict] = []
    # 3 frames, 11 home outfielders-like rows (one of them is the GK at x=5)
    for f in range(1, 4):
        t = 0.04 * f
        for i in range(11):
            rows.append(
                {
                    "frame_id": f,
                    "period": 1,
                    "time_seconds": t,
                    "player_id": f"h{i}",
                    "team_id": "home",
                    "x": 5.0 if i == 0 else 30.0 + i,
                    "y": 34.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": False,
                    "visible": True,
                }
            )
    ts = shape_time_series(pd.DataFrame(rows), team_id="home")
    assert len(ts) == 3
    # GK x=5 should not drag the defensive line down. With GK excluded the
    # minimum x among the 10 outfielders is 31 (i=1).
    assert ts["defensive_line_height"].min() >= 31.0


def test_shape_time_series_empty_team() -> None:
    rows = [
        {
            "frame_id": 1,
            "period": 1,
            "time_seconds": 0.0,
            "player_id": None,
            "team_id": None,
            "x": 50.0,
            "y": 34.0,
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": True,
            "visible": True,
        }
    ]
    ts = shape_time_series(pd.DataFrame(rows), team_id="home")
    assert ts.empty
