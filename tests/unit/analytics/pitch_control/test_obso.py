"""Tests for the OBSO surface + per-player attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_analysis.analytics.pitch_control.obso import (
    OBSOFrame,
    compute_obso_frame,
    per_player_obso,
)


def _tracking_with_ball_and_players(ball_xy: tuple[float, float] | None = (70.0, 34.0)) -> pd.DataFrame:
    rows = []
    # Home attackers in the final third; 2 forwards near the box
    rows.append(
        {
            "frame_id": 1,
            "team_id": "home",
            "player_id": "h_fwd1",
            "x": 90.0,
            "y": 30.0,
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": False,
            "visible": True,
        }
    )
    rows.append(
        {
            "frame_id": 1,
            "team_id": "home",
            "player_id": "h_fwd2",
            "x": 92.0,
            "y": 38.0,
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": False,
            "visible": True,
        }
    )
    rows.append(
        {
            "frame_id": 1,
            "team_id": "home",
            "player_id": "h_mid",
            "x": 55.0,
            "y": 34.0,
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": False,
            "visible": True,
        }
    )
    # Away defenders dispersed; one on each forward, one deep
    rows.append(
        {
            "frame_id": 1,
            "team_id": "away",
            "player_id": "a_cb1",
            "x": 95.0,
            "y": 30.0,
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": False,
            "visible": True,
        }
    )
    rows.append(
        {
            "frame_id": 1,
            "team_id": "away",
            "player_id": "a_cb2",
            "x": 93.0,
            "y": 40.0,
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": False,
            "visible": True,
        }
    )
    rows.append(
        {
            "frame_id": 1,
            "team_id": "away",
            "player_id": "a_gk",
            "x": 102.0,
            "y": 34.0,
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": False,
            "visible": True,
        }
    )
    # Ball
    if ball_xy is not None:
        rows.append(
            {
                "frame_id": 1,
                "team_id": None,
                "player_id": None,
                "x": ball_xy[0],
                "y": ball_xy[1],
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": True,
                "visible": True,
            }
        )
    return pd.DataFrame(rows)


def test_returns_frame_with_matching_shapes() -> None:
    out = compute_obso_frame(
        _tracking_with_ball_and_players(),
        frame_id=1,
        attacking_team_id="home",
        defending_team_id="away",
        rows=34,
        cols=52,
    )
    assert isinstance(out, OBSOFrame)
    assert out.obso.shape == (34, 52)
    assert out.control.shape == (34, 52)
    assert out.arrival.shape == (34, 52)
    assert out.goal.shape == (34, 52)


def test_obso_lies_in_unit_interval() -> None:
    out = compute_obso_frame(
        _tracking_with_ball_and_players(), frame_id=1, attacking_team_id="home", defending_team_id="away"
    )
    assert (out.obso >= 0.0).all()
    assert (out.obso <= 1.0).all()


def test_obso_factorises_exactly() -> None:
    """OBSO = control * arrival * goal (pointwise)."""
    out = compute_obso_frame(
        _tracking_with_ball_and_players(),
        frame_id=1,
        attacking_team_id="home",
        defending_team_id="away",
        rows=20,
        cols=30,
    )
    product = out.control * out.arrival * out.goal
    assert np.allclose(out.obso, product, atol=1e-12)


def test_hot_zone_is_near_goal() -> None:
    """With forwards in the box and ball 30m out, the peak OBSO cell should be in the
    attacking half close to the goal."""
    out = compute_obso_frame(
        _tracking_with_ball_and_players(),
        frame_id=1,
        attacking_team_id="home",
        defending_team_id="away",
        rows=68,
        cols=104,
    )
    # Coordinates of peak cell
    r, c = np.unravel_index(int(out.obso.argmax()), out.obso.shape)
    peak_x = out.xs[c]
    peak_y = out.ys[r]
    assert peak_x > 70.0
    assert 20.0 < peak_y < 48.0  # inside the width of the penalty area


def test_per_player_obso_ranks_forwards_over_midfielders() -> None:
    ranked = per_player_obso(
        _tracking_with_ball_and_players(), frame_id=1, attacking_team_id="home", defending_team_id="away"
    )
    assert len(ranked) == 3
    # h_fwd* (deep in attacking half) should beat h_mid (at x=55)
    assert ranked.iloc[0]["player_id"] in {"h_fwd1", "h_fwd2"}
    assert ranked.iloc[-1]["player_id"] == "h_mid"


def test_no_ball_in_frame_uses_uniform_arrival() -> None:
    """When the ball row is missing, arrival should be uniform (all 1s), making OBSO
    depend only on control * goal."""
    out = compute_obso_frame(
        _tracking_with_ball_and_players(ball_xy=None),
        frame_id=1,
        attacking_team_id="home",
        defending_team_id="away",
        rows=34,
        cols=52,
    )
    assert np.allclose(out.arrival, 1.0)
    assert np.allclose(out.obso, out.control * out.goal)


def test_per_player_obso_empty_when_no_attackers() -> None:
    # Tracking with only away players (no attackers)
    rows = [
        {
            "frame_id": 1,
            "team_id": "away",
            "player_id": "a1",
            "x": 50.0,
            "y": 34.0,
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": False,
            "visible": True,
        },
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
    ]
    ranked = per_player_obso(pd.DataFrame(rows), frame_id=1, attacking_team_id="home", defending_team_id="away")
    assert ranked.empty
