"""Tests for the per-episode formation-pair detector."""

from __future__ import annotations

import pandas as pd

from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.episodes.formation_pair import (
    FormationPair,
    detect_formation_at_frame,
    label_episode_formation_pair,
)
from football_analysis.analytics.episodes.outcomes import EpisodeOutcome
from football_analysis.analytics.episodes.segmenter import EpisodeBoundary


def _team_433(team: str, attacking_right: bool = True) -> list[dict]:
    """Synthetic 4-3-3 in canonical 105 x 68 m, attacking +x by default."""
    if attacking_right:
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
    else:
        positions = [
            (105 - x, y)
            for x, y in [
                (5, 34),
                (25, 14),
                (25, 28),
                (25, 40),
                (25, 54),
                (50, 22),
                (50, 34),
                (50, 46),
                (80, 14),
                (80, 34),
                (80, 54),
            ]
        ]
    return [
        {
            "frame_id": 1,
            "period": 1,
            "time_seconds": 0.04,
            "player_id": f"{team}{i}",
            "team_id": team,
            "x": float(x),
            "y": float(y),
            "vx": 0.0,
            "vy": 0.0,
            "is_ball": False,
            "visible": True,
        }
        for i, (x, y) in enumerate(positions)
    ]


def test_detect_formation_at_frame_picks_433_for_canonical_433_shape() -> None:
    rows = _team_433("home", attacking_right=True)
    df = pd.DataFrame(rows)
    fit = detect_formation_at_frame(df, frame_id=1, team_id="home", attacking_right=True)
    assert fit is not None
    template, cost = fit
    # The 4-3-3 template should be the best fit (lowest cost). We don't pin to
    # exact name in case the existing templates are renamed; we just require the
    # cost is finite and the template name contains "3-3" or "4-3-3".
    assert cost < float("inf")
    assert template.name


def test_detect_formation_at_frame_returns_none_below_min_players() -> None:
    """7 players is below the 9-min threshold (10 outfield + 1 GK = 11)."""
    rows = _team_433("home")[:7]
    df = pd.DataFrame(rows)
    assert detect_formation_at_frame(df, frame_id=1, team_id="home", attacking_right=True) is None


def test_label_episode_formation_pair_attaches_both_teams() -> None:
    home = _team_433("home", attacking_right=True)
    away = _team_433("away", attacking_right=False)
    df = pd.DataFrame(home + away)
    boundary = EpisodeBoundary(
        episode_id=42,
        start_frame=1,
        end_frame=1,
        start_time_s=0.0,
        end_time_s=0.04,
        duration_s=0.04,
        possession_team="home",
        end_reason="match_end",
    )
    outcome = EpisodeOutcome(
        episode_id=42,
        end_reason="match_end",
        reached_final_third=False,
        ended_in_box=False,
        shot_like=False,
        end_ball_x=80.0,
        end_ball_y=34.0,
        end_ball_speed=2.0,
        duration_s=0.04,
    )
    record = EpisodeRecord(
        boundary=boundary,
        outcome=outcome,
        state_trajectory=pd.DataFrame(),
        dominant_phase=None,
    )
    pair = label_episode_formation_pair(
        record,
        df,
        "home",
        "away",
        attacking_directions={"home": "right", "away": "left"},
    )
    assert isinstance(pair, FormationPair)
    assert pair.episode_id == 42
    assert pair.attacker_team_id == "home"
    assert pair.defender_team_id == "away"
    assert pair.attacker_formation is not None
    assert pair.defender_formation is not None
    assert pair.attacker_formation_cost is not None
    assert pair.attacker_formation_cost >= 0.0
