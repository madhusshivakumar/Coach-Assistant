"""Tests for the possession-episode segmenter."""

from __future__ import annotations

import pandas as pd
import pytest

from football_analysis.analytics.episodes.segmenter import (
    EpisodeBoundary,
    segment_episodes,
)
from football_analysis.analytics.phases.classifier import classify_frames


def _synth_tracking(
    n_frames: int = 200,
    fps: int = 25,
    ball_path: dict[int, tuple[float, float]] | None = None,
    ball_visible: dict[int, bool] | None = None,
) -> pd.DataFrame:
    """Build a synthetic match: 5 home + 5 away outfielders, controllable ball.

    By default the ball sits on top of a home player for frames 1..100 and an away
    player for frames 101..n_frames so ``classify_frames`` produces a clean
    possession flip mid-match.
    """
    rows: list[dict] = []
    for f in range(1, n_frames + 1):
        t = round(f / fps, 4)
        for i in range(5):
            rows.append(
                {
                    "frame_id": f,
                    "period": 1,
                    "time_seconds": t,
                    "player_id": f"h{i}",
                    "team_id": "home",
                    "x": 30.0 + i * 0.5,
                    "y": 30.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": False,
                    "visible": True,
                }
            )
            rows.append(
                {
                    "frame_id": f,
                    "period": 1,
                    "time_seconds": t,
                    "player_id": f"a{i}",
                    "team_id": "away",
                    "x": 70.0 - i * 0.5,
                    "y": 38.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": False,
                    "visible": True,
                }
            )
        # ball default: glued to first home player for f<=100, first away after
        if ball_path and f in ball_path:
            bx, by = ball_path[f]
        elif f <= n_frames // 2:
            bx, by = 30.0, 30.0  # on top of h0
        else:
            bx, by = 70.0, 38.0  # on top of a0

        is_visible = ball_visible.get(f, True) if ball_visible else True
        if is_visible:
            rows.append(
                {
                    "frame_id": f,
                    "period": 1,
                    "time_seconds": t,
                    "player_id": "ball",
                    "team_id": "home",  # nominal team_id, ignored when is_ball
                    "x": bx,
                    "y": by,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": True,
                    "visible": True,
                }
            )
        else:
            # ball row exists but not visible — also a dead-ball signal
            rows.append(
                {
                    "frame_id": f,
                    "period": 1,
                    "time_seconds": t,
                    "player_id": "ball",
                    "team_id": "home",
                    "x": bx,
                    "y": by,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": True,
                    "visible": False,
                }
            )
    return pd.DataFrame(rows)


def test_segment_episodes_empty_input_returns_empty() -> None:
    empty = pd.DataFrame(columns=["frame_id", "time_seconds", "possession_team", "phase"])
    assert segment_episodes(empty, pd.DataFrame()) == []


def test_segment_episodes_two_clean_possessions() -> None:
    """Synth match: home holds first half, away holds second half → 2 episodes."""
    tracking = _synth_tracking(n_frames=200)
    classified = classify_frames(tracking, home_team_id="home", away_team_id="away")
    eps = segment_episodes(classified, tracking)
    assert len(eps) == 2
    assert eps[0].possession_team == "home"
    assert eps[1].possession_team == "away"
    assert eps[0].end_reason == "possession_change"
    assert eps[1].end_reason == "match_end"


def test_segment_episodes_dead_ball_window_breaks_episode() -> None:
    """A long dead-ball window splits one continuous-possession run into two episodes."""
    # Ball glued to home for the entire match, but invisible for frames 80..100 (21 frames).
    invisible = dict.fromkeys(range(80, 101), False)
    tracking = _synth_tracking(
        n_frames=200,
        ball_path=dict.fromkeys(range(1, 201), (30.0, 30.0)),
        ball_visible=invisible,
    )
    classified = classify_frames(tracking, home_team_id="home", away_team_id="away")
    eps = segment_episodes(classified, tracking, min_dead_frames=5)
    # All segments belong to "home" but the dead-ball window breaks them apart.
    home_eps = [e for e in eps if e.possession_team == "home"]
    assert len(home_eps) >= 2
    assert home_eps[0].end_reason == "out_of_play"


def test_segment_episodes_short_glitch_does_not_split() -> None:
    """A 2-frame ball glitch is below min_dead_frames=5 and should NOT split episodes."""
    invisible = {85: False, 86: False}  # only 2 frames invisible
    tracking = _synth_tracking(
        n_frames=200,
        ball_path=dict.fromkeys(range(1, 201), (30.0, 30.0)),
        ball_visible=invisible,
    )
    classified = classify_frames(tracking, home_team_id="home", away_team_id="away")
    eps = segment_episodes(classified, tracking, min_dead_frames=5)
    # Only one home episode (the glitch was tolerated).
    assert sum(1 for e in eps if e.possession_team == "home") == 1


def test_segment_episodes_assigns_increasing_episode_ids() -> None:
    tracking = _synth_tracking(n_frames=200)
    classified = classify_frames(tracking, home_team_id="home", away_team_id="away")
    eps = segment_episodes(classified, tracking)
    assert [e.episode_id for e in eps] == list(range(len(eps)))


def test_segment_episodes_all_dead_returns_no_episodes() -> None:
    """If the ball is never visible there's no episode to emit."""
    invisible = dict.fromkeys(range(1, 101), False)
    tracking = _synth_tracking(n_frames=100, ball_visible=invisible)
    classified = classify_frames(tracking, home_team_id="home", away_team_id="away")
    eps = segment_episodes(classified, tracking, min_dead_frames=5)
    assert eps == []


@pytest.mark.parametrize(
    "ep_attr", ["episode_id", "start_frame", "end_frame", "duration_s", "possession_team", "end_reason"]
)
def test_segment_episodes_boundary_carries_required_fields(ep_attr: str) -> None:
    tracking = _synth_tracking(n_frames=200)
    classified = classify_frames(tracking, home_team_id="home", away_team_id="away")
    eps = segment_episodes(classified, tracking)
    assert eps, "expected at least one episode"
    assert hasattr(eps[0], ep_attr)
    assert isinstance(eps[0], EpisodeBoundary)
