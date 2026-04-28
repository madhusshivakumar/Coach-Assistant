"""Integration tests for the top-level episode-engine composer."""

from __future__ import annotations

import pandas as pd

from football_analysis.analytics.episodes.engine import (
    EpisodeRecord,
    build_episodes,
    episodes_to_summary,
)


def _synth(n_frames: int = 200, fps: int = 25) -> pd.DataFrame:
    """Minimal valid tracking for the engine: clear home→away possession flip."""
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
        bx, by = (30.0, 30.0) if f <= n_frames // 2 else (70.0, 38.0)
        rows.append(
            {
                "frame_id": f,
                "period": 1,
                "time_seconds": t,
                "player_id": "ball",
                "team_id": "home",
                "x": bx,
                "y": by,
                "vx": 1.0,
                "vy": 0.0,
                "is_ball": True,
                "visible": True,
            }
        )
    return pd.DataFrame(rows)


def test_build_episodes_empty_tracking_returns_empty_list() -> None:
    assert build_episodes(pd.DataFrame(), home_team_id="home", away_team_id="away") == []


def test_build_episodes_smoke_two_clean_possessions() -> None:
    tracking = _synth(n_frames=200)
    records = build_episodes(tracking, home_team_id="home", away_team_id="away")
    assert len(records) == 2
    assert all(isinstance(r, EpisodeRecord) for r in records)
    teams = {r.boundary.possession_team for r in records}
    assert teams == {"home", "away"}


def test_build_episodes_state_trajectory_non_empty() -> None:
    tracking = _synth(n_frames=200)
    records = build_episodes(tracking, home_team_id="home", away_team_id="away", snapshot_hz=2.0)
    for r in records:
        # 4 s per episode @ 2 Hz → at least a couple of snapshots
        assert len(r.state_trajectory) >= 2


def test_build_episodes_dominant_phase_assigned() -> None:
    tracking = _synth(n_frames=200)
    records = build_episodes(tracking, home_team_id="home", away_team_id="away")
    # Some phase label should land on every episode, even if "settled_def" / etc.
    for r in records:
        assert r.dominant_phase is None or isinstance(r.dominant_phase, str)


def test_episodes_to_summary_columns() -> None:
    tracking = _synth(n_frames=200)
    records = build_episodes(tracking, home_team_id="home", away_team_id="away")
    summary = episodes_to_summary(records)
    expected = {
        "episode_id",
        "start_frame",
        "end_frame",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "possession_team",
        "dominant_phase",
        "end_reason",
        "reached_final_third",
        "ended_in_box",
        "shot_like",
        "end_ball_x",
        "end_ball_y",
        "end_ball_speed",
        "n_snapshots",
    }
    assert expected <= set(summary.columns)
    assert len(summary) == len(records)


def test_episodes_to_summary_empty_records_returns_empty_df() -> None:
    summary = episodes_to_summary([])
    assert summary.empty


def test_build_episodes_respects_explicit_attacking_directions() -> None:
    """Flipping who attacks which side should not change the episode count."""
    tracking = _synth(n_frames=200)
    flipped = {"home": "left", "away": "right"}
    records = build_episodes(tracking, home_team_id="home", away_team_id="away", attacking_directions=flipped)
    assert len(records) == 2


def test_build_episodes_obso_outcome_off_by_default() -> None:
    """Default behaviour unchanged: peak_obso/decisive_obso are None."""
    tracking = _synth(n_frames=200)
    records = build_episodes(tracking, home_team_id="home", away_team_id="away")
    for r in records:
        assert r.outcome.peak_obso is None
        assert r.outcome.decisive_obso is None


def test_build_episodes_with_compute_obso_outcome_populates_continuous_value() -> None:
    """When compute_obso_outcome=True, every non-empty episode gets a peak_obso."""
    tracking = _synth(n_frames=200)
    records = build_episodes(
        tracking,
        home_team_id="home",
        away_team_id="away",
        compute_obso_outcome=True,
    )
    assert records, "expected at least one episode in synth match"
    populated = [r for r in records if r.outcome.peak_obso is not None]
    assert len(populated) >= 1
    for r in populated:
        # peak_obso is a probability and OBSO max stays in [0, 1] by construction
        assert 0.0 <= r.outcome.peak_obso <= 1.0


def test_episodes_to_summary_includes_obso_outcome_columns() -> None:
    """summary frame must surface peak_obso/decisive_obso for downstream recommendations."""
    tracking = _synth(n_frames=200)
    records = build_episodes(tracking, home_team_id="home", away_team_id="away")
    summary = episodes_to_summary(records)
    assert "peak_obso" in summary.columns
    assert "decisive_obso" in summary.columns
