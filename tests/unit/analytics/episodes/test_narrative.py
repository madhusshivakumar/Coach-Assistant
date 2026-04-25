"""Tests for trigger-frame detection + narrative builder."""

from __future__ import annotations

import pandas as pd

from football_analysis.analytics.episodes.contribution import EpisodeAttribution
from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.episodes.narrative import (
    EpisodeNarrative,
    build_narrative,
    find_trigger_frame,
)
from football_analysis.analytics.episodes.outcomes import EpisodeOutcome
from football_analysis.analytics.episodes.segmenter import EpisodeBoundary


def _record(states: pd.DataFrame, **outcome_overrides: object) -> EpisodeRecord:
    boundary = EpisodeBoundary(
        episode_id=7,
        start_frame=1,
        end_frame=100,
        start_time_s=0.0,
        end_time_s=4.0,
        duration_s=4.0,
        possession_team="home",
        end_reason="possession_change",
    )
    out_kwargs: dict = {
        "episode_id": 7,
        "end_reason": "possession_change",
        "reached_final_third": True,
        "ended_in_box": False,
        "shot_like": False,
        "end_ball_x": 72.0,
        "end_ball_y": 30.0,
        "end_ball_speed": 4.0,
        "duration_s": 4.0,
    }
    out_kwargs.update(outcome_overrides)
    outcome = EpisodeOutcome(**out_kwargs)  # type: ignore[arg-type]
    return EpisodeRecord(boundary=boundary, outcome=outcome, state_trajectory=states, dominant_phase="progression")


def test_find_trigger_frame_first_crossing_of_final_third() -> None:
    states = pd.DataFrame(
        [
            {"frame_id": 10, "time_s": 0.4, "ball_x_oriented": 50.0},
            {"frame_id": 25, "time_s": 1.0, "ball_x_oriented": 65.0},  # not crossed yet
            {"frame_id": 40, "time_s": 1.6, "ball_x_oriented": 75.0},  # CROSSED here
            {"frame_id": 60, "time_s": 2.4, "ball_x_oriented": 85.0},
        ]
    )
    rec = _record(states)
    tf, tt = find_trigger_frame(rec)
    assert tf == 40
    assert tt == 1.6


def test_find_trigger_frame_returns_none_when_no_crossing() -> None:
    states = pd.DataFrame(
        [
            {"frame_id": 10, "time_s": 0.4, "ball_x_oriented": 30.0},
            {"frame_id": 50, "time_s": 2.0, "ball_x_oriented": 50.0},
        ]
    )
    rec = _record(states, reached_final_third=False)
    tf, tt = find_trigger_frame(rec)
    assert tf is None
    assert tt is None


def test_find_trigger_frame_returns_none_for_empty_states() -> None:
    rec = _record(pd.DataFrame())
    tf, tt = find_trigger_frame(rec)
    assert tf is None
    assert tt is None


def test_build_narrative_with_attribution_includes_top_contributors() -> None:
    states = pd.DataFrame(
        [
            {"frame_id": 40, "time_s": 1.6, "ball_x_oriented": 75.0},
            {"frame_id": 100, "time_s": 4.0, "ball_x_oriented": 95.0},
        ]
    )
    rec = _record(states, ended_in_box=True, shot_like=True)
    attr = EpisodeAttribution(
        episode_id=7,
        peak_frame=100,
        peak_obso=0.42,
        contributions={"runner": 0.15, "h1": 0.02, "h2": 0.0, "h3": -0.05},
        baseline_obso_per_player={"runner": 0.27, "h1": 0.40, "h2": 0.42, "h3": 0.47},
    )
    narr = build_narrative(rec, attr, top_k=2)
    assert isinstance(narr, EpisodeNarrative)
    assert narr.episode_id == 7
    assert narr.trigger_frame == 40
    # Top-2 by absolute magnitude: runner (0.15) and h1 (0.02 vs h3 0.05). Actually |h3|=0.05 > |h1|=0.02
    contrib_ids = [p for p, _ in narr.top_contributors]
    assert "runner" in contrib_ids
    assert "h3" in contrib_ids  # |-0.05| > |0.02|
    assert "shot-like" in narr.text
    assert "runner" in narr.text


def test_build_narrative_without_attribution_falls_back() -> None:
    states = pd.DataFrame([{"frame_id": 1, "time_s": 0.0, "ball_x_oriented": 50.0}])
    rec = _record(states, reached_final_third=False)
    narr = build_narrative(rec, attribution=None)
    assert narr.top_contributors == []
    assert "did not reach the final third" in narr.text


def test_build_narrative_with_empty_contributions_falls_back() -> None:
    states = pd.DataFrame([{"frame_id": 1, "time_s": 0.0, "ball_x_oriented": 75.0}])
    rec = _record(states)
    attr = EpisodeAttribution(
        episode_id=7,
        peak_frame=100,
        peak_obso=0.0,
        contributions={},
        baseline_obso_per_player={},
    )
    narr = build_narrative(rec, attr)
    assert narr.top_contributors == []
    assert "ended in" in narr.text
