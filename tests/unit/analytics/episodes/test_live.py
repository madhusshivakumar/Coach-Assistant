"""Tests for the streaming LiveEpisodeEngine.

These verify reliability scaffolding (refusal-to-predict, deterministic replay,
calibration logging) and basic state-machine correctness on synthetic streams.
"""

from __future__ import annotations

import pandas as pd
import pytest

from football_analysis.analytics.episodes.engine import build_episodes
from football_analysis.analytics.episodes.index import EpisodeIndex
from football_analysis.analytics.episodes.live import (
    DEFAULT_BUFFER_FRAMES,
    LiveEpisodeEngine,
    LiveSnapshot,
)


def _synth_match(n_frames: int = 200) -> pd.DataFrame:
    """Synthetic 8-second match with two clean possessions."""
    rows: list[dict] = []
    for f in range(1, n_frames + 1):
        t = round(f / 25, 4)
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


def _frames(tracking: pd.DataFrame):
    """Yield per-frame_id slices in increasing order — replay primitive."""
    for fid in sorted(tracking["frame_id"].unique()):
        yield tracking[tracking["frame_id"] == fid]


@pytest.fixture
def fitted_index() -> EpisodeIndex:
    tracking = _synth_match(n_frames=200)
    records = build_episodes(tracking, "home", "away")
    idx = EpisodeIndex(k_default=2)
    idx.fit(records)
    return idx


def test_live_engine_empty_frame_returns_empty_snapshot(fitted_index: EpisodeIndex) -> None:
    engine = LiveEpisodeEngine(fitted_index, "home", "away")
    snap = engine.on_frame(pd.DataFrame())
    assert isinstance(snap, LiveSnapshot)
    assert snap.frame_id == -1
    assert snap.possession_team is None
    assert snap.current_episode_id == -1


def test_live_engine_processes_full_replay_without_error(fitted_index: EpisodeIndex) -> None:
    """End-to-end: feeding 200 frames produces 200 valid snapshots."""
    tracking = _synth_match(n_frames=200)
    engine = LiveEpisodeEngine(fitted_index, "home", "away", obso_every_k_frames=20, retrieval_every_k_frames=40)
    snaps = [engine.on_frame(rows) for rows in _frames(tracking)]
    assert len(snaps) == 200
    assert all(isinstance(s, LiveSnapshot) for s in snaps)


def test_live_engine_eventually_detects_active_episode(fitted_index: EpisodeIndex) -> None:
    """After sticky-frames buffer fills, possession should be classified."""
    tracking = _synth_match(n_frames=200)
    engine = LiveEpisodeEngine(fitted_index, "home", "away", obso_every_k_frames=20, retrieval_every_k_frames=40)
    saw_active = False
    for rows in _frames(tracking):
        snap = engine.on_frame(rows)
        if snap.current_episode_id >= 0:
            saw_active = True
            break
    assert saw_active, "engine never reported an active episode"


def test_live_engine_replay_is_deterministic(fitted_index: EpisodeIndex) -> None:
    """Same input → same snapshots. Required for testing + backtesting."""
    tracking = _synth_match(n_frames=200)
    snaps_1 = []
    snaps_2 = []
    for engine_snaps in (snaps_1, snaps_2):
        engine = LiveEpisodeEngine(fitted_index, "home", "away", obso_every_k_frames=20, retrieval_every_k_frames=40)
        for rows in _frames(tracking):
            engine_snaps.append(engine.on_frame(rows))
    # Compare frame_id, possession_team, current_episode_id at each tick.
    for s1, s2 in zip(snaps_1, snaps_2, strict=True):
        assert s1.frame_id == s2.frame_id
        assert s1.possession_team == s2.possession_team
        assert s1.current_episode_id == s2.current_episode_id
        assert abs(s1.threat_level - s2.threat_level) < 1e-9


def test_live_engine_resolves_predictions_when_episode_closes(fitted_index: EpisodeIndex) -> None:
    """After replay finishes, the prediction log has resolved outcomes for any
    closed episode that received a prediction during its lifetime."""
    tracking = _synth_match(n_frames=200)
    engine = LiveEpisodeEngine(fitted_index, "home", "away", obso_every_k_frames=10, retrieval_every_k_frames=20)
    for rows in _frames(tracking):
        engine.on_frame(rows)
    completed = engine.completed_episodes()
    log = engine.calibration_log()
    # Should have closed at least one episode (possession flips at midmatch).
    assert len(completed) >= 1
    # Every prediction tied to a closed episode should have resolved outcomes.
    closed_ids = {r.boundary.episode_id for r in completed}
    for p in log:
        if p.episode_id in closed_ids:
            assert p.realized_shot_like is not None, (
                f"prediction at frame {p.frame_id} for closed ep {p.episode_id} unresolved"
            )


def test_live_engine_low_confidence_when_no_close_analog(fitted_index: EpisodeIndex) -> None:
    """A tiny library means most queries find distant neighbors; confidence
    should be tagged ``low`` so a dashboard can suppress display."""
    tracking = _synth_match(n_frames=200)
    # Reduce confidence threshold to provoke a low-conf flag.
    engine = LiveEpisodeEngine(
        fitted_index,
        "home",
        "away",
        confidence_distance_threshold=0.001,
        obso_every_k_frames=20,
        retrieval_every_k_frames=40,
    )
    saw_low_confidence = False
    for rows in _frames(tracking):
        snap = engine.on_frame(rows)
        if snap.predicted.get("confidence") == "low":
            saw_low_confidence = True
            break
    assert saw_low_confidence, "expected at least one low-confidence prediction"


def test_live_engine_buffer_caps_at_configured_size(fitted_index: EpisodeIndex) -> None:
    """The rolling buffer should not grow without bound."""
    tracking = _synth_match(n_frames=600)  # > default buffer
    engine = LiveEpisodeEngine(
        fitted_index, "home", "away", buffer_frames=100, obso_every_k_frames=50, retrieval_every_k_frames=100
    )
    for rows in _frames(tracking):
        engine.on_frame(rows)
    # Buffer's frame_id range should span at most ~buffer_frames frames.
    span = engine._buffer["frame_id"].max() - engine._buffer["frame_id"].min()
    assert span <= 100 + 1, f"buffer not capped: span={span}"


def test_default_buffer_frames_is_sane() -> None:
    assert DEFAULT_BUFFER_FRAMES >= 50, "buffer should be at least 2 s at 25 Hz"
