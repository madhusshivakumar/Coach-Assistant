"""Tests for the pattern-clustering layer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.episodes.index import EpisodeIndex
from football_analysis.analytics.episodes.outcomes import EpisodeOutcome
from football_analysis.analytics.episodes.patterns import (
    PatternCluster,
    cluster_episodes,
    cluster_for_episode,
)
from football_analysis.analytics.episodes.segmenter import EpisodeBoundary


def _make_rec(eid: int, *, phase: str, end_x: float, shot_like: bool = False) -> EpisodeRecord:
    boundary = EpisodeBoundary(
        episode_id=eid,
        start_frame=eid * 100,
        end_frame=eid * 100 + 50,
        start_time_s=float(eid * 4),
        end_time_s=float(eid * 4 + 4),
        duration_s=4.0,
        possession_team="home",
        end_reason="match_end",
    )
    outcome = EpisodeOutcome(
        episode_id=eid,
        end_reason="match_end",
        reached_final_third=end_x > 70,
        ended_in_box=end_x > 88,
        shot_like=shot_like,
        end_ball_x=end_x,
        end_ball_y=34.0,
        end_ball_speed=10.0,
        duration_s=4.0,
    )
    n = 5
    states = pd.DataFrame(
        {
            "rel_time_s": np.linspace(0, 4, n),
            "ball_x_oriented": np.linspace(20, end_x, n),
            "ball_x": np.linspace(20, end_x, n),
            "ball_y": [34.0] * n,
            "ball_speed": [3.0] * n,
            "attackers_mean_x_oriented": np.linspace(40, 60, n),
            "defenders_line_height_oriented": [80.0] * n,
            "attackers_length": [30.0] * n,
            "attackers_width": [40.0] * n,
            "attackers_visible": [11] * n,
            "defenders_visible": [10] * n,
            "defenders_compactness_x": [3.0] * n,
        }
    )
    return EpisodeRecord(boundary=boundary, outcome=outcome, state_trajectory=states, dominant_phase=phase)


def test_cluster_episodes_returns_pattern_cluster_per_group() -> None:
    """6 episodes split into 2 distinct shapes → expect ≥1 cluster, with members."""
    library = [
        _make_rec(1, phase="finishing", end_x=98, shot_like=True),
        _make_rec(2, phase="finishing", end_x=96, shot_like=True),
        _make_rec(3, phase="settled_def", end_x=30),
        _make_rec(4, phase="settled_def", end_x=25),
        _make_rec(5, phase="progression", end_x=70),
        _make_rec(6, phase="progression", end_x=72),
    ]
    idx = EpisodeIndex()
    idx.fit(library)
    clusters = cluster_episodes(idx, n_clusters=3)
    assert all(isinstance(c, PatternCluster) for c in clusters)
    assert sum(c.n_episodes for c in clusters) == len(library)
    # Each cluster carries a non-empty label.
    assert all(c.label and "episodes" in c.label for c in clusters)


def test_cluster_episodes_empty_index_returns_empty_list() -> None:
    idx = EpisodeIndex()
    idx.fit([])
    assert cluster_episodes(idx) == []


def test_cluster_for_episode_finds_membership() -> None:
    library = [_make_rec(eid, phase="progression", end_x=70 + eid) for eid in range(1, 6)]
    idx = EpisodeIndex()
    idx.fit(library)
    clusters = cluster_episodes(idx, n_clusters=2)
    assert clusters
    # Every episode_id should map to exactly one cluster.
    for r in library:
        c = cluster_for_episode(clusters, r.boundary.episode_id)
        assert c is not None
        assert r.boundary.episode_id in c.episode_ids


def test_cluster_for_episode_returns_none_for_unknown_id() -> None:
    library = [_make_rec(1, phase="progression", end_x=70)]
    idx = EpisodeIndex()
    idx.fit(library)
    clusters = cluster_episodes(idx, n_clusters=1)
    assert cluster_for_episode(clusters, 999) is None


def test_cluster_episodes_n_clusters_clamps_to_population() -> None:
    """Asking for k > n produces at most n clusters."""
    library = [_make_rec(1, phase="progression", end_x=70)]
    idx = EpisodeIndex()
    idx.fit(library)
    clusters = cluster_episodes(idx, n_clusters=10)
    assert len(clusters) == 1


def test_cluster_label_includes_dominant_phase_and_pct_stats() -> None:
    """Cluster of 4 finishing episodes (3 shot-like) → label mentions finishing + 75% shot-like."""
    library = [_make_rec(eid, phase="finishing", end_x=95, shot_like=(eid != 4)) for eid in range(1, 5)]
    idx = EpisodeIndex()
    idx.fit(library)
    clusters = cluster_episodes(idx, n_clusters=1)
    assert len(clusters) == 1
    c = clusters[0]
    assert c.dominant_phase == "finishing"
    assert "finishing" in c.label
    assert c.pct_shot_like == 0.75
