"""Tests for the episode retrieval index."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.episodes.index import EpisodeIndex, Neighbor
from football_analysis.analytics.episodes.outcomes import EpisodeOutcome
from football_analysis.analytics.episodes.segmenter import EpisodeBoundary


def _make_record(
    episode_id: int,
    *,
    shot_like: bool = False,
    ended_in_box: bool = False,
    reached_final_third: bool = False,
    dominant_phase: str = "progression",
    ball_end_x: float = 80.0,
) -> EpisodeRecord:
    boundary = EpisodeBoundary(
        episode_id=episode_id,
        start_frame=episode_id * 100,
        end_frame=episode_id * 100 + 100,
        start_time_s=float(episode_id * 4),
        end_time_s=float(episode_id * 4 + 4),
        duration_s=4.0,
        possession_team="home",
        end_reason="match_end",
    )
    outcome = EpisodeOutcome(
        episode_id=episode_id,
        end_reason="match_end",
        reached_final_third=reached_final_third,
        ended_in_box=ended_in_box,
        shot_like=shot_like,
        end_ball_x=ball_end_x,
        end_ball_y=34.0,
        end_ball_speed=8.0,
        duration_s=4.0,
    )
    n = 6
    states = pd.DataFrame(
        {
            "rel_time_s": np.linspace(0, 4, n),
            "ball_x": np.linspace(20, ball_end_x, n),
            "ball_y": [34.0] * n,
            "ball_x_oriented": np.linspace(20, ball_end_x, n),
            "ball_speed": [3.0] * n,
            "attackers_mean_x_oriented": np.linspace(40, 60, n),
            "defenders_line_height_oriented": [80.0] * n,
            "attackers_length": [35.0] * n,
            "attackers_width": [40.0] * n,
            "attackers_visible": [11] * n,
            "defenders_visible": [10] * n,
            "defenders_compactness_x": [3.0] * n,
        }
    )
    return EpisodeRecord(boundary=boundary, outcome=outcome, state_trajectory=states, dominant_phase=dominant_phase)


def _library() -> list[EpisodeRecord]:
    """Mix of attacking + non-attacking episodes for retrieval to discriminate against."""
    return [
        _make_record(
            1, shot_like=True, ended_in_box=True, reached_final_third=True, dominant_phase="finishing", ball_end_x=98.0
        ),
        _make_record(
            2, shot_like=True, ended_in_box=True, reached_final_third=True, dominant_phase="finishing", ball_end_x=95.0
        ),
        _make_record(3, shot_like=False, reached_final_third=True, dominant_phase="progression", ball_end_x=80.0),
        _make_record(4, shot_like=False, dominant_phase="settled_def", ball_end_x=30.0),
        _make_record(5, shot_like=False, dominant_phase="att_transition", ball_end_x=50.0),
        _make_record(
            6, shot_like=True, ended_in_box=True, reached_final_third=True, dominant_phase="finishing", ball_end_x=99.0
        ),
    ]


def test_index_len_zero_before_fit() -> None:
    idx = EpisodeIndex()
    assert len(idx) == 0


def test_index_fit_then_query_returns_neighbors() -> None:
    idx = EpisodeIndex(k_default=3)
    idx.fit(_library())
    assert len(idx) == 6

    # Query with a record similar to the shot_like cluster (episodes 1, 2, 6).
    query = _make_record(
        99, shot_like=True, ended_in_box=True, reached_final_third=True, dominant_phase="finishing", ball_end_x=97.0
    )
    neighbors = idx.query(query, k=3)
    assert len(neighbors) == 3
    assert all(isinstance(n, Neighbor) for n in neighbors)
    # Top hits should be the finishing episodes.
    nearest_ids = [n.record.boundary.episode_id for n in neighbors]
    finishing_ids = {1, 2, 6}
    assert finishing_ids & set(nearest_ids), f"expected at least one finishing-cluster match, got {nearest_ids}"


def test_index_predict_outcome_returns_high_p_shot_for_shot_like_query() -> None:
    idx = EpisodeIndex(k_default=3)
    idx.fit(_library())
    query = _make_record(
        99, shot_like=True, ended_in_box=True, reached_final_third=True, dominant_phase="finishing", ball_end_x=97.0
    )
    pred = idx.predict_outcome(query, k=3)
    # Library has 3 finishing-cluster episodes (1, 2, 6) which are shot_like.
    # Top-3 retrieval should pick mostly those, so p_shot_like should be high.
    assert pred["p_shot_like"] >= 0.5  # type: ignore[operator]
    assert pred["n_neighbors"] == 3
    assert isinstance(pred["neighbor_episode_ids"], list)


def test_index_query_empty_index_returns_empty_list() -> None:
    idx = EpisodeIndex()
    idx.fit([])
    query = _make_record(1)
    assert idx.query(query) == []
    assert idx.predict_outcome(query)["n_neighbors"] == 0


def test_index_query_excludes_self_match_when_query_in_library() -> None:
    """Querying with an episode that's in the index should NOT return that episode."""
    library = _library()
    idx = EpisodeIndex(k_default=2)
    idx.fit(library)
    self_query = library[0]
    neighbors = idx.query(self_query, k=2)
    nearest_ids = [n.record.boundary.episode_id for n in neighbors]
    assert self_query.boundary.episode_id not in nearest_ids


def test_index_partial_query_uses_prefix_only() -> None:
    """Partial-prefix query should still return reasonable neighbors."""
    idx = EpisodeIndex(k_default=2)
    idx.fit(_library())
    query = _make_record(
        99, shot_like=True, ended_in_box=True, reached_final_third=True, dominant_phase="finishing", ball_end_x=97.0
    )
    full_neighbors = idx.query(query, k=2)
    partial_neighbors = idx.query(query, k=2, max_rel_time_s=1.0)  # only first 1s
    # Both should return 2 neighbors; the IDs may differ since prefix-state lacks
    # the late-episode signal.
    assert len(full_neighbors) == 2
    assert len(partial_neighbors) == 2
