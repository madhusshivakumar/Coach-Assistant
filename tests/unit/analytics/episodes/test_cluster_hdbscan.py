"""Tests for HDBSCAN-based episode clustering.

The v2 design replaces the old k-means path in ``patterns.py`` with HDBSCAN over
a *fused* distance matrix that combines:

1. Euclidean distance on the z-scored extended feature vector
   (``EXTENDED_FEATURE_NAMES`` from embedding.py).
2. Soft-DTW on the role-anchored trajectory tensor.

HDBSCAN's ``metric="precomputed"`` mode lets us hand it the fused matrix
directly, so the math splits cleanly: fusion is testable in isolation, and the
clustering wrapper is testable on hand-crafted distance matrices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from football_analysis.analytics.episodes.cluster_hdbscan import (
    ClusterResult,
    cluster_episode_records,
    cluster_from_distance_matrix,
    fuse_distance_matrices,
)
from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.episodes.outcomes import EpisodeOutcome
from football_analysis.analytics.episodes.segmenter import EpisodeBoundary
from football_analysis.analytics.formations.roles import FORMATION_4_3_3


# ---------------------------------------------------------------------------
# Core: HDBSCAN over a precomputed distance matrix
# ---------------------------------------------------------------------------


def _two_blob_distance_matrix(n_per: int = 6, gap: float = 100.0) -> np.ndarray:
    """Synthetic distance matrix: two tight blobs of ``n_per`` points separated
    by ``gap``. HDBSCAN should recover this as 2 clusters."""
    rng = np.random.default_rng(42)
    n = 2 * n_per
    # Embed in 2D space for ground-truth, then distance.
    pts = np.vstack([
        rng.normal(loc=0.0, scale=0.5, size=(n_per, 2)),
        rng.normal(loc=gap, scale=0.5, size=(n_per, 2)),
    ])
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def test_cluster_from_distance_matrix_recovers_two_blobs() -> None:
    n_per = 6
    D = _two_blob_distance_matrix(n_per=n_per, gap=100.0)
    eids = list(range(2 * n_per))
    result = cluster_from_distance_matrix(D, eids, min_cluster_size=3)
    # Two distinct clusters (plus possibly noise = -1)
    non_noise = [c for c in result.cluster_labels if c >= 0]
    assert len(set(non_noise)) == 2
    # Each blob should be assigned to its own cluster
    first_half = result.cluster_labels[:n_per]
    second_half = result.cluster_labels[n_per:]
    # All members of each blob agree on a label
    assert len(set(first_half)) == 1
    assert len(set(second_half)) == 1
    # And they disagree across blobs
    assert first_half[0] != second_half[0]


def test_cluster_from_distance_matrix_returns_result_dataclass() -> None:
    D = _two_blob_distance_matrix(n_per=5, gap=50.0)
    eids = list(range(10))
    result = cluster_from_distance_matrix(D, eids)
    assert isinstance(result, ClusterResult)
    assert len(result.cluster_labels) == 10
    assert result.episode_ids == eids
    # Distance matrix is preserved for stability-gate consumption
    assert result.distance_matrix.shape == (10, 10)


def test_cluster_from_distance_matrix_empty_input() -> None:
    result = cluster_from_distance_matrix(np.zeros((0, 0)), [], min_cluster_size=3)
    assert result.cluster_labels == []
    assert result.episode_ids == []
    assert result.distance_matrix.shape == (0, 0)


def test_cluster_from_distance_matrix_single_episode_marked_noise() -> None:
    """One episode → can't form a cluster (min_cluster_size=2 minimum) → noise."""
    result = cluster_from_distance_matrix(np.array([[0.0]]), [42], min_cluster_size=2)
    assert result.cluster_labels == [-1]
    assert result.episode_ids == [42]


def test_cluster_from_distance_matrix_validates_square() -> None:
    with pytest.raises(ValueError, match="square"):
        cluster_from_distance_matrix(np.zeros((3, 4)), [1, 2, 3], min_cluster_size=2)


def test_cluster_from_distance_matrix_validates_eids_length() -> None:
    D = np.zeros((3, 3))
    with pytest.raises(ValueError, match="episode_ids"):
        cluster_from_distance_matrix(D, [1, 2], min_cluster_size=2)


# ---------------------------------------------------------------------------
# Distance fusion: combine feature-distance and trajectory-distance matrices
# ---------------------------------------------------------------------------


def test_fuse_distance_matrices_normalises_by_max() -> None:
    """Both matrices contribute on a comparable scale: each is divided by its
    max non-zero entry before being weighted-summed."""
    feat = np.array([[0.0, 2.0], [2.0, 0.0]])  # max = 2
    traj = np.array([[0.0, 200.0], [200.0, 0.0]])  # max = 200
    fused = fuse_distance_matrices(feat, traj, feature_weight=1.0, trajectory_weight=1.0)
    # After normalisation both [i,j] = 1.0 ⇒ fused = 2.0
    assert abs(fused[0, 1] - 2.0) < 1e-9
    assert fused[0, 0] == 0.0


def test_fuse_distance_matrices_respects_weights() -> None:
    feat = np.array([[0.0, 1.0], [1.0, 0.0]])
    traj = np.array([[0.0, 1.0], [1.0, 0.0]])
    only_feat = fuse_distance_matrices(feat, traj, feature_weight=1.0, trajectory_weight=0.0)
    only_traj = fuse_distance_matrices(feat, traj, feature_weight=0.0, trajectory_weight=1.0)
    half_half = fuse_distance_matrices(feat, traj, feature_weight=0.5, trajectory_weight=0.5)
    assert abs(only_feat[0, 1] - 1.0) < 1e-9
    assert abs(only_traj[0, 1] - 1.0) < 1e-9
    assert abs(half_half[0, 1] - 1.0) < 1e-9  # 0.5*1 + 0.5*1


def test_fuse_distance_matrices_validates_shapes_match() -> None:
    feat = np.zeros((3, 3))
    traj = np.zeros((4, 4))
    with pytest.raises(ValueError, match="shape"):
        fuse_distance_matrices(feat, traj)


def test_fuse_handles_all_zero_matrix_without_division_by_zero() -> None:
    """Edge case: if every distance in one matrix is 0 (single-record corpus or
    all-identical episodes), the normaliser must not divide by zero."""
    feat = np.zeros((3, 3))
    traj = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    fused = fuse_distance_matrices(feat, traj, feature_weight=1.0, trajectory_weight=1.0)
    assert np.isfinite(fused).all()


# ---------------------------------------------------------------------------
# Integration: from EpisodeRecord list end-to-end
# ---------------------------------------------------------------------------


def _record(episode_id: int, ball_x: float, dominant_phase: str = "progression") -> EpisodeRecord:
    """Build a minimal record with a flat state_trajectory at ``ball_x``."""
    boundary = EpisodeBoundary(
        episode_id=episode_id,
        start_frame=1 + (episode_id - 1) * 10,
        end_frame=10 + (episode_id - 1) * 10,
        start_time_s=(1 + (episode_id - 1) * 10) / 25.0,
        end_time_s=(10 + (episode_id - 1) * 10) / 25.0,
        duration_s=9 / 25.0,
        possession_team="home",
        end_reason="match_end",
    )
    outcome = EpisodeOutcome(
        episode_id=episode_id,
        end_reason="match_end",
        reached_final_third=ball_x > 70.0,
        ended_in_box=False,
        shot_like=False,
        end_ball_x=ball_x,
        end_ball_y=34.0,
        end_ball_speed=2.0,
        duration_s=9 / 25.0,
    )
    states = pd.DataFrame(
        {
            "rel_time_s": np.linspace(0.0, 0.36, 5),
            "ball_x": [ball_x] * 5,
            "ball_y": [34.0] * 5,
            "ball_x_oriented": [ball_x] * 5,
            "ball_speed": [2.0] * 5,
            "attackers_mean_x_oriented": [ball_x] * 5,
            "defenders_line_height_oriented": [85.0] * 5,
            "attackers_length": [40.0] * 5,
            "attackers_width": [50.0] * 5,
            "attackers_visible": [11] * 5,
            "defenders_visible": [10] * 5,
            "defenders_compactness_x": [3.5] * 5,
        }
    )
    return EpisodeRecord(
        boundary=boundary, outcome=outcome, state_trajectory=states, dominant_phase=dominant_phase
    )


def _full_team_at_offset(frame_id: int, t: float, x_offset: float) -> list[dict]:
    tpl = FORMATION_4_3_3
    rows: list[dict] = []
    for i, (tx, ty) in enumerate(zip(tpl.xs, tpl.ys, strict=True)):
        rows.append(
            {
                "frame_id": frame_id, "period": 1, "time_seconds": t,
                "player_id": f"home_{i}", "team_id": "home",
                "x": tx + x_offset, "y": ty,
                "vx": 0.0, "vy": 0.0, "is_ball": False, "visible": True,
            }
        )
    rows.append(
        {
            "frame_id": frame_id, "period": 1, "time_seconds": t,
            "player_id": "home_GK", "team_id": "home",
            "x": 5.0 + x_offset, "y": 34.0,
            "vx": 0.0, "vy": 0.0, "is_ball": False, "visible": True,
        }
    )
    rows.append(
        {
            "frame_id": frame_id, "period": 1, "time_seconds": t,
            "player_id": "away_0", "team_id": "away",
            "x": 100.0, "y": 34.0,
            "vx": 0.0, "vy": 0.0, "is_ball": False, "visible": True,
        }
    )
    rows.append(
        {
            "frame_id": frame_id, "period": 1, "time_seconds": t,
            "player_id": "ball", "team_id": None,
            "x": 30.0 + x_offset, "y": 34.0, "vx": 0.0, "vy": 0.0,
            "is_ball": True, "visible": True,
        }
    )
    return rows


def _tracking_for_records(n_per_blob: int = 4) -> tuple[list[EpisodeRecord], pd.DataFrame]:
    """Two blobs of episodes — one set with the team at low x (deep build-up)
    and one with the team shifted forward by 50 m (final-third entry)."""
    records: list[EpisodeRecord] = []
    rows: list[dict] = []
    fps = 25
    for ep_idx in range(n_per_blob):
        episode_id = ep_idx + 1
        ball_x = 30.0
        for f_local in range(10):
            frame_id = (episode_id - 1) * 10 + f_local + 1
            t = frame_id / fps
            rows.extend(_full_team_at_offset(frame_id, t, x_offset=0.0))
        records.append(_record(episode_id, ball_x=ball_x))
    for ep_idx in range(n_per_blob):
        episode_id = n_per_blob + ep_idx + 1
        ball_x = 80.0
        for f_local in range(10):
            frame_id = (episode_id - 1) * 10 + f_local + 1
            t = frame_id / fps
            rows.extend(_full_team_at_offset(frame_id, t, x_offset=50.0))
        records.append(_record(episode_id, ball_x=ball_x))
    return records, pd.DataFrame(rows)


def test_cluster_episode_records_recovers_two_blob_pattern() -> None:
    """Two clearly different attacking patterns → HDBSCAN finds two clusters."""
    records, tracking = _tracking_for_records(n_per_blob=5)
    result = cluster_episode_records(
        records,
        tracking,
        home_team_id="home",
        away_team_id="away",
        min_cluster_size=3,
    )
    non_noise = sorted({c for c in result.cluster_labels if c >= 0})
    assert len(non_noise) == 2
    # Inspect blob assignments — first 5 belong to one cluster, last 5 to another
    first_blob_labels = result.cluster_labels[:5]
    second_blob_labels = result.cluster_labels[5:]
    assert len({l for l in first_blob_labels if l >= 0}) == 1
    assert len({l for l in second_blob_labels if l >= 0}) == 1


def test_cluster_episode_records_empty_input_returns_empty_result() -> None:
    result = cluster_episode_records(
        [],
        pd.DataFrame(),
        home_team_id="home",
        away_team_id="away",
    )
    assert result.cluster_labels == []
    assert result.episode_ids == []
    assert result.distance_matrix.shape == (0, 0)


def test_cluster_episode_records_preserves_episode_id_order() -> None:
    """The labels[i] corresponds to records[i].boundary.episode_id — order
    preserved so callers can zip records and labels directly."""
    records, tracking = _tracking_for_records(n_per_blob=3)
    result = cluster_episode_records(
        records, tracking, home_team_id="home", away_team_id="away", min_cluster_size=2
    )
    expected_eids = [r.boundary.episode_id for r in records]
    assert result.episode_ids == expected_eids


def test_cluster_episode_records_returns_distance_matrix_for_stability_gate() -> None:
    """M2 task #6 (Jaccard stability gate) reads ``result.distance_matrix`` to
    bootstrap-resample the corpus. The matrix must be there and consistent."""
    records, tracking = _tracking_for_records(n_per_blob=3)
    result = cluster_episode_records(
        records, tracking, home_team_id="home", away_team_id="away", min_cluster_size=2
    )
    n = len(records)
    assert result.distance_matrix.shape == (n, n)
    # Symmetry — fused distance is symmetric
    np.testing.assert_array_almost_equal(result.distance_matrix, result.distance_matrix.T)
    # Diagonal is zero (or very close)
    assert np.allclose(np.diag(result.distance_matrix), 0.0, atol=1e-6)
