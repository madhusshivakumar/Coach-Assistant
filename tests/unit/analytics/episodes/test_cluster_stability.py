"""Tests for the bootstrap Jaccard stability gate.

The v2 ship target is mean Jaccard ≥ 0.75 across bootstraps for any pattern
that lands on slide 2 of the Coach Brief. This gate is what prevents HDBSCAN
from spuriously promoting a phantom density blip to "Flick's third attacking
pattern."
"""

from __future__ import annotations

import numpy as np
import pytest

from football_analysis.analytics.episodes.cluster_hdbscan import (
    ClusterResult,
    cluster_from_distance_matrix,
)
from football_analysis.analytics.episodes.cluster_stability import (
    StabilityResult,
    bootstrap_jaccard_stability,
    jaccard,
)


# ---------------------------------------------------------------------------
# jaccard primitive
# ---------------------------------------------------------------------------


def test_jaccard_identical_sets() -> None:
    assert jaccard({1, 2, 3}, {1, 2, 3}) == 1.0


def test_jaccard_disjoint_sets() -> None:
    assert jaccard({1, 2}, {3, 4}) == 0.0


def test_jaccard_partial_overlap() -> None:
    # |∩| = 2, |∪| = 4, J = 0.5
    assert jaccard({1, 2, 3}, {2, 3, 4}) == 0.5


def test_jaccard_empty_sets_returns_zero() -> None:
    """Vacuous case — no members in either side ⇒ no overlap to measure.
    Returning 0 keeps the gate conservative (won't ship a phantom cluster)."""
    assert jaccard(set(), set()) == 0.0


def test_jaccard_one_empty() -> None:
    assert jaccard({1, 2}, set()) == 0.0


# ---------------------------------------------------------------------------
# Stability gate on hand-crafted distance matrices
# ---------------------------------------------------------------------------


def _two_blob_distance_matrix(n_per: int = 8, gap: float = 100.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = np.vstack([
        rng.normal(loc=0.0, scale=0.3, size=(n_per, 2)),
        rng.normal(loc=gap, scale=0.3, size=(n_per, 2)),
    ])
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def test_stability_high_for_clearly_separated_blobs() -> None:
    """Two tight blobs separated by 100 m → bootstrap should consistently
    recover both; mean Jaccard per cluster ≥ 0.75."""
    n_per = 8
    D = _two_blob_distance_matrix(n_per=n_per, gap=100.0)
    result = cluster_from_distance_matrix(D, list(range(2 * n_per)), min_cluster_size=3)
    stab = bootstrap_jaccard_stability(result, n_bootstraps=10, sample_frac=0.8, random_state=42, min_cluster_size=3)
    # Both reference clusters should pass the 0.75 gate
    assert all(score >= 0.75 for score in stab.cluster_jaccard.values()), stab.cluster_jaccard
    assert sorted(stab.stable_clusters) == sorted(stab.cluster_jaccard.keys())


def test_stability_low_for_noisy_random_distances() -> None:
    """Random distances → HDBSCAN may still find a "cluster", but it will not
    survive bootstrap. Some-or-all clusters should fail the gate."""
    rng = np.random.default_rng(7)
    n = 30
    raw = rng.uniform(0.0, 5.0, size=(n, n))
    D = (raw + raw.T) / 2.0
    np.fill_diagonal(D, 0.0)
    result = cluster_from_distance_matrix(D, list(range(n)), min_cluster_size=3)
    stab = bootstrap_jaccard_stability(result, n_bootstraps=10, sample_frac=0.8, random_state=11, min_cluster_size=3)
    # Random data has no real clusters; if HDBSCAN found any, none should be
    # stable enough to publish.
    if stab.cluster_jaccard:
        assert any(score < 0.75 for score in stab.cluster_jaccard.values())


def test_stability_filters_clusters_by_threshold() -> None:
    """``stable_clusters`` is just ``[c for c, j in cluster_jaccard.items() if j >= threshold]``."""
    n_per = 8
    D = _two_blob_distance_matrix(n_per=n_per, gap=100.0)
    result = cluster_from_distance_matrix(D, list(range(2 * n_per)), min_cluster_size=3)
    permissive = bootstrap_jaccard_stability(result, n_bootstraps=5, threshold=0.0, random_state=1, min_cluster_size=3)
    strict = bootstrap_jaccard_stability(result, n_bootstraps=5, threshold=0.999, random_state=1, min_cluster_size=3)
    assert set(permissive.stable_clusters) == set(permissive.cluster_jaccard.keys())
    # 0.999 is impossibly high; almost certainly no cluster passes
    assert len(strict.stable_clusters) <= len(permissive.stable_clusters)


def test_stability_is_deterministic_with_seed() -> None:
    n_per = 6
    D = _two_blob_distance_matrix(n_per=n_per, gap=80.0)
    result = cluster_from_distance_matrix(D, list(range(2 * n_per)), min_cluster_size=3)
    a = bootstrap_jaccard_stability(result, n_bootstraps=5, random_state=42, min_cluster_size=3)
    b = bootstrap_jaccard_stability(result, n_bootstraps=5, random_state=42, min_cluster_size=3)
    assert a.cluster_jaccard == b.cluster_jaccard


def test_stability_empty_cluster_result_returns_empty() -> None:
    empty = ClusterResult(episode_ids=[], cluster_labels=[], distance_matrix=np.zeros((0, 0)))
    stab = bootstrap_jaccard_stability(empty, n_bootstraps=5)
    assert isinstance(stab, StabilityResult)
    assert stab.cluster_jaccard == {}
    assert stab.stable_clusters == []
    assert stab.n_bootstraps == 5


def test_stability_skips_noise_label() -> None:
    """Noise (cluster_id == -1) is not a real cluster — it must never appear
    in cluster_jaccard or stable_clusters."""
    n_per = 8
    D = _two_blob_distance_matrix(n_per=n_per, gap=100.0)
    result = cluster_from_distance_matrix(D, list(range(2 * n_per)), min_cluster_size=3)
    stab = bootstrap_jaccard_stability(result, n_bootstraps=5, random_state=42, min_cluster_size=3)
    assert -1 not in stab.cluster_jaccard
    assert -1 not in stab.stable_clusters


def test_stability_validates_n_bootstraps_positive() -> None:
    n_per = 5
    D = _two_blob_distance_matrix(n_per=n_per, gap=80.0)
    result = cluster_from_distance_matrix(D, list(range(2 * n_per)), min_cluster_size=3)
    with pytest.raises(ValueError, match="n_bootstraps"):
        bootstrap_jaccard_stability(result, n_bootstraps=0)


def test_stability_validates_sample_frac_in_range() -> None:
    n_per = 5
    D = _two_blob_distance_matrix(n_per=n_per, gap=80.0)
    result = cluster_from_distance_matrix(D, list(range(2 * n_per)), min_cluster_size=3)
    with pytest.raises(ValueError, match="sample_frac"):
        bootstrap_jaccard_stability(result, n_bootstraps=2, sample_frac=0.0)
    with pytest.raises(ValueError, match="sample_frac"):
        bootstrap_jaccard_stability(result, n_bootstraps=2, sample_frac=1.5)


def test_stability_returns_per_cluster_score_in_zero_one_range() -> None:
    n_per = 8
    D = _two_blob_distance_matrix(n_per=n_per, gap=80.0)
    result = cluster_from_distance_matrix(D, list(range(2 * n_per)), min_cluster_size=3)
    stab = bootstrap_jaccard_stability(result, n_bootstraps=5, random_state=1, min_cluster_size=3)
    for score in stab.cluster_jaccard.values():
        assert 0.0 <= score <= 1.0


def test_stability_dataclass_fields() -> None:
    """Schema contract for downstream report layer."""
    n_per = 6
    D = _two_blob_distance_matrix(n_per=n_per, gap=80.0)
    result = cluster_from_distance_matrix(D, list(range(2 * n_per)), min_cluster_size=3)
    stab = bootstrap_jaccard_stability(result, n_bootstraps=3, random_state=1, min_cluster_size=3)
    assert hasattr(stab, "cluster_jaccard")
    assert hasattr(stab, "stable_clusters")
    assert hasattr(stab, "n_bootstraps")
    assert hasattr(stab, "threshold")
    assert stab.threshold == 0.75
