"""Bootstrap Jaccard stability gate for HDBSCAN clusters.

A cluster only earns a slot on slide 2 of the Coach Brief if it survives
bootstrap re-clustering: drop a fraction of episodes, re-run HDBSCAN on the
preserved distance sub-matrix, and check that the *same* episodes still land
together. The Jaccard overlap between original and bootstrap memberships,
averaged across bootstraps, is the stability score.

The v2 ship target is mean Jaccard ≥ 0.75 (the threshold default below). This
is the single most important guard against shipping spurious patterns —
HDBSCAN on football-shaped data WILL find density blips that aren't real, and
those blips don't survive a bootstrap.

Implementation note: we re-use the precomputed distance matrix from
``ClusterResult`` for every bootstrap. Soft-DTW pairwise computation (the
expensive step in M2 #5) does NOT run again — bootstrap re-clustering is
cheap, O(n_bootstraps · n²) for the HDBSCAN call and O(n_bootstraps · K) for
the Jaccard match. On a 10⁴ episode corpus, 100 bootstraps run in seconds
once the distance matrix exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from football_analysis.analytics.episodes.cluster_hdbscan import (
    ClusterResult,
    cluster_from_distance_matrix,
)


@dataclass(frozen=True)
class StabilityResult:
    """Per-cluster mean Jaccard plus the threshold-filtered survivor list.

    Fields:
        cluster_jaccard: ``{cluster_id: mean_jaccard_across_bootstraps}``.
            Noise (-1) is excluded.
        stable_clusters: cluster_ids whose mean Jaccard ≥ ``threshold``.
        n_bootstraps: how many bootstrap iterations were run.
        threshold: the cutoff used for ``stable_clusters``.
    """

    cluster_jaccard: dict[int, float]
    stable_clusters: list[int]
    n_bootstraps: int
    threshold: float


def jaccard(a: set[int], b: set[int]) -> float:
    """Jaccard index ``|A ∩ B| / |A ∪ B|``. Returns 0.0 when both are empty
    (vacuous case — keeps the gate conservative)."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def _members_by_cluster(labels: list[int], indices: list[int] | None = None) -> dict[int, set[int]]:
    """Group indices by cluster label, dropping noise (-1).

    Args:
        labels: cluster labels in input order.
        indices: indices to use as the set members. If None, uses positional
            indices 0..len(labels)-1. Pass the bootstrap subsample's original
            indices when scoring against a reference clustering.
    """
    out: dict[int, set[int]] = {}
    use_indices = indices if indices is not None else list(range(len(labels)))
    for i, lbl in zip(use_indices, labels, strict=True):
        if lbl < 0:
            continue
        out.setdefault(int(lbl), set()).add(int(i))
    return out


def _best_jaccard_match(
    reference_members: set[int],
    bootstrap_groups: dict[int, set[int]],
    bootstrap_subsample: set[int],
) -> float:
    """Maximum Jaccard between a reference cluster and any bootstrap cluster.

    Args:
        reference_members: episode indices in the reference cluster.
        bootstrap_groups: cluster_id → episode indices for the bootstrap.
        bootstrap_subsample: the indices that were KEPT in this bootstrap. We
            score the reference cluster's intersection with the subsample —
            episodes that were dropped can't be expected to match anything.
    """
    # Only the reference members that survived the subsample are eligible.
    eligible = reference_members & bootstrap_subsample
    if not eligible:
        return 0.0
    if not bootstrap_groups:
        return 0.0
    return max(jaccard(eligible, members) for members in bootstrap_groups.values())


def bootstrap_jaccard_stability(
    cluster_result: ClusterResult,
    n_bootstraps: int = 20,
    sample_frac: float = 0.8,
    threshold: float = 0.75,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    random_state: int = 42,
) -> StabilityResult:
    """Run the bootstrap Jaccard test on an existing ``ClusterResult``.

    Args:
        cluster_result: output of ``cluster_episode_records`` (or
            ``cluster_from_distance_matrix``). Both labels and the precomputed
            distance matrix are required.
        n_bootstraps: number of bootstrap iterations. 20 is the practical
            minimum; 50–100 is the research-paper standard.
        sample_frac: fraction of episodes kept in each bootstrap. 0.8 is the
            sklearn-cluster-stability default and matches Le et al. 2017.
        threshold: cutoff for ``stable_clusters``. Default 0.75 matches the v2
            ship target; pass a lower value to inspect borderline clusters.
        min_cluster_size, min_samples: forwarded to HDBSCAN for the bootstrap
            re-clustering. Should usually match the values used to produce
            ``cluster_result``.
        random_state: seed for the bootstrap subsample draws.

    Returns:
        ``StabilityResult`` with per-cluster mean Jaccard and the
        threshold-filtered ``stable_clusters`` list.

    Raises:
        ValueError: ``n_bootstraps`` ≤ 0 or ``sample_frac`` ∉ (0, 1].
    """
    if n_bootstraps <= 0:
        raise ValueError(f"n_bootstraps must be > 0; got {n_bootstraps}")
    if not (0.0 < sample_frac <= 1.0):
        raise ValueError(f"sample_frac must be in (0, 1]; got {sample_frac}")

    n = len(cluster_result.episode_ids)
    if n == 0:
        return StabilityResult(
            cluster_jaccard={}, stable_clusters=[], n_bootstraps=n_bootstraps, threshold=threshold
        )

    reference_groups = _members_by_cluster(cluster_result.cluster_labels)
    if not reference_groups:
        # No real clusters to score (everything was noise).
        return StabilityResult(
            cluster_jaccard={}, stable_clusters=[], n_bootstraps=n_bootstraps, threshold=threshold
        )

    rng = np.random.default_rng(random_state)
    sub_size = max(2, int(round(n * sample_frac)))

    # Accumulate per-cluster Jaccard across bootstraps. We average rather than
    # take min so a single unlucky bootstrap doesn't sink an otherwise-stable
    # cluster — the v2 design follows the cluster-stability literature here.
    accum: dict[int, list[float]] = {cid: [] for cid in reference_groups}

    distance_matrix = cluster_result.distance_matrix
    for _ in range(n_bootstraps):
        subsample = sorted(rng.choice(n, size=sub_size, replace=False).tolist())
        sub_D = distance_matrix[np.ix_(subsample, subsample)]
        # Re-cluster the sub-matrix; episode_ids list is positional and unused
        # for matching, so we pass placeholders.
        sub_result = cluster_from_distance_matrix(
            sub_D,
            episode_ids=list(subsample),
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        bootstrap_groups = _members_by_cluster(sub_result.cluster_labels, indices=subsample)
        subsample_set = set(subsample)
        for cid, ref_members in reference_groups.items():
            score = _best_jaccard_match(ref_members, bootstrap_groups, subsample_set)
            accum[cid].append(score)

    cluster_jaccard = {cid: float(np.mean(scores)) for cid, scores in accum.items()}
    stable_clusters = sorted(cid for cid, score in cluster_jaccard.items() if score >= threshold)

    return StabilityResult(
        cluster_jaccard=cluster_jaccard,
        stable_clusters=stable_clusters,
        n_bootstraps=n_bootstraps,
        threshold=threshold,
    )
