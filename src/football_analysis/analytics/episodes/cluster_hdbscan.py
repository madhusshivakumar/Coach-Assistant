"""HDBSCAN-based episode clustering — replaces the old k-means path.

The v2 design rejected k-means because football episodes don't sit in spherical
clusters (research review: Le et al. 2017, Stöckl et al. 2021). HDBSCAN handles
density-based clusters of varying shape and size, and — critically for our
case — labels low-density regions as ``noise`` (cluster id ``-1``) instead of
forcing every episode into some cluster. Patterns that aren't really patterns
get filtered out automatically, which is what the Coach Brief layer wants.

The distance fed to HDBSCAN fuses two complementary signals:

1. **Feature distance** — Euclidean distance between z-scored extended feature
   vectors (``embed_episode_extended``). Captures static aggregates: pressing
   intensity, formation match, attacking-third commitment.
2. **Trajectory distance** — soft-DTW between role-anchored player
   trajectories. Captures *how* the episode moves through space-time.

Both matrices are normalised by their max non-zero entry before being
weighted-summed, so the choice of weight is dimensionless. Default weight is
``1.0`` for each, which gives equal contribution.

Public API:
    fuse_distance_matrices(D_feat, D_traj, ...)        — math primitive
    cluster_from_distance_matrix(D, episode_ids, ...)  — HDBSCAN wrapper
    cluster_episode_records(records, tracking, ...)    — end-to-end
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

from football_analysis.analytics.episodes.embedding import embed_episode_extended
from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.episodes.soft_dtw import soft_dtw_from_trajectory
from football_analysis.analytics.formations.roles import (
    FORMATION_4_3_3,
    FormationTemplate,
    episode_role_trajectory,
)


@dataclass(frozen=True)
class ClusterResult:
    """Per-episode cluster labels plus the distance matrix used to produce them.

    The matrix is preserved so the M2 stability gate (bootstrap Jaccard) can
    re-cluster sub-samples without recomputing pairwise distances — the
    expensive step is the soft-DTW pass, which is O(N²·T²).
    """

    episode_ids: list[int]
    cluster_labels: list[int]  # -1 = HDBSCAN noise
    distance_matrix: np.ndarray  # (N, N), symmetric, zero diagonal


def _safe_max(matrix: np.ndarray) -> float:
    """Largest non-zero entry, with a 1.0 fallback so we never divide by zero."""
    if matrix.size == 0:
        return 1.0
    m = float(matrix.max())
    return m if m > 0 else 1.0


def fuse_distance_matrices(
    D_feat: np.ndarray,
    D_traj: np.ndarray,
    feature_weight: float = 1.0,
    trajectory_weight: float = 1.0,
) -> np.ndarray:
    """Linearly combine feature and trajectory distance matrices.

    Each matrix is divided by its own max non-zero entry first so the two
    signals contribute on a comparable scale regardless of unit. The fused
    distance is ``feature_weight · D̃_feat + trajectory_weight · D̃_traj``.

    Args:
        D_feat: (N, N) Euclidean distance matrix on standardised features.
        D_traj: (N, N) soft-DTW matrix on role trajectories.
        feature_weight: contribution of the feature matrix.
        trajectory_weight: contribution of the trajectory matrix.

    Returns:
        (N, N) fused distance matrix. Symmetric, zero diagonal.

    Raises:
        ValueError: shapes disagree.
    """
    if D_feat.shape != D_traj.shape:
        raise ValueError(f"distance matrix shape mismatch: feat={D_feat.shape}, traj={D_traj.shape}")
    feat_norm = D_feat / _safe_max(D_feat)
    traj_norm = D_traj / _safe_max(D_traj)
    fused = feature_weight * feat_norm + trajectory_weight * traj_norm
    # Force exact symmetry in case of float-rounding drift.
    return 0.5 * (fused + fused.T)


def cluster_from_distance_matrix(
    distance_matrix: np.ndarray,
    episode_ids: list[int],
    min_cluster_size: int = 5,
    min_samples: int | None = None,
) -> ClusterResult:
    """Run HDBSCAN over a precomputed distance matrix.

    Args:
        distance_matrix: (N, N) symmetric, zero-diagonal.
        episode_ids: ids in the same order as the matrix rows/cols.
        min_cluster_size: HDBSCAN minimum cluster size. Smaller values find
            more (smaller) clusters; default 5 matches the research review's
            recommendation for ~10⁴ episode corpora.
        min_samples: HDBSCAN ``min_samples``. Defaults to ``min_cluster_size``
            when None, which is the conservative high-density choice.

    Returns:
        ``ClusterResult`` with per-episode labels (``-1`` = noise) and the
        unmodified distance matrix.

    Raises:
        ValueError: matrix not square, or rows/episode_ids length mismatch.
    """
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError(f"distance_matrix must be square; got {distance_matrix.shape}")
    n = distance_matrix.shape[0]
    if len(episode_ids) != n:
        raise ValueError(f"episode_ids length {len(episode_ids)} ≠ matrix size {n}")

    if n == 0:
        return ClusterResult(episode_ids=[], cluster_labels=[], distance_matrix=distance_matrix)
    if n == 1:
        # HDBSCAN on a single point yields an exception or degenerate result;
        # we shortcut to noise. ``min_cluster_size >= 2`` is HDBSCAN's contract.
        return ClusterResult(
            episode_ids=list(episode_ids),
            cluster_labels=[-1],
            distance_matrix=distance_matrix,
        )

    eff_min_samples = min_samples if min_samples is not None else min_cluster_size
    # HDBSCAN's min_cluster_size must be ≥ 2; clamp defensively.
    eff_mcs = max(2, min_cluster_size)
    clusterer = HDBSCAN(
        min_cluster_size=eff_mcs,
        min_samples=eff_min_samples,
        metric="precomputed",
    )
    labels = clusterer.fit_predict(distance_matrix)
    return ClusterResult(
        episode_ids=list(episode_ids),
        cluster_labels=[int(l) for l in labels],
        distance_matrix=distance_matrix,
    )


def _attacking_right(team_id: str, possessing: str, attacking_directions: dict[str, str] | None) -> bool:
    if attacking_directions is None:
        return True  # default: possessing team attacks +x
    return attacking_directions.get(possessing, "right") == "right"


def cluster_episode_records(
    records: list[EpisodeRecord],
    tracking: pd.DataFrame,
    home_team_id: str,
    away_team_id: str,
    attacking_directions: dict[str, str] | None = None,
    feature_weight: float = 1.0,
    trajectory_weight: float = 1.0,
    template: FormationTemplate = FORMATION_4_3_3,
    snapshot_hz: float = 2.0,
    gamma: float = 1.0,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
) -> ClusterResult:
    """Cluster a list of EpisodeRecords end-to-end.

    Pipeline:
      1. Build the extended feature vector for each record.
      2. Z-score the feature matrix and compute pairwise Euclidean distances.
      3. Build the role trajectory for each record's possessing team.
      4. Compute pairwise soft-DTW (O(N²) calls).
      5. Fuse the two matrices via ``fuse_distance_matrices``.
      6. HDBSCAN over the fused matrix.

    Heavy on compute for large N (soft-DTW is O(N²·T²)). M2 task #6 (stability
    gate) re-uses ``result.distance_matrix`` to bootstrap-resample without
    re-running steps 1–4.

    Args:
        records: list of ``EpisodeRecord`` from ``build_episodes``.
        tracking: canonical tracking DataFrame for the whole match (or the
            concatenation across matches if clustering across a corpus).
        home_team_id, away_team_id: needed for pressing-distance feature.
        attacking_directions: optional possessing-team → "left"|"right" map.
            Default treats the possessing team as attacking +x.
        feature_weight, trajectory_weight: fusion weights, both default 1.0.
        template, snapshot_hz, gamma: forwarded to embedding / soft-DTW.
        min_cluster_size, min_samples: forwarded to HDBSCAN.

    Returns:
        ``ClusterResult`` with episode_ids in input order, cluster labels, and
        the fused distance matrix.
    """
    if not records:
        return ClusterResult(episode_ids=[], cluster_labels=[], distance_matrix=np.zeros((0, 0)))

    n = len(records)
    episode_ids = [r.boundary.episode_id for r in records]

    # Step 1+2: feature distances on z-scored extended embeddings.
    feature_rows = np.vstack(
        [
            embed_episode_extended(
                rec,
                tracking,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                attacking_to_right=_attacking_right(
                    rec.boundary.possession_team, rec.boundary.possession_team, attacking_directions
                ),
                template=template,
                snapshot_hz=snapshot_hz,
            )
            for rec in records
        ]
    )
    # StandardScaler errors on zero-variance dims when n ≤ 1; manual safe scale.
    if n >= 2:
        scaled = StandardScaler().fit_transform(feature_rows)
    else:
        scaled = feature_rows
    diff = scaled[:, None, :] - scaled[None, :, :]
    D_feat = np.sqrt(np.sum(diff * diff, axis=2))

    # Step 3+4: trajectory distances via pairwise soft-DTW.
    trajectories: list[pd.DataFrame] = []
    for rec in records:
        possessing = rec.boundary.possession_team
        traj = episode_role_trajectory(
            tracking,
            rec.boundary,
            team_id=possessing,
            template=template,
            snapshot_hz=snapshot_hz,
            attacking_right=_attacking_right(possessing, possessing, attacking_directions),
        )
        trajectories.append(traj)

    D_traj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            t_i, t_j = trajectories[i], trajectories[j]
            if t_i.empty or t_j.empty:
                # Episodes with no usable role trajectory get max distance so
                # they don't pull any cluster toward them. They'll typically
                # land as HDBSCAN noise.
                d = float("inf")
            else:
                d = soft_dtw_from_trajectory(t_i, t_j, gamma=gamma)
            D_traj[i, j] = d
            D_traj[j, i] = d
    # Replace inf with a finite "max + 1" so HDBSCAN doesn't choke.
    finite_max = float(D_traj[np.isfinite(D_traj)].max()) if np.isfinite(D_traj).any() else 1.0
    D_traj[~np.isfinite(D_traj)] = finite_max + 1.0

    fused = fuse_distance_matrices(
        D_feat, D_traj, feature_weight=feature_weight, trajectory_weight=trajectory_weight
    )
    return cluster_from_distance_matrix(
        fused, episode_ids, min_cluster_size=min_cluster_size, min_samples=min_samples
    )
