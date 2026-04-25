"""K-means clustering over episode embeddings → recurring play-pattern library.

Once we have an embedding index, clustering exposes the *vocabulary* of patterns
the team uses: "build-up through midfield," "high-press turnover," "switch to
weak side," etc. Each cluster gets a plain-English label derived from the
dominant phase + outcome stats of its members.

Usage pattern: cluster the library once after each ingestion run; query-time
similarity (in ``index.py``) uses the embeddings directly without needing the
cluster assignments — clusters are an interpretability layer on top.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.cluster import KMeans

from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.episodes.index import EpisodeIndex


@dataclass(frozen=True)
class PatternCluster:
    """One discovered pattern: cluster id, its members, and a human label."""

    cluster_id: int
    episode_ids: list[int]
    label: str  # plain-English summary
    n_episodes: int
    pct_reached_final_third: float
    pct_shot_like: float
    dominant_phase: str | None


def _describe_cluster(records: list[EpisodeRecord]) -> tuple[str, str | None, float, float]:
    """Build a one-sentence label + summary stats for a cluster of episodes."""
    n = len(records)
    if n == 0:
        return "(empty cluster)", None, 0.0, 0.0
    phase_counts: dict[str, int] = {}
    for r in records:
        key = r.dominant_phase or "(none)"
        phase_counts[key] = phase_counts.get(key, 0) + 1
    top_phase, top_n = max(phase_counts.items(), key=lambda kv: kv[1])
    pct_f3 = sum(1 for r in records if r.outcome.reached_final_third) / n
    pct_shot = sum(1 for r in records if r.outcome.shot_like) / n
    label = (
        f"{n} episodes; mostly {top_phase} ({100 * top_n / n:.0f}%); "
        f"{100 * pct_f3:.0f}% reached final third; "
        f"{100 * pct_shot:.0f}% shot-like"
    )
    return label, top_phase, pct_f3, pct_shot


def cluster_episodes(
    index: EpisodeIndex,
    n_clusters: int = 8,
    random_state: int = 42,
) -> list[PatternCluster]:
    """Run k-means on the index's embeddings and label each cluster.

    Args:
        index: a ``fit`` ``EpisodeIndex``.
        n_clusters: target cluster count; auto-clamps to ``min(n_clusters, n_episodes)``.
        random_state: for reproducibility.

    Returns:
        List of ``PatternCluster`` ordered by cluster_id. Empty if the index
        contains no records.
    """
    if len(index) == 0 or index._embeddings is None or index._embeddings.shape[0] == 0:
        return []

    mat = index._embeddings  # already scaled if normalize=True at fit time
    k = min(n_clusters, len(mat))
    if k < 1:
        return []

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(mat)

    clusters: dict[int, list[EpisodeRecord]] = {}
    for cid, rec in zip(labels.astype(int), index._records, strict=True):
        clusters.setdefault(int(cid), []).append(rec)

    out: list[PatternCluster] = []
    for cid in sorted(clusters):
        members = clusters[cid]
        label, top_phase, pct_f3, pct_shot = _describe_cluster(members)
        out.append(
            PatternCluster(
                cluster_id=cid,
                episode_ids=[r.boundary.episode_id for r in members],
                label=label,
                n_episodes=len(members),
                pct_reached_final_third=pct_f3,
                pct_shot_like=pct_shot,
                dominant_phase=top_phase,
            )
        )
    return out


def cluster_for_episode(clusters: list[PatternCluster], episode_id: int) -> PatternCluster | None:
    """Find the cluster an episode belongs to. Linear scan — fine for ``len(clusters) << 100``."""
    for c in clusters:
        if episode_id in c.episode_ids:
            return c
    return None
