"""k-NN retrieval index over episode embeddings.

Given a library of past ``EpisodeRecord`` objects, the index lets you ask:

- "What past episodes look most like this one?" — full-episode similarity.
- "Given the first ``T`` seconds of an in-progress episode, what tends to follow?"
  — partial-episode prediction via retrieval.

Backed by sklearn's ``NearestNeighbors`` (BallTree by default, exact search). For
millions of episodes a swap to FAISS/HNSW is a one-line change; at our scale
(thousands per ingested match) exact search is fine and zero new dependencies.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from football_analysis.analytics.episodes.embedding import (
    EPISODE_FEATURE_NAMES,
    embed_episode,
)
from football_analysis.analytics.episodes.engine import EpisodeRecord


@dataclass(frozen=True)
class Neighbor:
    """One nearest-neighbor result."""

    record: EpisodeRecord
    distance: float
    rank: int


class EpisodeIndex:
    """In-memory k-NN index of episode embeddings.

    The index stores ``EpisodeRecord`` objects alongside their feature vectors
    so retrieval results carry full context (boundary, outcome, narrative-ready).

    The same feature schema is used at fit-time and at query-time, including the
    partial-prefix mode — this is what makes "given the first 2 s, predict what
    follows" coherent: the query vector lives in the same space as the library.
    """

    def __init__(self, k_default: int = 5, normalize: bool = True) -> None:
        self.k_default = k_default
        self.normalize = normalize
        self._records: list[EpisodeRecord] = []
        self._embeddings: np.ndarray | None = None  # shape (n, d)
        self._scaler: StandardScaler | None = None
        self._nn: NearestNeighbors | None = None

    @property
    def feature_dim(self) -> int:
        return len(EPISODE_FEATURE_NAMES)

    def __len__(self) -> int:
        return len(self._records)

    def fit(self, records: Iterable[EpisodeRecord]) -> None:
        """Embed every record and build the underlying nearest-neighbor structure."""
        self._records = list(records)
        if not self._records:
            self._embeddings = np.zeros((0, self.feature_dim))
            self._scaler = None
            self._nn = None
            return

        mat = np.vstack([embed_episode(r) for r in self._records])
        if self.normalize and len(mat) >= 2:
            self._scaler = StandardScaler().fit(mat)
            mat = self._scaler.transform(mat)
        else:
            self._scaler = None
        self._embeddings = mat

        n_neighbors_cap = min(self.k_default + 1, len(mat))
        self._nn = NearestNeighbors(n_neighbors=n_neighbors_cap)
        self._nn.fit(mat)

    def query(
        self,
        record: EpisodeRecord,
        k: int | None = None,
        max_rel_time_s: float | None = None,
        exclude_self: bool = True,
    ) -> list[Neighbor]:
        """Return the k most similar past episodes (excluding the query if it lives in the index)."""
        if self._nn is None or self._embeddings is None or len(self._records) == 0:
            return []
        k = k if k is not None else self.k_default

        x = embed_episode(record, max_rel_time_s=max_rel_time_s)
        x = x.reshape(1, -1)
        if self._scaler is not None:
            x = self._scaler.transform(x)

        # +1 so we can drop a self-match if present.
        n_query = min(k + 1, len(self._records))
        distances, indices = self._nn.kneighbors(x, n_neighbors=n_query)

        out: list[Neighbor] = []
        for d, i in zip(distances[0], indices[0], strict=True):
            r = self._records[i]
            if exclude_self and r.boundary.episode_id == record.boundary.episode_id:
                continue
            out.append(Neighbor(record=r, distance=float(d), rank=len(out)))
            if len(out) == k:
                break
        return out

    def predict_outcome(
        self,
        record: EpisodeRecord,
        k: int = 5,
        max_rel_time_s: float | None = None,
    ) -> dict[str, float | int | list[int]]:
        """Predict outcome distribution from the k nearest neighbors.

        Returns probability estimates for ``shot_like``, ``ended_in_box``, and
        ``reached_final_third``, plus the neighbor episode_ids for trace-back.
        With small library and similar matches, low-confidence (high min-distance)
        results are signal that this state has no good analog — the model
        knows when it doesn't know.
        """
        neighbors = self.query(record, k=k, max_rel_time_s=max_rel_time_s)
        if not neighbors:
            return {
                "p_shot_like": 0.0,
                "p_ended_in_box": 0.0,
                "p_reached_final_third": 0.0,
                "n_neighbors": 0,
                "max_distance": 0.0,
                "neighbor_episode_ids": [],
            }

        n = len(neighbors)
        n_shot = sum(1 for nb in neighbors if nb.record.outcome.shot_like)
        n_box = sum(1 for nb in neighbors if nb.record.outcome.ended_in_box)
        n_f3 = sum(1 for nb in neighbors if nb.record.outcome.reached_final_third)
        return {
            "p_shot_like": n_shot / n,
            "p_ended_in_box": n_box / n,
            "p_reached_final_third": n_f3 / n,
            "n_neighbors": n,
            "max_distance": float(max(nb.distance for nb in neighbors)),
            "neighbor_episode_ids": [nb.record.boundary.episode_id for nb in neighbors],
        }
