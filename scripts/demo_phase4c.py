"""Phase 4 Slice C smoke: episode retrieval + pattern library.

For the Metrica match:

1. Build the EpisodeIndex (embed every episode, fit k-NN structure).
2. For each ``shot_like`` episode, retrieve its 3 nearest neighbors and predict
   outcome distribution from those neighbors. This is what the prediction side
   of the engine actually does — find similar past states, vote on outcome.
3. Cluster the full library (k=8). Each cluster is a recurring play pattern;
   labels expose what the team's vocabulary looks like.

Outputs to ``data/features/phase4c/``:

- ``index_embeddings.parquet`` — feature matrix (one row per episode).
- ``retrieval_demo.json`` — for each shot_like episode: query_id → 3 neighbor
  ids + distances + outcome prediction.
- ``patterns.json`` — list of clusters with labels and member episode_ids.
- ``summary.json`` — counts + the same-old honest limitations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from football_analysis.analytics.episodes.embedding import (
    EPISODE_FEATURE_NAMES,
    embed_episode,
)
from football_analysis.analytics.episodes.engine import build_episodes
from football_analysis.analytics.episodes.index import EpisodeIndex
from football_analysis.analytics.episodes.patterns import cluster_episodes, cluster_for_episode
from football_analysis.config import get_settings


def load_tracking(match_id: str) -> pd.DataFrame:
    settings = get_settings()
    key = match_id.split(":", 1)[-1]
    parquets: list[Path] = []
    for d in (settings.processed_dir / "tracking").rglob(f"match_id=metrica-{key}"):
        parquets.extend(d.rglob("*.parquet"))
    if not parquets:
        raise SystemExit(f"No tracking parquet for {match_id!r}")
    return pd.concat([pd.read_parquet(p) for p in sorted(parquets)], ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--match", default="metrica:1")
    p.add_argument("--out-dir", type=Path, default=Path("data/features/phase4c"))
    p.add_argument("--k", type=int, default=3, help="neighbors per query")
    p.add_argument("--n-clusters", type=int, default=8)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracking = load_tracking(args.match)
    print(f"--- {args.match} ---")
    print(f"  tracking: {len(tracking):,} rows, {tracking['frame_id'].nunique():,} frames")

    records = build_episodes(
        tracking,
        home_team_id="home",
        away_team_id="away",
        attacking_directions={"home": "right", "away": "left"},
    )
    print(f"  built {len(records)} episodes")

    # Build the retrieval index.
    index = EpisodeIndex(k_default=args.k)
    index.fit(records)
    print(f"  index fit on {len(index)} episodes ({index.feature_dim} features each)")

    # Persist embeddings for future retrieval (so we don't re-fit per query session).
    embed_rows = []
    for r in records:
        v = embed_episode(r)
        row = {
            "episode_id": r.boundary.episode_id,
            "possession_team": r.boundary.possession_team,
            "dominant_phase": r.dominant_phase,
        }
        for name, val in zip(EPISODE_FEATURE_NAMES, v, strict=True):
            row[name] = float(val)
        embed_rows.append(row)
    pd.DataFrame(embed_rows).to_parquet(args.out_dir / "index_embeddings.parquet")
    print(f"  -> {args.out_dir / 'index_embeddings.parquet'}")

    # Retrieval demo — for every shot_like episode, find its 3 nearest neighbors
    # and predict outcome distribution from them (excluding self).
    shot_like = [r for r in records if r.outcome.shot_like]
    print(f"\n--- retrieval demo on {len(shot_like)} shot_like episodes ---")
    retrieval_results = []
    for r in shot_like:
        neighbors = index.query(r, k=args.k, exclude_self=True)
        prediction = index.predict_outcome(r, k=args.k)
        result = {
            "query_episode_id": r.boundary.episode_id,
            "query_summary": {
                "possession_team": r.boundary.possession_team,
                "duration_s": r.boundary.duration_s,
                "dominant_phase": r.dominant_phase,
                "end_ball_x": r.outcome.end_ball_x,
            },
            "neighbors": [
                {
                    "rank": n.rank,
                    "episode_id": n.record.boundary.episode_id,
                    "distance": round(n.distance, 4),
                    "possession_team": n.record.boundary.possession_team,
                    "dominant_phase": n.record.dominant_phase,
                    "shot_like": n.record.outcome.shot_like,
                    "ended_in_box": n.record.outcome.ended_in_box,
                    "reached_final_third": n.record.outcome.reached_final_third,
                }
                for n in neighbors
            ],
            "prediction": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in prediction.items()},
        }
        retrieval_results.append(result)
        print(
            f"  ep {r.boundary.episode_id} ({r.dominant_phase}, end_x={r.outcome.end_ball_x:.1f}): "
            f"top neighbors = {[n.record.boundary.episode_id for n in neighbors]}"
        )

    (args.out_dir / "retrieval_demo.json").write_text(
        json.dumps(retrieval_results, indent=2, default=str), encoding="utf-8"
    )

    # Pattern clustering.
    print(f"\n--- pattern clustering (k={args.n_clusters}) ---")
    clusters = cluster_episodes(index, n_clusters=args.n_clusters)
    for c in clusters:
        print(f"  cluster {c.cluster_id}: {c.label}")
    (args.out_dir / "patterns.json").write_text(
        json.dumps([asdict(c) for c in clusters], indent=2, default=str), encoding="utf-8"
    )

    # Per-episode → cluster mapping for downstream analysis.
    cluster_assignments = []
    for r in records:
        c = cluster_for_episode(clusters, r.boundary.episode_id)
        cluster_assignments.append(
            {
                "episode_id": r.boundary.episode_id,
                "cluster_id": c.cluster_id if c else -1,
                "cluster_label": c.label if c else "(none)",
                "shot_like": r.outcome.shot_like,
                "reached_final_third": r.outcome.reached_final_third,
                "possession_team": r.boundary.possession_team,
            }
        )
    pd.DataFrame(cluster_assignments).to_parquet(args.out_dir / "cluster_assignments.parquet")

    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "match": args.match,
                "n_episodes": len(records),
                "feature_dim": index.feature_dim,
                "k_neighbors": args.k,
                "n_clusters": len(clusters),
                "n_shot_like_queries": len(shot_like),
                "limitations": [
                    "Library = one match (~76 episodes). Retrieval predictions are tautological"
                    " until more tracking matches are ingested — k-NN of one match against itself"
                    " surfaces close neighbors *within* this match, not population priors.",
                    "Hand-crafted feature schema (~25 dims). A learned encoder (autoencoder,"
                    " contrastive) can drop in via the same embed_episode signature when N is large.",
                    "Retrieval distance uses StandardScaler-Euclidean. For larger libraries swap"
                    " sklearn.NearestNeighbors for FAISS/HNSW (one-line change in EpisodeIndex.fit).",
                    "Patterns are k-means clusters; for richer pattern discovery consider"
                    " time-warping (DTW) on full state trajectories or HDBSCAN for noise-robustness.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out_dir}")
    np.set_printoptions()  # guard against numpy print-state side-effects


if __name__ == "__main__":
    main()
