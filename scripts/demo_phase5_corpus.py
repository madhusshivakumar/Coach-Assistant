# ruff: noqa: PLR0912, PLR0915
"""Phase 5 corpus demo: build episodes across the full multi-source tracking
library, build the cross-match retrieval index, and characterise it.

Outputs to ``data/features/phase5_corpus/``:

- ``episode_summary.parquet`` — every episode from every match, flat
- ``embedding_matrix.parquet`` — per-episode 24-dim feature vector
- ``corpus_summary.json`` — counts per source + sanity stats
- ``cross_source_neighbors.json`` — for each shot_like episode in the corpus,
  list its top-3 nearest neighbors *with* their source. The honest test of
  cross-match retrieval: do shot_like episodes from Metrica retrieve shot_like
  episodes from SoccerNet? If yes, retrieval works across leagues + collection
  styles. If no, the embedding is overfit to one source's statistical quirks
  and we know what to fix.
- ``patterns.json`` — k-means cluster labels over the full corpus

This is the slice that proves the retrieval architecture isn't single-match
tautological any more.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from football_analysis.analytics.episodes.embedding import (
    EPISODE_FEATURE_NAMES,
    embed_episode,
)
from football_analysis.analytics.episodes.engine import build_episodes
from football_analysis.analytics.episodes.index import EpisodeIndex
from football_analysis.analytics.episodes.patterns import cluster_episodes
from football_analysis.config import get_settings


def _source_of(match_id: str) -> str:
    if match_id.startswith("metrica"):
        return "Metrica"
    if match_id.startswith("skillcorner"):
        return "SkillCorner"
    if match_id.startswith("soccernet"):
        return "SoccerNet"
    return "unknown"


def load_all_tracking_groups() -> list[tuple[str, pd.DataFrame]]:
    """For each match in ``data/processed/tracking/``, return (canonical_match_id, df).

    Concatenates the per-period parquets per match.
    """
    settings = get_settings()
    root = settings.processed_dir / "tracking"
    out: list[tuple[str, pd.DataFrame]] = []
    for match_dir in sorted(root.rglob("match_id=*")):
        match_label = match_dir.name.removeprefix("match_id=")
        parts = sorted(match_dir.glob("period=*.parquet"))
        if not parts:
            continue
        try:
            df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        except Exception as e:
            print(f"  skipped {match_label}: {e}")
            continue
        out.append((match_label, df))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/features/phase5_corpus"))
    p.add_argument("--max-matches", type=int, default=None, help="cap how many matches to load (smoke / debug)")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    matches = load_all_tracking_groups()
    if args.max_matches:
        matches = matches[: args.max_matches]
    if not matches:
        raise SystemExit("no tracking matches found under data/processed/tracking/")
    print(f"corpus: {len(matches)} matches across {len({_source_of(m) for m, _ in matches})} sources")
    for source, count in Counter(_source_of(m) for m, _ in matches).items():
        print(f"  {source}: {count}")

    # 1. Build episodes per match
    all_records: list = []
    record_source: dict[int, str] = {}
    record_match: dict[int, str] = {}
    record_id_counter = 0
    t0 = time.time()
    for match_label, df in matches:
        try:
            recs = build_episodes(df, home_team_id="home", away_team_id="away")
        except Exception as e:
            print(f"  build_episodes failed for {match_label}: {e}")
            continue
        for r in recs:
            new_id = record_id_counter
            # Re-id so episodes have globally unique IDs across the corpus.
            r_renumbered = type(r)(
                boundary=type(r.boundary)(
                    episode_id=new_id,
                    start_frame=r.boundary.start_frame,
                    end_frame=r.boundary.end_frame,
                    start_time_s=r.boundary.start_time_s,
                    end_time_s=r.boundary.end_time_s,
                    duration_s=r.boundary.duration_s,
                    possession_team=r.boundary.possession_team,
                    end_reason=r.boundary.end_reason,
                ),
                outcome=r.outcome,
                state_trajectory=r.state_trajectory,
                dominant_phase=r.dominant_phase,
            )
            all_records.append(r_renumbered)
            record_source[new_id] = _source_of(match_label)
            record_match[new_id] = match_label
            record_id_counter += 1
        print(f"  {match_label} ({_source_of(match_label)}): +{len(recs)} episodes")
    elapsed = time.time() - t0
    print(f"\nbuilt {len(all_records)} episodes in {elapsed:.1f}s")

    # 2. Episode summary
    rows = []
    for r in all_records:
        rows.append(
            {
                "episode_id": r.boundary.episode_id,
                "match_id": record_match[r.boundary.episode_id],
                "source": record_source[r.boundary.episode_id],
                "possession_team": r.boundary.possession_team,
                "duration_s": r.boundary.duration_s,
                "dominant_phase": r.dominant_phase,
                "end_reason": r.outcome.end_reason,
                "shot_like": r.outcome.shot_like,
                "ended_in_box": r.outcome.ended_in_box,
                "reached_final_third": r.outcome.reached_final_third,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_parquet(args.out_dir / "episode_summary.parquet")

    print("\n--- episode counts by source ---")
    by_source = (
        summary.groupby("source")
        .agg(
            n_episodes=("episode_id", "count"),
            median_duration=("duration_s", "median"),
            pct_final_third=("reached_final_third", "mean"),
            n_shot_like=("shot_like", "sum"),
        )
        .round(3)
    )
    print(by_source.to_string())

    # 3. Build retrieval index across the whole corpus
    index = EpisodeIndex(k_default=3)
    index.fit(all_records)
    print(f"\nindex fit on {len(index)} episodes ({index.feature_dim} features)")

    # 4. Persist embeddings (for future querying without re-fitting)
    embed_rows = []
    for r in all_records:
        v = embed_episode(r)
        row = {
            "episode_id": r.boundary.episode_id,
            "source": record_source[r.boundary.episode_id],
            "match_id": record_match[r.boundary.episode_id],
        }
        for name, val in zip(EPISODE_FEATURE_NAMES, v, strict=True):
            row[name] = float(val)
        embed_rows.append(row)
    pd.DataFrame(embed_rows).to_parquet(args.out_dir / "embedding_matrix.parquet")

    # 5. Cross-source retrieval honesty test
    shot_like = [r for r in all_records if r.outcome.shot_like]
    print(f"\n--- cross-source retrieval test on {len(shot_like)} shot_like episodes ---")
    cross_results = []
    cross_source_hits = 0
    same_source_hits = 0
    for r in shot_like:
        neighbors = index.query(r, k=3, exclude_self=True)
        if not neighbors:
            continue
        query_source = record_source[r.boundary.episode_id]
        for n in neighbors:
            n_source = record_source[n.record.boundary.episode_id]
            if n_source != query_source:
                cross_source_hits += 1
            else:
                same_source_hits += 1
        cross_results.append(
            {
                "query_episode_id": r.boundary.episode_id,
                "query_source": query_source,
                "query_match": record_match[r.boundary.episode_id],
                "neighbors": [
                    {
                        "rank": n.rank,
                        "episode_id": n.record.boundary.episode_id,
                        "source": record_source[n.record.boundary.episode_id],
                        "match": record_match[n.record.boundary.episode_id],
                        "distance": round(n.distance, 4),
                        "shot_like": n.record.outcome.shot_like,
                        "dominant_phase": n.record.dominant_phase,
                    }
                    for n in neighbors
                ],
            }
        )
    total_neighbors = same_source_hits + cross_source_hits
    if total_neighbors:
        print(
            f"  cross-source neighbor rate: {100 * cross_source_hits / total_neighbors:.1f}% "
            f"({cross_source_hits}/{total_neighbors})"
        )
        # If ~0%, retrieval is overfit to source quirks.
        # If ~33% (3 sources, equal weight), retrieval is source-agnostic.
        # Real signal will be somewhere between depending on similarity actually existing.

    (args.out_dir / "cross_source_neighbors.json").write_text(
        json.dumps(cross_results, indent=2, default=str),
        encoding="utf-8",
    )

    # 6. Pattern clustering
    print("\n--- pattern library across full corpus ---")
    clusters = cluster_episodes(index, n_clusters=12)
    pattern_rows = []
    for c in clusters:
        # Source mix per cluster — does each pattern bridge sources?
        source_mix = Counter(record_source[eid] for eid in c.episode_ids)
        pattern_rows.append(
            {
                **asdict(c),
                "source_mix": dict(source_mix),
            }
        )
        sources_str = ", ".join(f"{s}:{n}" for s, n in source_mix.items())
        print(f"  cluster {c.cluster_id} (n={c.n_episodes}, {sources_str}): {c.label}")
    (args.out_dir / "patterns.json").write_text(
        json.dumps(pattern_rows, indent=2, default=str),
        encoding="utf-8",
    )

    # 7. Summary JSON
    (args.out_dir / "corpus_summary.json").write_text(
        json.dumps(
            {
                "n_matches": len(matches),
                "n_episodes": len(all_records),
                "n_shot_like": len(shot_like),
                "feature_dim": index.feature_dim,
                "n_clusters": len(clusters),
                "elapsed_s": round(elapsed, 1),
                "by_source": {
                    s: {
                        "n_matches": int(c),
                        "n_episodes": int((summary["source"] == s).sum()),
                        "n_shot_like": int(((summary["source"] == s) & summary["shot_like"]).sum()),
                    }
                    for s, c in Counter(_source_of(m) for m, _ in matches).items()
                },
                "cross_source_neighbor_rate": (
                    round(cross_source_hits / total_neighbors, 3) if total_neighbors else None
                ),
                "limitations": [
                    "Episode IDs are corpus-global. They do not align with per-match offline IDs;"
                    " the engine is rebuilt from scratch here.",
                    "SoccerNet clips are short (30 s each). Many produce a single episode that"
                    " spans the whole clip — median duration will be inflated for that source.",
                    "Cross-source retrieval is the first honest test of generalisation. With only"
                    " 13 continuous matches + 49 short clips, predictive value is still constrained;"
                    " more matches per source unblock proper calibration.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
