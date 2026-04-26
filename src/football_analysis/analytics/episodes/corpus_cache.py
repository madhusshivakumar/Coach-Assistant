"""Persist (records, formation_pairs) to parquet so re-runs skip the rebuild.

Phase 6 demos rebuild ~15 min of episodes + formation labels every run. Caching
that to parquet on disk means subsequent recommendation queries take seconds
instead of half an hour. Cache invalidates on tracking-corpus changes by
hashing the input parquet mtimes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.episodes.formation_pair import FormationPair
from football_analysis.analytics.episodes.outcomes import EpisodeOutcome
from football_analysis.analytics.episodes.segmenter import EpisodeBoundary

_CACHE_VERSION = "1"


def _hash_inputs(tracking_paths: list[Path]) -> str:
    """Content-addressing the corpus by mtime + file size of every tracking parquet."""
    h = hashlib.sha256()
    for p in sorted(tracking_paths):
        if p.exists():
            stat = p.stat()
            h.update(str(p).encode())
            h.update(str(stat.st_mtime).encode())
            h.update(str(stat.st_size).encode())
    return h.hexdigest()[:16]


def write_corpus_cache(
    cache_dir: Path,
    records: list[EpisodeRecord],
    formation_pairs: list[FormationPair],
    record_to_match: dict[int, str],
    tracking_paths: list[Path],
) -> Path:
    """Persist the heavy outputs of a corpus rebuild.

    Stored:
      - ``records.parquet`` — flattened EpisodeBoundary + EpisodeOutcome + dominant_phase + match
      - ``state_trajectories.parquet`` — long-form snapshot rows keyed by episode_id
      - ``formation_pairs.parquet`` — (episode_id, attacker_formation, defender_formation, ...)
      - ``manifest.json`` — version + content hash for invalidation
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    rec_rows = []
    state_rows = []
    for r in records:
        rec_rows.append(
            {
                "episode_id": r.boundary.episode_id,
                "match_id": record_to_match.get(r.boundary.episode_id),
                "start_frame": r.boundary.start_frame,
                "end_frame": r.boundary.end_frame,
                "start_time_s": r.boundary.start_time_s,
                "end_time_s": r.boundary.end_time_s,
                "duration_s": r.boundary.duration_s,
                "possession_team": r.boundary.possession_team,
                "end_reason": r.boundary.end_reason,
                "dominant_phase": r.dominant_phase,
                "outcome_end_reason": r.outcome.end_reason,
                "reached_final_third": r.outcome.reached_final_third,
                "ended_in_box": r.outcome.ended_in_box,
                "shot_like": r.outcome.shot_like,
                "end_ball_x": r.outcome.end_ball_x,
                "end_ball_y": r.outcome.end_ball_y,
                "end_ball_speed": r.outcome.end_ball_speed,
            }
        )
        if not r.state_trajectory.empty:
            states = r.state_trajectory.copy()
            states["episode_id"] = r.boundary.episode_id
            state_rows.append(states)
    pd.DataFrame(rec_rows).to_parquet(cache_dir / "records.parquet")
    if state_rows:
        pd.concat(state_rows, ignore_index=True).to_parquet(cache_dir / "state_trajectories.parquet")
    pd.DataFrame([asdict(fp) for fp in formation_pairs]).to_parquet(cache_dir / "formation_pairs.parquet")
    manifest = {
        "version": _CACHE_VERSION,
        "input_hash": _hash_inputs(tracking_paths),
        "n_records": len(records),
        "n_formation_pairs": len(formation_pairs),
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return cache_dir


def cache_is_valid(cache_dir: Path, tracking_paths: list[Path]) -> bool:
    """True iff a cache exists at ``cache_dir`` and its input_hash matches."""
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        m = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False
    if m.get("version") != _CACHE_VERSION:
        return False
    return bool(m.get("input_hash") == _hash_inputs(tracking_paths))


def read_corpus_cache(
    cache_dir: Path,
) -> tuple[list[EpisodeRecord], list[FormationPair], dict[int, str]]:
    """Reconstruct (records, formation_pairs, record_to_match) from a cache."""
    rec_df = pd.read_parquet(cache_dir / "records.parquet")
    pair_df = pd.read_parquet(cache_dir / "formation_pairs.parquet")
    state_path = cache_dir / "state_trajectories.parquet"
    state_df = pd.read_parquet(state_path) if state_path.exists() else pd.DataFrame()

    state_by_eid: dict[int, pd.DataFrame] = {}
    if not state_df.empty:
        for eid, group in state_df.groupby("episode_id"):
            state_by_eid[int(eid)] = group.drop(columns=["episode_id"]).reset_index(drop=True)

    records: list[EpisodeRecord] = []
    record_to_match: dict[int, str] = {}
    for _, row in rec_df.iterrows():
        eid = int(row["episode_id"])
        boundary = EpisodeBoundary(
            episode_id=eid,
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
            start_time_s=float(row["start_time_s"]),
            end_time_s=float(row["end_time_s"]),
            duration_s=float(row["duration_s"]),
            possession_team=str(row["possession_team"]),
            end_reason=str(row["end_reason"]),
        )
        outcome = EpisodeOutcome(
            episode_id=eid,
            end_reason=str(row["outcome_end_reason"]),
            reached_final_third=bool(row["reached_final_third"]),
            ended_in_box=bool(row["ended_in_box"]),
            shot_like=bool(row["shot_like"]),
            end_ball_x=float(row["end_ball_x"]),
            end_ball_y=float(row["end_ball_y"]),
            end_ball_speed=float(row["end_ball_speed"]),
            duration_s=float(row["duration_s"]),
        )
        records.append(
            EpisodeRecord(
                boundary=boundary,
                outcome=outcome,
                state_trajectory=state_by_eid.get(eid, pd.DataFrame()),
                dominant_phase=(None if pd.isna(row["dominant_phase"]) else str(row["dominant_phase"])),
            )
        )
        record_to_match[eid] = str(row["match_id"])

    pairs: list[FormationPair] = []
    for _, row in pair_df.iterrows():
        pairs.append(
            FormationPair(
                episode_id=int(row["episode_id"]),
                representative_frame=int(row["representative_frame"]),
                attacker_team_id=str(row["attacker_team_id"]),
                defender_team_id=str(row["defender_team_id"]),
                attacker_formation=(None if pd.isna(row["attacker_formation"]) else str(row["attacker_formation"])),
                attacker_formation_cost=(
                    None if pd.isna(row["attacker_formation_cost"]) else float(row["attacker_formation_cost"])
                ),
                defender_formation=(None if pd.isna(row["defender_formation"]) else str(row["defender_formation"])),
                defender_formation_cost=(
                    None if pd.isna(row["defender_formation_cost"]) else float(row["defender_formation_cost"])
                ),
            )
        )
    return records, pairs, record_to_match
