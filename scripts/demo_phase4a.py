"""Phase 4 Slice A smoke: build the possession-episode library for a tracking match.

Outputs to ``data/features/phase4a/``:

- ``episode_summary.parquet`` — one row per episode, ready to feed retrieval.
- ``state_trajectory.parquet`` — concatenated per-snapshot rows (long form).
- ``summary.json`` — counts, distributions, and a written list of limitations.

This is the data backbone for Slice B (attribution) and Slice C (retrieval). Heavy
compute (pitch control, OBSO surfaces) is intentionally NOT here — Slice B owns it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from football_analysis.analytics.episodes.engine import (
    build_episodes,
    episodes_to_summary,
)
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
    p.add_argument("--out-dir", type=Path, default=Path("data/features/phase4a"))
    p.add_argument("--snapshot-hz", type=float, default=2.0)
    p.add_argument("--min-dead-frames", type=int, default=5)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracking = load_tracking(args.match)
    print(f"--- {args.match} ---")
    print(f"  tracking: {len(tracking):,} rows, {tracking['frame_id'].nunique():,} unique frames")

    records = build_episodes(
        tracking,
        home_team_id="home",
        away_team_id="away",
        attacking_directions={"home": "right", "away": "left"},
        snapshot_hz=args.snapshot_hz,
        min_dead_frames=args.min_dead_frames,
    )
    print(f"  built {len(records)} episode records")

    summary = episodes_to_summary(records)
    summary.to_parquet(args.out_dir / "episode_summary.parquet")
    print(f"  -> {args.out_dir / 'episode_summary.parquet'}")

    n_state_rows = 0
    if records:
        all_states = pd.concat(
            [r.state_trajectory for r in records if not r.state_trajectory.empty],
            ignore_index=True,
        )
        all_states.to_parquet(args.out_dir / "state_trajectory.parquet")
        n_state_rows = len(all_states)
        print(f"  -> {args.out_dir / 'state_trajectory.parquet'} ({n_state_rows:,} rows)")

    if not summary.empty:
        print("\n--- distributions ---")
        print(f"  median duration:      {summary['duration_s'].median():.2f} s")
        print(
            f"  reached final third:  {int(summary['reached_final_third'].sum())} "
            f"({100 * summary['reached_final_third'].mean():.1f}%)"
        )
        print(f"  ended in box:         {int(summary['ended_in_box'].sum())}")
        print(f"  shot-like:            {int(summary['shot_like'].sum())}")
        print("  end_reason mix:")
        for reason, count in summary["end_reason"].value_counts().items():
            print(f"    {reason:20s} {count}")
        print("  dominant phase mix:")
        for phase, count in summary["dominant_phase"].value_counts(dropna=False).items():
            label = "(none)" if pd.isna(phase) else phase
            print(f"    {label:20s} {count}")
        print("  by team:")
        for team, count in summary["possession_team"].value_counts().items():
            print(f"    {team:8s} {count}")

    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "match": args.match,
                "n_episodes": len(records),
                "n_state_rows": n_state_rows,
                "snapshot_hz": args.snapshot_hz,
                "min_dead_frames": args.min_dead_frames,
                "limitations": [
                    "Slice A: lightweight per-snapshot state only (positions + ball + simple shape)."
                    " Pitch-control and OBSO surfaces deferred to Slice B (attribution).",
                    "shot_like is a heuristic (ball in box at high speed), not a true shot detector;"
                    " it will produce false positives on heavy clearances.",
                    "Single-tracking-match dataset (Metrica 8-min sample);"
                    " the engine is sound but episode patterns won't generalize until more tracking is ingested.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
