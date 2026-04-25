"""Phase 4 Slice B smoke: leave-one-out OBSO attribution + narrative for Metrica.

For every episode that has a clear peak (shot_like / ended_in_box / reached the final
third), compute per-attacker contributions and a templated narrative. Outputs:

- ``data/features/phase4b/attribution.parquet`` — long form: episode_id × player_id × Δ OBSO
- ``data/features/phase4b/narratives.json`` — list of plain-English episode descriptions
- ``data/features/phase4b/summary.json`` — counts + limitations

Slice B is the answer to "who/what made this attack happen." Slice C will retrieve
similar past episodes; together they form the attribution + retrieval engine the
roadmap was aimed at.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from football_analysis.analytics.episodes.contribution import compute_episode_attribution
from football_analysis.analytics.episodes.engine import build_episodes
from football_analysis.analytics.episodes.narrative import build_narrative
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
    p.add_argument("--out-dir", type=Path, default=Path("data/features/phase4b"))
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Cap how many episodes get attribution (for fast iteration). Default: all.",
    )
    p.add_argument("--obso-rows", type=int, default=40)
    p.add_argument("--obso-cols", type=int, default=60)
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

    # Filter to episodes that have a clear peak (saves compute on garbage episodes).
    peak_eligible = [
        r for r in records if r.outcome.shot_like or r.outcome.ended_in_box or r.outcome.reached_final_third
    ]
    print(f"  peak-eligible: {len(peak_eligible)} (shot_like + ended_in_box + reached_final_third)")

    if args.max_episodes is not None:
        peak_eligible = peak_eligible[: args.max_episodes]
        print(f"  capping to {len(peak_eligible)} episodes for fast smoke run")

    attribution_rows: list[dict] = []
    narratives: list[dict] = []
    t0 = time.time()
    for i, rec in enumerate(peak_eligible):
        attr = compute_episode_attribution(
            rec,
            tracking,
            "home",
            "away",
            rows=args.obso_rows,
            cols=args.obso_cols,
        )
        narr = build_narrative(rec, attr)
        narratives.append(
            {
                **{k: v for k, v in asdict(narr).items() if k != "top_contributors"},
                "top_contributors": [{"player_id": p, "contribution": round(c, 4)} for p, c in narr.top_contributors],
            }
        )
        if attr is not None:
            for player_id, contrib in attr.contributions.items():
                attribution_rows.append(
                    {
                        "episode_id": rec.boundary.episode_id,
                        "peak_frame": attr.peak_frame,
                        "peak_obso": round(attr.peak_obso, 4),
                        "possession_team": rec.boundary.possession_team,
                        "player_id": player_id,
                        "contribution": round(contrib, 4),
                        "baseline_obso": round(attr.baseline_obso_per_player[player_id], 4),
                        "shot_like": rec.outcome.shot_like,
                        "ended_in_box": rec.outcome.ended_in_box,
                        "reached_final_third": rec.outcome.reached_final_third,
                        "dominant_phase": rec.dominant_phase,
                    }
                )
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(
                f"    [{i + 1}/{len(peak_eligible)}] elapsed={elapsed:.1f}s (avg {elapsed / (i + 1):.2f}s per episode)"
            )
    elapsed = time.time() - t0
    print(f"  attribution complete in {elapsed:.1f}s ({elapsed / max(1, len(peak_eligible)):.2f}s/episode)")

    df_attr = pd.DataFrame(attribution_rows)
    df_attr.to_parquet(args.out_dir / "attribution.parquet")
    (args.out_dir / "narratives.json").write_text(json.dumps(narratives, indent=2, default=str), encoding="utf-8")

    if not df_attr.empty:
        print("\n--- top contributions across the match ---")
        top = df_attr.reindex(df_attr["contribution"].abs().sort_values(ascending=False).index).head(10)
        for _, row in top.iterrows():
            print(
                f"  ep={int(row['episode_id']):3d}  player={row['player_id']:8s}  "
                f"d_obso={row['contribution']:+.3f}  peak_obso={row['peak_obso']:.3f}  "
                f"phase={row['dominant_phase']}  shot_like={bool(row['shot_like'])}"
            )

        print("\n--- per-team total contribution magnitude ---")
        per_team = df_attr.groupby("possession_team")["contribution"].agg(
            n_attributions="count",
            sum_abs=lambda s: float(s.abs().sum()),
        )
        print(per_team.to_string())

    print("\n--- sample narratives ---")
    for narr in narratives[:5]:
        print(f"  ep {narr['episode_id']}: {narr['text'][:240]}")

    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "match": args.match,
                "n_episodes": len(records),
                "n_peak_eligible": len(peak_eligible),
                "n_attribution_rows": len(attribution_rows),
                "obso_grid": {"rows": args.obso_rows, "cols": args.obso_cols},
                "elapsed_s": round(elapsed, 1),
                "limitations": [
                    "Counterfactual = freeze player at episode-mean position; defenders don't react."
                    " This is interpretable contribution, NOT true causal inference.",
                    "Coarse OBSO grid (40x60 default) for speed; finer grids would shift Δ values"
                    " but not their relative ordering.",
                    "Trigger frame = ball first crosses x_oriented=70 (final-third entry). v1 heuristic;"
                    " future versions can use OBSO-rise or counterfactual-divergence triggers.",
                    "Attribution only at the peak frame, not the full trajectory."
                    " Slice C's retrieval index addresses the trajectory side via similarity.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
