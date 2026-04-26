# ruff: noqa: PLR0912, PLR0915
"""Phase 6 prescriptive demo: given a defender formation, recommend the patterns
that broke it; given an attacker formation, recommend the defensive setups that
contained it.

Outputs to ``data/features/phase6_recommend/``:

- ``RECOMMENDATIONS.md`` — readable answers to a handful of representative
  questions: "how do I break a 4-4-2?", "how do I defend against a 4-3-3?", etc.
- ``recommendations.json`` — machine-readable bundle of the same answers.
- ``formation_distribution.png`` — a quick histogram showing which formations
  actually appear in our corpus (so the user can see what the engine has data
  on).
- ``recommendation_heatmap_<question>.png`` — for each top-1 recommendation,
  a pitch heatmap of where the matching episodes' attacks happened.

Built on Phase 4 (episode engine), Phase 5 (multi-source corpus,
LiveEpisodeEngine), and Phase 6-A (formation_pair labeling). The heavy work is
labeling formations across the corpus — done once here, cached to a parquet so
re-runs are fast.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from football_analysis.analytics.episodes.corpus_cache import (
    cache_is_valid,
    read_corpus_cache,
    write_corpus_cache,
)
from football_analysis.analytics.episodes.engine import build_episodes
from football_analysis.analytics.episodes.formation_pair import (
    FormationPair,
    label_episode_formation_pair,
)
from football_analysis.analytics.episodes.index import EpisodeIndex
from football_analysis.analytics.episodes.recommend import (
    recommend_defensive_setup_against_attacker,
    recommend_for_defender_formation,
)
from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M
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


def draw_pitch(ax: plt.Axes) -> None:
    ax.set_xlim(-2, PITCH_LENGTH_M + 2)
    ax.set_ylim(-2, PITCH_WIDTH_M + 2)
    ax.set_aspect("equal")
    ax.add_patch(mpatches.Rectangle((0, 0), PITCH_LENGTH_M, PITCH_WIDTH_M, fill=False, color="white", linewidth=1.5))
    ax.plot([PITCH_LENGTH_M / 2, PITCH_LENGTH_M / 2], [0, PITCH_WIDTH_M], color="white", linewidth=1.0)
    ax.add_patch(
        mpatches.Circle((PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2), 9.15, fill=False, color="white", linewidth=1.0)
    )
    for side_x in (0, PITCH_LENGTH_M - 16.5):
        ax.add_patch(
            mpatches.Rectangle(
                (side_x, 13.84),
                16.5,
                40.32,
                fill=False,
                color="white",
                linewidth=1.0,
            )
        )
    ax.set_facecolor("#0e6b3a")
    ax.set_xticks([])
    ax.set_yticks([])


def render_formation_distribution(pairs: list[FormationPair], out_path: Path) -> None:
    counts_def = Counter(p.defender_formation for p in pairs if p.defender_formation)
    counts_atk = Counter(p.attacker_formation for p in pairs if p.attacker_formation)
    all_forms = sorted(set(counts_def) | set(counts_atk))
    if not all_forms:
        return
    atk = [counts_atk.get(f, 0) for f in all_forms]
    def_ = [counts_def.get(f, 0) for f in all_forms]
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
    x = np.arange(len(all_forms))
    ax.bar(x - 0.2, atk, 0.4, label="as attacker", color="#d62728")
    ax.bar(x + 0.2, def_, 0.4, label="as defender", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(all_forms, rotation=30, ha="right")
    ax.set_ylabel("episodes")
    ax.set_title("Formation distribution across the corpus")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_recommendation_heatmap(
    records,
    formation_pairs: list[FormationPair],
    rec_episode_ids: list[int],
    title: str,
    out_path: Path,
) -> None:
    """Heatmap of where the recommended-pattern episodes ended up.

    All attacks are oriented to point toward the right goal (+x) so the heatmap
    shows ATTACK direction independent of which team had possession. We use a
    2-D histogram density rather than scatter — far more readable for ~hundreds
    of episodes.
    """
    eid_to_record = {r.boundary.episode_id: r for r in records}
    end_xs: list[float] = []
    end_ys: list[float] = []
    for eid in rec_episode_ids:
        rec = eid_to_record.get(eid)
        if rec is None:
            continue
        # Orient: home attacks +x, away attacks -x in canonical coords. Flip the
        # away team's positions so EVERY attack ends up in the right-side goal.
        ex = rec.outcome.end_ball_x
        ey = rec.outcome.end_ball_y
        if rec.boundary.possession_team != "home":
            ex = PITCH_LENGTH_M - ex
            ey = PITCH_WIDTH_M - ey
        if not (0 <= ex <= PITCH_LENGTH_M and 0 <= ey <= PITCH_WIDTH_M):
            continue
        end_xs.append(ex)
        end_ys.append(ey)
    if not end_xs:
        return

    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=140)
    draw_pitch(ax)
    # 2-D histogram with a hot colormap; transparent low-density cells so the
    # pitch shows through. ~1.5 m bins (70 x 45 grid).
    hist, xedges, yedges = np.histogram2d(
        end_xs,
        end_ys,
        bins=[70, 45],
        range=[[0, PITCH_LENGTH_M], [0, PITCH_WIDTH_M]],
    )
    # Mask zero bins so they're transparent over the pitch.
    masked = np.ma.masked_equal(hist.T, 0)
    cmap = plt.cm.YlOrRd
    cmap.set_bad(alpha=0)
    im = ax.imshow(
        masked,
        origin="lower",
        extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
        cmap=cmap,
        alpha=0.78,
        zorder=4,
        interpolation="bilinear",
    )
    ax.set_title(title + f"  (n={len(end_xs)} attacks oriented +x)", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("attack-end density", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="#0a4d28")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/features/phase6_recommend"))
    p.add_argument("--max-matches", type=int, default=None)
    p.add_argument("--cache-dir", type=Path, default=Path("data/features/phase6_corpus_cache"))
    p.add_argument("--no-cache", action="store_true", help="Force a full rebuild")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    matches = load_all_tracking_groups()
    if args.max_matches:
        matches = matches[: args.max_matches]
    print(f"corpus: {len(matches)} matches")

    # Determine the input parquet paths for cache invalidation.
    settings = get_settings()
    tracking_root = settings.processed_dir / "tracking"
    input_paths: list[Path] = []
    for match_label, _ in matches:
        for p_path in tracking_root.rglob(f"match_id={match_label}/period=*.parquet"):
            input_paths.append(p_path)

    # 1. Try the cache first.
    if (
        not args.no_cache
        and not args.max_matches  # cap-by-N invalidates the hash space anyway
        and cache_is_valid(args.cache_dir, input_paths)
    ):
        print(f"cache hit: {args.cache_dir}")
        all_records, formation_pairs, record_match = read_corpus_cache(args.cache_dir)
        print(f"  loaded {len(all_records)} records, {len(formation_pairs)} pairs")
    else:
        # 1. Build episodes from scratch
        all_records = []
        record_match = {}
        eid_counter = 0
        t0 = time.time()
        for match_label, df in matches:
            try:
                recs = build_episodes(df, home_team_id="home", away_team_id="away")
            except Exception as e:
                print(f"  skip {match_label}: {e}")
                continue
            for r in recs:
                new = type(r)(
                    boundary=type(r.boundary)(
                        episode_id=eid_counter,
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
                all_records.append(new)
                record_match[eid_counter] = match_label
                eid_counter += 1
        print(f"built {len(all_records)} episodes in {time.time() - t0:.1f}s")
        formation_pairs = []  # populated next; cache write below

    # 2. Label formation pairs (skipped on cache hit — already loaded above).
    if not formation_pairs:
        print("labeling formation pairs...")
        tracking_by_match = dict(matches)
        t0 = time.time()
        for r in all_records:
            match_label = record_match[r.boundary.episode_id]
            try:
                fp = label_episode_formation_pair(
                    r,
                    tracking_by_match[match_label],
                    home_team_id="home",
                    away_team_id="away",
                    attacking_directions={"home": "right", "away": "left"},
                )
            except Exception:
                continue
            formation_pairs.append(fp)
        print(f"  labeled {len(formation_pairs)} pairs in {time.time() - t0:.1f}s")

        # Persist for future runs (cold rebuild path).
        if not args.max_matches and input_paths:
            print(f"writing cache to {args.cache_dir}...")
            write_corpus_cache(
                args.cache_dir,
                all_records,
                formation_pairs,
                record_match,
                input_paths,
            )

    # Always emit the formation_pairs parquet alongside the recommendation outputs.
    pd.DataFrame([asdict(fp) for fp in formation_pairs]).to_parquet(
        args.out_dir / "formation_pairs.parquet",
    )

    # 3. Build retrieval index (used by recommend_for_defender_formation)
    index = EpisodeIndex(k_default=3)
    index.fit(all_records)

    # 4. Quick distribution of formations
    render_formation_distribution(formation_pairs, args.out_dir / "formation_distribution.png")
    n_with_def = sum(1 for fp in formation_pairs if fp.defender_formation)
    print(f"  formations labeled (defender): {n_with_def}/{len(formation_pairs)}")
    top_def = Counter(fp.defender_formation for fp in formation_pairs if fp.defender_formation).most_common(5)
    print(f"  top defender formations: {top_def}")
    top_atk = Counter(fp.attacker_formation for fp in formation_pairs if fp.attacker_formation).most_common(5)
    print(f"  top attacker formations: {top_atk}")

    # 5. Run recommendations for the most common defender + attacker formations.
    # We pick the top-3 defender formations actually present in the corpus.
    def_targets = [f for f, _ in top_def[:3]]
    atk_targets = [f for f, _ in top_atk[:3]]

    bundle = {
        "n_matches": len(matches),
        "n_episodes": len(all_records),
        "n_formation_pairs": len(formation_pairs),
        "top_defender_formations": top_def,
        "top_attacker_formations": top_atk,
        "questions": [],
    }

    md_lines: list[str] = [
        "# Phase 6 Prescriptive Demo — formation analysis",
        "",
        f"**Corpus:** {len(matches)} matches → {len(all_records)} episodes →"
        f" {len(formation_pairs)} formation-pair labels.",
        "",
        "![Formation distribution](formation_distribution.png)",
        "",
    ]

    for def_form in def_targets:
        recs = recommend_for_defender_formation(
            defender_formation=def_form,
            records=all_records,
            formation_pairs=formation_pairs,
            index=index,
            top_k_patterns=5,
            min_episodes_per_pattern=2,
            n_clusters=12,
        )
        q_label = f"break-{def_form.replace(' ', '_').replace('/', '_')}"
        bundle["questions"].append(
            {
                "question": f"How do I break a {def_form}?",
                "type": "offensive_breakdown",
                "n_recommendations": len(recs),
                "recommendations": [asdict(r) for r in recs],
            }
        )
        md_lines.append(f"## How do I break a `{def_form}`?")
        if not recs:
            md_lines.append("> *(no patterns matched in the current corpus)*")
            md_lines.append("")
            continue
        md_lines.append("| rank | dominant phase | n_eps | avg value | shot-like % | in-box % | example episode IDs |")
        md_lines.append("|---|---|---:|---:|---:|---:|---|")
        for r in recs:
            phase = r.dominant_phase or "—"
            ex_ids = ", ".join(str(e) for e in r.example_episode_ids[:3])
            md_lines.append(
                f"| {r.rank + 1} | {phase} | {r.n_supporting_episodes} | "
                f"{r.avg_outcome_value:.2f} | {100 * r.pct_shot_like:.0f}% | "
                f"{100 * r.pct_ended_in_box:.0f}% | {ex_ids} |"
            )
        md_lines.append("")
        # Heatmap of where attacks against this defender formation ended.
        if recs:
            top = recs[0]
            heat_path = args.out_dir / f"recommendation_heatmap_{q_label}.png"
            matching_eids = [fp.episode_id for fp in formation_pairs if fp.defender_formation == def_form]
            render_recommendation_heatmap(
                all_records,
                formation_pairs,
                matching_eids,
                title=f"Where attacks against {def_form} ended up "
                f"(n={len(matching_eids)} episodes; top pattern: {top.pattern_label})",
                out_path=heat_path,
            )
            md_lines.append(f"![heatmap]({heat_path.name})")
            md_lines.append("")

    for atk_form in atk_targets:
        setups = recommend_defensive_setup_against_attacker(
            attacker_formation=atk_form,
            records=all_records,
            formation_pairs=formation_pairs,
            top_k_setups=3,
        )
        bundle["questions"].append(
            {
                "question": f"How do I defend against a {atk_form}?",
                "type": "defensive_prescription",
                "n_setups": len(setups),
                "setups": [{"defender_formation": f, **stats} for f, stats in setups],
            }
        )
        md_lines.append(f"## How do I defend against a `{atk_form}`?")
        if not setups:
            md_lines.append("> *(no defensive setups had enough samples)*")
            md_lines.append("")
            continue
        md_lines.append("| rank | defender formation | n_eps | avg outcome conceded | shot conceded % |")
        md_lines.append("|---|---|---:|---:|---:|")
        for i, (form, stats) in enumerate(setups):
            md_lines.append(
                f"| {i + 1} | {form} | {int(stats['n_episodes'])} | "
                f"{stats['avg_outcome_value']:.2f} | "
                f"{100 * stats['pct_shot_conceded']:.0f}% |"
            )
        md_lines.append("")

    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Caveats")
    md_lines.append("- Cross-source neighbor rate in the retrieval index is currently ~6%, meaning")
    md_lines.append("  patterns mostly cluster within their own data source. Recommendations are")
    md_lines.append("  more reliable when query and library overlap by source.")
    md_lines.append("- Outcome value is a v1 boolean composite (shot_like > ended_in_box >")
    md_lines.append("  reached_final_third). Phase 6-B replaces this with calibrated P(shot|state).")
    md_lines.append("- `Bialkowski` formation fitting picks the best of a fixed template set;")
    md_lines.append("  we may be over-confidently classifying ambiguous shapes. The")
    md_lines.append("  `attacker_formation_cost` field exposes goodness-of-fit for filtering.")
    md_lines.append("- SoccerNet clips are 30 s; their formation labels reflect that snapshot,")
    md_lines.append("  not necessarily a team's *base* formation across a full match.")

    (args.out_dir / "RECOMMENDATIONS.md").write_text("\n".join(md_lines), encoding="utf-8")
    (args.out_dir / "recommendations.json").write_text(
        json.dumps(bundle, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
