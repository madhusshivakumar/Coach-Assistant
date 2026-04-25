"""Phase-3 Slice B smoke: extract formation profiles + compute baseline + render radars.

Two pipelines (because no single ingested match has both layers yet):

- **Tracking** (Metrica): classify phases → per-(team, phase) compactness profile →
  z-score vs in-sample population → one radar per phase showing home vs away.
- **Events** (Argentina vs France WC2022): apply Phase-1 pipeline → per-team xT/PPDA/
  field-tilt/shots profile → one radar showing home vs away.

Outputs to data/features/phase3b/:
  tracking_radar_<phase>.png   one per phase that has enough frames
  events_radar.png             single chart, home vs away
  summary.json                 baseline, profiles, strengths/weaknesses per slice
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from football_analysis.analytics.formations.profile import (
    FormationProfile,
    extract_events_profile,
    extract_tracking_profile,
)
from football_analysis.analytics.formations.strengths import (
    classify_strengths_weaknesses,
    compute_baseline,
    score_profile,
)
from football_analysis.analytics.phases.classifier import classify_frames
from football_analysis.analytics.pipeline.runner import run as run_events_pipeline
from football_analysis.config import get_settings
from football_analysis.viz.static.formation_radar import plot_formation_radar


def load_tracking(match_id: str) -> pd.DataFrame:
    settings = get_settings()
    key = match_id.split(":", 1)[-1]
    parquets: list[Path] = []
    for d in (settings.processed_dir / "tracking").rglob(f"match_id=metrica-{key}"):
        parquets.extend(d.rglob("*.parquet"))
    if not parquets:
        raise SystemExit(f"No tracking parquet for {match_id!r}")
    return pd.concat([pd.read_parquet(p) for p in sorted(parquets)], ignore_index=True)


def load_events(match_id: str) -> pd.DataFrame:
    settings = get_settings()
    key = match_id.split(":", 1)[-1]
    for p in (settings.processed_dir / "events").rglob("*.parquet"):
        if key in p.stem:
            return pd.read_parquet(p)
    raise SystemExit(f"No events parquet for {match_id!r}")


def _profile_to_summary(profile: FormationProfile) -> dict:
    return {
        "team_id": profile.team_id,
        "phase": profile.phase,
        "sample": profile.sample,
        "n_frames": profile.n_frames,
        "metrics": {k: round(v, 3) for k, v in profile.metrics.items()},
    }


def render_tracking_radars(match_id: str, out_dir: Path) -> dict:
    print(f"--- TRACKING profiles for {match_id} ---")
    tracking = load_tracking(match_id)
    classified = classify_frames(tracking, home_team_id="home", away_team_id="away")

    home_profiles = extract_tracking_profile(tracking, classified, "home", attacking_right=True)
    away_profiles = extract_tracking_profile(tracking, classified, "away", attacking_right=False)
    all_profiles = list(home_profiles.values()) + list(away_profiles.values())
    if not all_profiles:
        print("  no tracking profiles produced (insufficient frames per phase)")
        return {"profiles": [], "phases_rendered": []}

    baseline = compute_baseline(all_profiles)
    print(f"  baseline over {baseline.n_samples} (team×phase) profiles")

    rendered_phases: list[str] = []
    profile_summaries: list[dict] = []
    for phase in sorted(set(home_profiles) | set(away_profiles)):
        per_phase = [p for p in (home_profiles.get(phase), away_profiles.get(phase)) if p is not None]
        if not per_phase:
            continue
        fig = plot_formation_radar(
            per_phase,
            baseline,
            title=f"Tracking shape — {phase}  ({match_id})",
        )
        png = out_dir / f"tracking_radar_{phase}.png"
        fig.savefig(png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        rendered_phases.append(phase)
        for p in per_phase:
            z = score_profile(p, baseline)
            cls = classify_strengths_weaknesses(z)
            profile_summaries.append(
                {
                    **_profile_to_summary(p),
                    "z_scores": {k: round(v, 2) for k, v in z.items()},
                    "strengths": cls.strengths,
                    "weaknesses": cls.weaknesses,
                }
            )
            print(f"  {p.team_id:5s} {phase:14s}  strengths={cls.strengths}  weaknesses={cls.weaknesses}")

    return {
        "baseline": {
            "means": {k: round(v, 3) for k, v in baseline.means.items()},
            "stds": {k: round(v, 3) for k, v in baseline.stds.items()},
            "n_samples": baseline.n_samples,
        },
        "profiles": profile_summaries,
        "phases_rendered": rendered_phases,
    }


def render_events_radar(match_id: str, out_dir: Path) -> dict:
    print(f"\n--- EVENTS profiles for {match_id} ---")
    events = load_events(match_id)
    analytics = run_events_pipeline(events)
    enriched = analytics.events
    team_ids = sorted(t for t in enriched["team_id"].dropna().unique() if str(t).isdigit() or len(str(t)) > 0)
    if len(team_ids) < 2:
        print(f"  not enough teams in events ({team_ids})")
        return {"profiles": []}
    home_id, away_id = team_ids[0], team_ids[1]
    profiles = [
        extract_events_profile(enriched, home_id, away_id),
        extract_events_profile(enriched, away_id, home_id),
    ]
    baseline = compute_baseline(profiles)

    fig = plot_formation_radar(profiles, baseline, title=f"Events profile — {match_id}")
    png = out_dir / "events_radar.png"
    fig.savefig(png, dpi=140, bbox_inches="tight")
    plt.close(fig)

    summaries = []
    for p in profiles:
        z = score_profile(p, baseline)
        cls = classify_strengths_weaknesses(z)
        summaries.append(
            {
                **_profile_to_summary(p),
                "z_scores": {k: round(v, 2) for k, v in z.items()},
                "strengths": cls.strengths,
                "weaknesses": cls.weaknesses,
            }
        )
        print(f"  team {p.team_id} strengths={cls.strengths}  weaknesses={cls.weaknesses}")
        for k, v in p.metrics.items():
            print(f"      {k:18s} = {v:.3f}  z={z.get(k, 0.0):+.2f}")

    return {
        "baseline": {
            "means": {k: round(v, 3) for k, v in baseline.means.items()},
            "stds": {k: round(v, 3) for k, v in baseline.stds.items()},
            "n_samples": baseline.n_samples,
        },
        "profiles": summaries,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-match", default="metrica:1")
    p.add_argument("--events-match", default="statsbomb:3869685")
    p.add_argument("--out-dir", type=Path, default=Path("data/features/phase3b"))
    p.add_argument("--skip-tracking", action="store_true")
    p.add_argument("--skip-events", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "tracking_match": args.tracking_match,
        "events_match": args.events_match,
        "tracking": {},
        "events": {},
        "limitations": [
            "baseline = in-sample population (this match only) — not a real corpus",
            "tracking and events profiles are separate; no unified radar until a match"
            " with both is ingested (PFF FC pending)",
            "events-side has no phase axis (no events-only phase classifier yet)",
        ],
    }

    if not args.skip_tracking:
        try:
            summary["tracking"] = render_tracking_radars(args.tracking_match, args.out_dir)
        except SystemExit as e:
            print(f"  skipped tracking: {e}")

    if not args.skip_events:
        try:
            summary["events"] = render_events_radar(args.events_match, args.out_dir)
        except SystemExit as e:
            print(f"  skipped events: {e}")

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
