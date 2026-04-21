"""Phase-1 smoke: run the full analytics pipeline on an ingested match and save all viz outputs.

Reads a processed events Parquet, applies the xT model, computes team metrics, and renders:
- shot map
- pass networks (per team)
- player heatmap (top xT contributor)
- xT surface

Writes PNGs under data/features/phase1/<match_id>/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd

from football_analysis.analytics.pipeline.runner import run as run_pipeline
from football_analysis.config import get_settings
from football_analysis.viz.static.heatmap import plot_player_heatmap
from football_analysis.viz.static.pass_network import plot_pass_network
from football_analysis.viz.static.shot_map import plot_shot_map
from football_analysis.viz.static.xt_surface import plot_xt_surface


def load_match_parquet(match_id: str) -> Path:
    root = get_settings().processed_dir / "events"
    key = match_id.split(":", 1)[-1]
    for p in root.rglob("*.parquet"):
        if key in p.stem:
            return p
    raise SystemExit(f"No parquet matches {match_id!r}")


def team_name_map_for_statsbomb(match_id: str) -> dict[str, str]:
    """Look up team names from the StatsBomb matches manifest."""
    settings = get_settings()
    manifest = settings.raw_dir / "statsbomb" / "matches" / "43" / "106.json"
    if not manifest.exists():
        return {}
    key = int(match_id.split(":", 1)[-1])
    for m in json.loads(manifest.read_text(encoding="utf-8")):
        if m["match_id"] == key:
            return {
                str(m["home_team"]["home_team_id"]): m["home_team"]["home_team_name"],
                str(m["away_team"]["away_team_id"]): m["away_team"]["away_team_name"],
            }
    return {}


def render_all(match_id: str, out_dir: Path) -> dict:
    parquet = load_match_parquet(match_id)
    events = pd.read_parquet(parquet)
    out_dir.mkdir(parents=True, exist_ok=True)

    analytics = run_pipeline(events)
    enriched = analytics.events

    names = team_name_map_for_statsbomb(match_id)
    team_ids = sorted(enriched["team_id"].dropna().unique())

    summary: dict = {
        "match_id": match_id,
        "events": len(events),
        "teams": {tid: names.get(tid, tid) for tid in team_ids},
        "ppda": analytics.ppda,
        "field_tilt": analytics.field_tilt,
        "top_xt_players": [],
    }

    # Shot map (all shots)
    fig = plot_shot_map(enriched, title=f"Shots — {match_id}")
    fig.savefig(out_dir / "shot_map.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Pass network per team
    for tid in team_ids:
        label = names.get(tid, tid)
        fig = plot_pass_network(enriched, team_id=tid, min_passes_edge=5, title=f"Pass network — {label}")
        fig.savefig(out_dir / f"pass_network_{tid}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    # xT surface
    fig = plot_xt_surface(analytics.xt_grid, title=f"xT surface — {match_id}")
    fig.savefig(out_dir / "xt_surface.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Top xT contributors (positive xt_delta only)
    pos_moves = enriched[enriched["xt_delta"].notna() & (enriched["xt_delta"] > 0)]
    if not pos_moves.empty:
        ranked = (
            pos_moves.groupby("player_id")["xt_delta"]
            .agg(["sum", "count"])
            .sort_values("sum", ascending=False)
            .head(10)
        )
        summary["top_xt_players"] = [
            {"player_id": str(p), "xt_sum": float(row["sum"]), "actions": int(row["count"])}
            for p, row in ranked.iterrows()
        ]
        # Heatmap for top player
        top_player = str(ranked.index[0])
        fig = plot_player_heatmap(enriched, player_id=top_player, title=f"Heatmap — player {top_player}")
        fig.savefig(out_dir / f"heatmap_top_player_{top_player}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--match-id", required=True, help="e.g. statsbomb:3869685")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    out = args.out_dir or (Path("data/features/phase1") / args.match_id.replace(":", "_"))
    summary = render_all(args.match_id, out)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
