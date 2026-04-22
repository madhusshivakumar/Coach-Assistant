"""Phase-1 smoke: run the full analytics pipeline on an ingested match and save all viz outputs.

Reads a processed events Parquet, applies the xT model, computes team metrics, and renders:
- shot map (team-coloured + scorer names + match-score header)
- pass networks (per team, directional arrows, player names)
- player heatmap for the top xT contributor
- xT surface (with colorbar + attack-direction hint)

Writes PNGs + a summary.json under data/features/phase1/<match_id>/.
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


def _statsbomb_match_context(
    match_id: str,
) -> tuple[dict[str, str], str | None, dict[str, str]]:
    """Return (team_names, home_team_id, player_names) from StatsBomb raw JSONs."""
    settings = get_settings()
    key = match_id.split(":", 1)[-1]
    key_int = int(key)

    team_names: dict[str, str] = {}
    home_team_id: str | None = None
    for manifest in (settings.raw_dir / "statsbomb" / "matches").rglob("*.json"):
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for m in payload:
            if m.get("match_id") == key_int:
                home_id = str(m["home_team"]["home_team_id"])
                away_id = str(m["away_team"]["away_team_id"])
                team_names = {
                    home_id: m["home_team"]["home_team_name"],
                    away_id: m["away_team"]["away_team_name"],
                }
                home_team_id = home_id
                break
        if team_names:
            break

    player_names: dict[str, str] = {}
    lineup_path = settings.raw_dir / "statsbomb" / "lineups" / f"{key}.json"
    if lineup_path.exists():
        try:
            lineup = json.loads(lineup_path.read_text(encoding="utf-8"))
            for team in lineup:
                for p in team.get("lineup", []):
                    pid = p.get("player_id")
                    if pid is not None:
                        player_names[str(pid)] = p.get("player_nickname") or p.get("player_name") or str(pid)
        except (OSError, json.JSONDecodeError):
            pass

    return team_names, home_team_id, player_names


def render_all(match_id: str, out_dir: Path) -> dict:
    parquet = load_match_parquet(match_id)
    events = pd.read_parquet(parquet)
    out_dir.mkdir(parents=True, exist_ok=True)

    analytics = run_pipeline(events)
    enriched = analytics.events

    team_names, home_team_id, player_names = _statsbomb_match_context(match_id)
    team_ids = sorted(enriched["team_id"].dropna().unique())

    summary: dict = {
        "match_id": match_id,
        "events": len(events),
        "home_team_id": home_team_id,
        "teams": {tid: team_names.get(tid, tid) for tid in team_ids},
        "ppda": analytics.ppda,
        "field_tilt": analytics.field_tilt,
        "top_xt_players": [],
    }

    fig = plot_shot_map(
        enriched,
        home_team_id=home_team_id,
        team_names=team_names,
        player_names=player_names,
    )
    fig.savefig(out_dir / "shot_map.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    for tid in team_ids:
        fig = plot_pass_network(
            enriched,
            team_id=tid,
            min_passes_edge=5,
            home_team_id=home_team_id,
            team_names=team_names,
            player_names=player_names,
        )
        fig.savefig(out_dir / f"pass_network_{tid}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    fig = plot_xt_surface(analytics.xt_grid)
    fig.savefig(out_dir / "xt_surface.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    pos_moves = enriched[enriched["xt_delta"].notna() & (enriched["xt_delta"] > 0)]
    if not pos_moves.empty:
        ranked = (
            pos_moves.groupby("player_id")["xt_delta"]
            .agg(["sum", "count"])
            .sort_values("sum", ascending=False)
            .head(10)
        )
        summary["top_xt_players"] = [
            {
                "player_id": str(p),
                "player_name": player_names.get(str(p), str(p)),
                "xt_sum": float(row["sum"]),
                "actions": int(row["count"]),
            }
            for p, row in ranked.iterrows()
        ]
        top_player = str(ranked.index[0])
        fig = plot_player_heatmap(
            enriched,
            player_id=top_player,
            player_names=player_names,
            team_names=team_names,
        )
        fig.savefig(
            out_dir / f"heatmap_top_player_{top_player}.png",
            dpi=140,
            bbox_inches="tight",
        )
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
    print(
        json.dumps(
            {k: summary[k] for k in ("match_id", "events", "teams", "ppda", "field_tilt")},
            indent=2,
            default=str,
        )
    )
    top = summary.get("top_xt_players") or []
    if top:
        print("\nTop xT contributors:")
        for row in top:
            print(f"  {row['player_name']:30s}  xt={row['xt_sum']:.3f}  ({row['actions']} actions)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
