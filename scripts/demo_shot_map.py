"""Phase-0 smoke: render a shot map from a processed events Parquet.

Usage:
    uv run python scripts/demo_shot_map.py --match-id statsbomb:<id>

If no match is specified, uses the first events parquet found under data/processed/events/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from mplsoccer import Pitch

from football_analysis.config import get_settings


def find_events_parquet(match_id: str | None) -> Path:
    root = get_settings().processed_dir / "events"
    paths = sorted(root.rglob("*.parquet"))
    if not paths:
        raise SystemExit(f"No events parquet under {root}. Run `fa-data ingest statsbomb --match-id <id>` first.")
    if match_id is None:
        return paths[0]
    match_key = match_id.split(":", 1)[-1]
    for p in paths:
        if match_key in p.stem:
            return p
    raise SystemExit(f"No parquet matches match_id {match_id!r}.")


def render(path: Path, out: Path) -> None:
    df = pd.read_parquet(path)
    shots = df[df["action_type"] == "shot"].copy()
    print(f"Loaded {len(df)} events from {path.name}; {len(shots)} shots")

    pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68, line_color="#444")
    fig, ax = pitch.draw(figsize=(10, 7))

    if shots.empty:
        ax.set_title("No shots in this match")
    else:
        goals = shots[shots["result"] == "success"]
        others = shots[shots["result"] != "success"]
        pitch.scatter(
            others["start_x"],
            others["start_y"],
            ax=ax,
            s=80,
            color="#cc3344",
            alpha=0.7,
            edgecolors="black",
            label="shot",
        )
        pitch.scatter(
            goals["start_x"],
            goals["start_y"],
            ax=ax,
            s=140,
            color="#ffd54a",
            edgecolors="black",
            linewidth=1.5,
            label="goal",
            marker="*",
        )
        ax.legend(loc="upper left")
        ax.set_title(f"Shots — {path.stem}  ({len(shots)} total, {len(goals)} goals)")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--match-id", default=None, help="Canonical match_id, e.g. statsbomb:3869685")
    p.add_argument("--out", default="data/features/demo_shot_map.png", type=Path)
    args = p.parse_args()

    path = find_events_parquet(args.match_id)
    render(path, args.out)


if __name__ == "__main__":
    main()
