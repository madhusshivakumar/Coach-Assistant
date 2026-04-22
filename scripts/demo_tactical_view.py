"""Phase-2 Slice C smoke: save an animated Plotly tactical view to standalone HTML.

Defaults to 50 frames (= 2 s of play at 25 Hz) starting at frame 600 so the animation
is quick to load and visibly non-trivial.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from football_analysis.config import get_settings
from football_analysis.viz.interactive.tactical_view import animate


def load_tracking(match_id: str) -> pd.DataFrame:
    settings = get_settings()
    key = match_id.split(":", 1)[-1]
    matching_dirs = list((settings.processed_dir / "tracking").rglob(f"match_id=metrica-{key}"))
    parquets: list[Path] = []
    for d in matching_dirs:
        parquets.extend(d.rglob("*.parquet"))
    if not parquets:
        raise SystemExit(f"No tracking parquet for {match_id!r}")
    return pd.concat([pd.read_parquet(p) for p in sorted(parquets)], ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--match-id", default="metrica:1")
    p.add_argument("--start", type=int, default=600)
    p.add_argument("--end", type=int, default=650)
    p.add_argument("--out", default="data/features/phase2/tactical_view.html", type=Path)
    args = p.parse_args()

    tracking = load_tracking(args.match_id)
    fig = animate(
        tracking,
        home_team_id="home",
        away_team_id="away",
        frame_range=(args.start, args.end),
        title=f"Tactical view — {args.match_id} frames {args.start}-{args.end}",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.out), include_plotlyjs="cdn", full_html=True, auto_play=False)
    print(f"wrote {args.out} ({args.end - args.start + 1} animation frames)")
    print(f"open in a browser: file:///{args.out.resolve().as_posix()}")


if __name__ == "__main__":
    main()
