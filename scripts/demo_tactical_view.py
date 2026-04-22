"""Phase-2 Slice C smoke: save an animated Plotly tactical view to standalone HTML.

Defaults to 50 frames (2 s at 25 Hz) starting at a frame where the ball is in the
attacking third so the animation shows real build-up play, not a kickoff.

Passing --with-pitch-control overlays the Spearman surface per frame. This is
~1 s per frame of pre-compute at 34x52 resolution, so is opt-in.
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


def pick_attacking_window(tracking: pd.DataFrame, length: int = 50) -> tuple[int, int]:
    """Pick a start frame where the ball is in the attacking third and one team is closing."""
    ball = tracking[tracking["is_ball"] & tracking["visible"]].copy()
    attacking = ball[ball["x"] > 60.0]
    if attacking.empty:
        start = int(ball["frame_id"].iloc[len(ball) // 4]) if not ball.empty else 1
    else:
        start = int(attacking["frame_id"].iloc[len(attacking) // 4])
    return start, start + length - 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--match-id", default="metrica:1")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--length", type=int, default=50, help="frames, if start/end not given")
    p.add_argument(
        "--with-pitch-control", action="store_true", help="pre-compute and overlay the pitch-control surface per frame"
    )
    p.add_argument("--out", default=None, type=Path)
    args = p.parse_args()

    tracking = load_tracking(args.match_id)
    if args.start is None or args.end is None:
        start, end = pick_attacking_window(tracking, length=args.length)
    else:
        start, end = args.start, args.end

    out = args.out or Path(
        f"data/features/phase2/tactical_view{'_with_control' if args.with_pitch_control else ''}.html"
    )

    print(f"rendering frames {start}-{end} (with_pitch_control={args.with_pitch_control})")
    fig = animate(
        tracking,
        home_team_id="home",
        away_team_id="away",
        frame_range=(start, end),
        title=f"Tactical view — {args.match_id} frames {start}-{end}",
        with_pitch_control=args.with_pitch_control,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True, auto_play=False)
    print(f"wrote {out}  ({end - start + 1} frames, {out.stat().st_size // 1024} KB)")
    print(f"open in browser: file:///{out.resolve().as_posix()}")


if __name__ == "__main__":
    main()
