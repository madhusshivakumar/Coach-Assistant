"""Phase-2 smoke: render a single Metrica frame with the Spearman pitch-control surface.

Usage:
    uv run python scripts/demo_pitch_control.py --match-id metrica:1 --frame-id 1000

If the match isn't ingested yet, runs `fa-data ingest metrica --match-id N --limit ...`
equivalent inline so the demo works from a cold start.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from football_analysis.analytics.pitch_control.spearman import compute_frame_from_tracking
from football_analysis.config import get_settings
from football_analysis.viz.interactive.pitch_control import plot_frame


def load_tracking(match_id: str) -> pd.DataFrame:
    settings = get_settings()
    key = match_id.split(":", 1)[-1]
    matching_dirs = list((settings.processed_dir / "tracking").rglob(f"match_id=metrica-{key}"))
    if not matching_dirs:
        raise SystemExit(
            f"No tracking parquet for {match_id!r}. Run `uv run fa-data ingest metrica --match-id {key}` first.",
        )
    parquets = []
    for d in matching_dirs:
        parquets.extend(d.rglob("*.parquet"))
    if not parquets:
        raise SystemExit(f"match_id directory exists but contains no parquet files: {matching_dirs[0]}")
    return pd.concat([pd.read_parquet(p) for p in sorted(parquets)], ignore_index=True)


def pick_interesting_frame(tracking: pd.DataFrame) -> int:
    """Pick a frame where the ball is near the middle of the pitch (mid-build-up).

    Skips the first few frames (kickoff is not visually interesting).
    """
    ball_frames = tracking[tracking["is_ball"] & tracking["visible"]].copy()
    # mid-pitch-ness metric: closer to centre circle = more interesting than kickoff/corner
    ball_frames["dist_to_centre"] = ((ball_frames["x"] - 52.5) ** 2 + (ball_frames["y"] - 34.0) ** 2) ** 0.5
    # skip first 5s worth of frames
    ball_frames = ball_frames[ball_frames["time_seconds"] > 10.0]
    if ball_frames.empty:
        return int(tracking["frame_id"].iloc[0])
    # Pick a frame within 15-30m of centre (good for a visible action)
    cand = ball_frames[(ball_frames["dist_to_centre"] > 15.0) & (ball_frames["dist_to_centre"] < 30.0)]
    if cand.empty:
        cand = ball_frames
    return int(cand.iloc[len(cand) // 4]["frame_id"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--match-id", default="metrica:1", help="e.g. metrica:1")
    p.add_argument("--frame-id", type=int, default=None, help="Frame to render; auto-picks if omitted")
    p.add_argument(
        "--out",
        default="data/features/phase2/pitch_control_demo.png",
        type=Path,
    )
    args = p.parse_args()

    tracking = load_tracking(args.match_id)
    team_ids = sorted(t for t in tracking["team_id"].dropna().unique())
    if len(team_ids) != 2:
        print(f"[ERROR] expected exactly 2 team_ids, got {team_ids}", file=sys.stderr)
        sys.exit(1)
    # Metrica convention: "home" team_id comes first alphabetically
    home_id, away_id = team_ids  # 'away' < 'home' alphabetically, so swap
    if home_id == "away":
        home_id, away_id = "home", "away"

    frame_id = args.frame_id if args.frame_id is not None else pick_interesting_frame(tracking)
    print(f"rendering {args.match_id} frame {frame_id} (home={home_id}, away={away_id})")

    control = compute_frame_from_tracking(tracking, frame_id, home_id, away_id)
    home_pct = float(control.home_control.mean() * 100)
    print(f"avg pitch control at this frame: home {home_pct:.1f}% / away {100 - home_pct:.1f}%")

    fig = plot_frame(
        tracking,
        frame_id=frame_id,
        home_team_id=home_id,
        away_team_id=away_id,
        control=control,
        team_names={home_id: "Home", away_id: "Away"},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
