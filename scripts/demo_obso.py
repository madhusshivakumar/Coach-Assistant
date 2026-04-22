"""Phase-2 Slice B smoke: render the OBSO surface for a single attacking frame.

Picks a frame where the ball is in the middle third and one team is pushing forward,
so the hot OBSO zone lands near the opposition goal rather than on the halfway line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from football_analysis.analytics.pitch_control.obso import (
    compute_obso_frame,
    per_player_obso,
)
from football_analysis.config import get_settings
from football_analysis.viz.static.obso_surface import plot_obso_frame


def load_tracking(match_id: str) -> pd.DataFrame:
    settings = get_settings()
    key = match_id.split(":", 1)[-1]
    matching_dirs = list((settings.processed_dir / "tracking").rglob(f"match_id=metrica-{key}"))
    if not matching_dirs:
        raise SystemExit(f"No tracking parquet for {match_id!r}")
    parquets: list[Path] = []
    for d in matching_dirs:
        parquets.extend(d.rglob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in sorted(parquets)], ignore_index=True)


def pick_attacking_frame(tracking: pd.DataFrame, attacking_team_id: str) -> int:
    """Pick a frame where the attacking team has the ball in the opponent half.

    Heuristic: the ball is past the halfway line, and an attacker is within 5m of it.
    """
    ball = tracking[tracking["is_ball"] & tracking["visible"]].copy()
    ball = ball[ball["x"] > 60.0]  # in attacking zone
    for _, b in ball.iterrows():
        frame_id = int(b["frame_id"])
        atk = tracking[
            (tracking["frame_id"] == frame_id) & (tracking["team_id"] == attacking_team_id) & (~tracking["is_ball"])
        ]
        if atk.empty:
            continue
        near = (((atk["x"] - b["x"]) ** 2 + (atk["y"] - b["y"]) ** 2) ** 0.5 < 5.0).any()
        if near:
            return frame_id
    # Fallback: just any frame where ball is past halfway
    if not ball.empty:
        return int(ball.iloc[len(ball) // 2]["frame_id"])
    return int(tracking["frame_id"].iloc[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--match-id", default="metrica:1")
    p.add_argument("--attacking", default="home", help="attacking team_id")
    p.add_argument("--defending", default="away", help="defending team_id")
    p.add_argument("--frame-id", type=int, default=None)
    p.add_argument("--out", default="data/features/phase2/obso_demo.png", type=Path)
    args = p.parse_args()

    tracking = load_tracking(args.match_id)
    frame_id = args.frame_id if args.frame_id is not None else pick_attacking_frame(tracking, args.attacking)
    print(f"rendering OBSO at frame {frame_id} (attacking={args.attacking}, defending={args.defending})")

    surface = compute_obso_frame(tracking, frame_id, args.attacking, args.defending)
    peak = float(surface.obso.max())
    mean = float(surface.obso.mean())
    hot = float((surface.obso > 0.02).sum() / surface.obso.size)
    print(f"  OBSO peak={peak:.4f}  mean={mean:.5f}  'hot' cells (>0.02) cover {hot:.1%} of pitch")

    ranked = per_player_obso(tracking, frame_id, args.attacking, args.defending).head(5)
    print("  top 5 attacking players by OBSO at this frame:")
    for _, r in ranked.iterrows():
        print(
            f"    {r['player_id']:10s} @ ({r['x']:5.1f}, {r['y']:5.1f})"
            f"  OBSO={r['obso']:.4f}  (control={r['control']:.2f}  "
            f"arrival={r['arrival']:.2f}  goal={r['goal']:.2f})"
        )

    fig = plot_obso_frame(
        tracking,
        frame_id,
        attacking_team_id=args.attacking,
        defending_team_id=args.defending,
        surface=surface,
        team_names={args.attacking: "Home", args.defending: "Away"},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
