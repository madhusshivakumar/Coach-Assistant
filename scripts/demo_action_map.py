"""Phase-0 smoke: render all on-ball actions from a processed events Parquet.

Passes as arrows, dribbles as dashed arrows, shots as stars, defensive actions as X markers.
Uses only Phase-0 modules (mplsoccer + pandas) — no Phase-1 analytics.
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
        raise SystemExit(f"No events parquet under {root}.")
    if match_id is None:
        return paths[0]
    match_key = match_id.split(":", 1)[-1]
    for p in paths:
        if match_key in p.stem:
            return p
    raise SystemExit(f"No parquet matches {match_id!r}.")


def render(path: Path, out: Path) -> None:
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} events from {path.name}")

    pitch = Pitch(pitch_type="custom", pitch_length=105, pitch_width=68, line_color="#333")
    fig, ax = pitch.draw(figsize=(10, 7))

    passes = df[(df["action_type"] == "pass") & df["end_x"].notna()]
    for _, r in passes.iterrows():
        color = "#1f77b4" if r["result"] == "success" else "#888"
        pitch.arrows(
            r["start_x"],
            r["start_y"],
            r["end_x"],
            r["end_y"],
            ax=ax,
            color=color,
            width=2,
            alpha=0.85,
            headwidth=6,
            headlength=6,
            zorder=2,
        )

    dribbles = df[(df["action_type"] == "dribble") & df["end_x"].notna()]
    for _, r in dribbles.iterrows():
        pitch.arrows(
            r["start_x"],
            r["start_y"],
            r["end_x"],
            r["end_y"],
            ax=ax,
            color="#2ca02c",
            width=2,
            linestyle="--",
            alpha=0.9,
            zorder=2,
        )

    shots = df[df["action_type"] == "shot"]
    goals = shots[shots["result"] == "success"]
    others = shots[shots["result"] != "success"]
    if not others.empty:
        pitch.scatter(
            others["start_x"],
            others["start_y"],
            ax=ax,
            s=90,
            color="#cc3344",
            edgecolors="black",
            zorder=3,
            label="shot",
        )
    if not goals.empty:
        pitch.scatter(
            goals["start_x"],
            goals["start_y"],
            ax=ax,
            s=180,
            color="#ffd54a",
            edgecolors="black",
            linewidth=1.5,
            marker="*",
            zorder=4,
            label="goal",
        )

    def_actions = df[df["action_type"].isin(["interception", "clearance", "tackle", "foul"])]
    if not def_actions.empty:
        pitch.scatter(
            def_actions["start_x"],
            def_actions["start_y"],
            ax=ax,
            s=100,
            color="#555",
            marker="x",
            linewidth=2,
            zorder=3,
            label="def. action",
        )

    ax.set_title(
        f"Action map — {path.stem}  "
        f"({len(passes)} passes, {len(dribbles)} dribbles, {len(shots)} shots, {len(def_actions)} def.)"
    )
    ax.legend(loc="upper left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--match-id", default=None)
    p.add_argument("--out", default="data/features/demo_action_map.png", type=Path)
    args = p.parse_args()
    render(find_events_parquet(args.match_id), args.out)


if __name__ == "__main__":
    main()
