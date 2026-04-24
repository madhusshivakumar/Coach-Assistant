"""Phase-3 Slice A smoke: run the phase classifier + Bialkowski role assignment +
compactness time-series against a Metrica match and render:

    data/features/phase3/phase_timeline.png    phase-of-play colour-strip over time
    data/features/phase3/roles_frame.png       one tactical frame with role labels
    data/features/phase3/compactness.png       compactness time-series per team
    data/features/phase3/summary.json          counts + durations + top roles
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import Pitch

from football_analysis.analytics.formations.roles import (
    assign_roles,
    best_template_for_frame,
)
from football_analysis.analytics.formations.shape import shape_time_series
from football_analysis.analytics.phases.classifier import (
    PHASE_LABELS,
    classify_frames,
    segment_phases,
)
from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M
from football_analysis.config import get_settings

_PHASE_COLORS = {
    "build_up": "#5a8fd2",
    "progression": "#38b37a",
    "finishing": "#d65a5a",
    "att_transition": "#f2c14e",
    "def_transition": "#b56a5e",
    "settled_def": "#7f7f7f",
    "set_piece": "#bbbbbb",
}


def load_tracking(match_id: str) -> pd.DataFrame:
    settings = get_settings()
    key = match_id.split(":", 1)[-1]
    parquets: list[Path] = []
    for d in (settings.processed_dir / "tracking").rglob(f"match_id=metrica-{key}"):
        parquets.extend(d.rglob("*.parquet"))
    if not parquets:
        raise SystemExit(f"No tracking parquet for {match_id!r}")
    return pd.concat([pd.read_parquet(p) for p in sorted(parquets)], ignore_index=True)


def render_phase_timeline(classified: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 1.8))
    for _, r in segment_phases(classified).iterrows():
        ax.axvspan(r["start_time"], r["end_time"], color=_PHASE_COLORS.get(r["phase"], "#cccccc"), alpha=0.8)
    ax.set_xlim(classified["time_seconds"].min(), classified["time_seconds"].max())
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("time in match (s)")
    # Legend
    patches = [
        plt.matplotlib.patches.Patch(color=_PHASE_COLORS[p], label=p) for p in PHASE_LABELS if p in _PHASE_COLORS
    ]
    ax.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=7, fontsize=8)
    ax.set_title("Phase of play (tracking-based rules, 8 min sample)")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def render_roles_frame(tracking: pd.DataFrame, frame_id: int, team_id: str, out: Path) -> dict:
    frame = tracking[tracking["frame_id"] == frame_id]
    players = frame[(frame["team_id"] == team_id) & (~frame["is_ball"]) & frame["visible"]]
    # Drop GK (lowest mean x across match) and keep 10
    mean_x_by_player = (
        tracking[(tracking["team_id"] == team_id) & (~tracking["is_ball"])].groupby("player_id")["x"].mean()
    )
    gk_id = mean_x_by_player.idxmin()
    outfielders = players[players["player_id"] != gk_id].head(10)
    if len(outfielders) != 10:
        raise SystemExit(f"frame {frame_id} has {len(outfielders)} outfielders for {team_id!r}, need 10")

    best_tpl, cost = best_template_for_frame(outfielders[["x", "y"]].assign(player_id=outfielders["player_id"]))
    assigned = assign_roles(outfielders[["player_id", "x", "y"]], template=best_tpl)

    pitch = Pitch(pitch_type="custom", pitch_length=PITCH_LENGTH_M, pitch_width=PITCH_WIDTH_M, line_color="#333")
    fig, ax = pitch.draw(figsize=(11.5, 7.4))
    pitch.scatter(assigned["x"], assigned["y"], ax=ax, s=520, color="#1f77b4", edgecolors="black", zorder=3, alpha=0.85)
    for _, r in assigned.iterrows():
        ax.annotate(
            r["role"],
            xy=(r["x"], r["y"]),
            ha="center",
            va="center",
            fontsize=9,
            color="white",
            fontweight="bold",
            zorder=4,
        )
    ball = frame[frame["is_ball"] & frame["visible"]]
    if not ball.empty:
        pitch.scatter(ball["x"], ball["y"], ax=ax, s=180, color="white", edgecolors="black", linewidth=1.5, zorder=5)

    ax.set_title(
        f"Role assignment — {team_id}, frame {frame_id}\n"
        f"Best template: {best_tpl.name}   total displacement cost: {cost:.0f}"
    )
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return {"template": best_tpl.name, "cost": cost, "assignments": assigned.to_dict("records")}


def render_compactness(tracking: pd.DataFrame, out: Path) -> dict:
    home_ts = shape_time_series(tracking, team_id="home", attacking_right=True)
    away_ts = shape_time_series(tracking, team_id="away", attacking_right=False)

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for df, label, color in [(home_ts, "Home", "#1f77b4"), (away_ts, "Away", "#d62728")]:
        axes[0].plot(df["time_seconds"], df["convex_hull_area"], color=color, label=label, lw=1.5)
        axes[1].plot(df["time_seconds"], df["length"], color=color, label=f"{label} length", lw=1.2)
        axes[1].plot(df["time_seconds"], df["width"], color=color, label=f"{label} width", lw=1.0, ls="--", alpha=0.7)
        axes[2].plot(df["time_seconds"], df["vertical_compactness"], color=color, label=f"{label} vertical", lw=1.5)

    axes[0].set_ylabel("hull area (m²)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set_ylabel("length / width (m)")
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)
    axes[2].set_ylabel("vertical compactness (m)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[0].set_title("Team shape over the ingested sample")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {
        "home_mean_hull": round(float(home_ts["convex_hull_area"].mean()), 1),
        "away_mean_hull": round(float(away_ts["convex_hull_area"].mean()), 1),
        "home_mean_vertical_compactness": round(float(home_ts["vertical_compactness"].mean()), 2),
        "away_mean_vertical_compactness": round(float(away_ts["vertical_compactness"].mean()), 2),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--match-id", default="metrica:1")
    p.add_argument("--role-frame", type=int, default=None, help="Specific frame to render the role assignment on")
    p.add_argument("--out-dir", type=Path, default=Path("data/features/phase3"))
    args = p.parse_args()

    tracking = load_tracking(args.match_id)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Phase classifier
    classified = classify_frames(tracking, home_team_id="home", away_team_id="away")
    counts = classified["phase"].value_counts().to_dict()
    durations = segment_phases(classified).groupby("phase")["duration_s"].sum().round(2).to_dict()
    print(f"frames: {len(classified)}  phase counts: {counts}")
    render_phase_timeline(classified, args.out_dir / "phase_timeline.png")

    # 2) Role assignment at a frame where possession is settled in the middle third
    if args.role_frame is not None:
        role_frame = args.role_frame
    else:
        candidates = classified[classified["phase"].isin({"progression", "build_up"})]
        role_frame = (
            int(candidates["frame_id"].iloc[len(candidates) // 3])
            if not candidates.empty
            else int(classified["frame_id"].iloc[0])
        )
    role_summary = render_roles_frame(tracking, role_frame, team_id="home", out=args.out_dir / "roles_frame.png")
    print(f"role frame {role_frame}: template={role_summary['template']}, cost={role_summary['cost']:.0f}")

    # 3) Compactness time-series
    compact_summary = render_compactness(tracking, args.out_dir / "compactness.png")
    h_hull = compact_summary["home_mean_hull"]
    a_hull = compact_summary["away_mean_hull"]
    print(f"compactness: home_mean_hull={h_hull}  away_mean_hull={a_hull}")

    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "match_id": args.match_id,
                "phase_counts": {k: int(v) for k, v in counts.items()},
                "phase_durations_s": {k: float(v) for k, v in durations.items()},
                "role_frame": role_frame,
                "role_template": role_summary["template"],
                "role_cost": role_summary["cost"],
                "compactness": compact_summary,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
