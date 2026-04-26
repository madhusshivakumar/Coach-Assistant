# ruff: noqa: E501, PLR0915, RUF046, B007
"""Complete Phase 4 walkthrough — engine, attribution, OBSO trajectory, decisive
moment, retrieval, pattern, animation, movement traces, and narrative — all for
one auto-selected episode.

Outputs to ``data/features/phase4_complete/``:

- ``STORY.md`` — narrative walkthrough with every layer's output inline.
- ``play.gif`` — animated playback of the episode (all 22 players + ball).
- ``movement_traces.png`` — every attacker's path + ball path, decisive + peak
  moments annotated.
- ``obso_timeseries.png`` — OBSO_max vs time, decisive + peak frames marked.
- ``pitch_at_peak.png`` — pitch at peak frame, attackers colored by Δ OBSO.
- ``ball_trajectory.png`` — ball x_oriented vs time.
- ``episode.json`` — machine-readable bundle of every layer's output.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation

from football_analysis.analytics.episodes.contribution import compute_episode_attribution
from football_analysis.analytics.episodes.engine import build_episodes
from football_analysis.analytics.episodes.index import EpisodeIndex
from football_analysis.analytics.episodes.narrative import build_narrative
from football_analysis.analytics.episodes.obso_trajectory import (
    compute_obso_trajectory,
    find_decisive_moment,
)
from football_analysis.analytics.episodes.outcomes import (
    PENALTY_AREA_X_MIN,
    PENALTY_AREA_Y_MAX,
    PENALTY_AREA_Y_MIN,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    SHOT_SPEED_THRESHOLD_M_S,
)
from football_analysis.analytics.episodes.patterns import (
    cluster_episodes,
    cluster_for_episode,
)
from football_analysis.config import get_settings


def load_tracking(match_id: str) -> pd.DataFrame:
    settings = get_settings()
    key = match_id.split(":", 1)[-1]
    parquets: list[Path] = []
    for d in (settings.processed_dir / "tracking").rglob(f"match_id=metrica-{key}"):
        parquets.extend(d.rglob("*.parquet"))
    if not parquets:
        raise SystemExit(f"No tracking parquet for {match_id!r}")
    return pd.concat([pd.read_parquet(p) for p in sorted(parquets)], ignore_index=True)


def draw_pitch(ax: plt.Axes) -> None:
    ax.set_xlim(-2, PITCH_LENGTH_M + 2)
    ax.set_ylim(-2, PITCH_WIDTH_M + 2)
    ax.set_aspect("equal")
    ax.add_patch(mpatches.Rectangle((0, 0), PITCH_LENGTH_M, PITCH_WIDTH_M, fill=False, color="white", linewidth=1.5))
    ax.plot([PITCH_LENGTH_M / 2, PITCH_LENGTH_M / 2], [0, PITCH_WIDTH_M], color="white", linewidth=1.0)
    ax.add_patch(
        mpatches.Circle((PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2), 9.15, fill=False, color="white", linewidth=1.0)
    )
    for side_x in (0, PITCH_LENGTH_M - 16.5):
        ax.add_patch(
            mpatches.Rectangle(
                (side_x, PENALTY_AREA_Y_MIN),
                16.5,
                PENALTY_AREA_Y_MAX - PENALTY_AREA_Y_MIN,
                fill=False,
                color="white",
                linewidth=1.0,
            )
        )
    ax.plot([0, 0], [PITCH_WIDTH_M / 2 - 3.66, PITCH_WIDTH_M / 2 + 3.66], color="white", linewidth=2.5)
    ax.plot(
        [PITCH_LENGTH_M, PITCH_LENGTH_M],
        [PITCH_WIDTH_M / 2 - 3.66, PITCH_WIDTH_M / 2 + 3.66],
        color="white",
        linewidth=2.5,
    )
    ax.set_facecolor("#0e6b3a")
    ax.set_xticks([])
    ax.set_yticks([])


def render_pitch_at_peak(tracking, record, attribution, out_path: Path) -> None:
    peak_frame = attribution.peak_frame
    sub = tracking[tracking["frame_id"] == peak_frame]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 8.0), dpi=150)
    draw_pitch(ax)
    attacking_team = record.boundary.possession_team
    contribs = attribution.contributions
    max_abs = max(0.001, max((abs(v) for v in contribs.values()), default=0.001))

    # Render attackers in weakest-first order so contributors land on top.
    rows_atk = [r for _, r in sub.iterrows() if not r["is_ball"] and r["team_id"] == attacking_team]
    rows_atk.sort(key=lambda r: abs(contribs.get(str(r["player_id"]), 0.0)))
    rows_def = [r for _, r in sub.iterrows() if not r["is_ball"] and r["team_id"] != attacking_team]

    # Defenders: small gray X markers (no labels — context only).
    for r in rows_def:
        ax.scatter(r["x"], r["y"], s=44, c="#9e9e9e", marker="x", linewidths=1.4, zorder=3)

    # Attackers: size + saturation scale with |contribution|.
    for r in rows_atk:
        pid = str(r["player_id"])
        d = contribs.get(pid, 0.0)
        color, size, _ = _contrib_visual(d, max_abs)
        # Pitch-at-peak deserves slightly larger markers than the trace plot.
        size = 60 + (size - 24) * 2.5  # widen the dynamic range a bit
        ax.scatter(r["x"], r["y"], s=size, c=[color], edgecolors="white", linewidths=1.4, zorder=5)
        label = pid.rsplit("_", maxsplit=1)[-1] if "_" in pid else pid
        rel = abs(d) / max_abs if max_abs > 0 else 0.0
        fontsize = 9 + int(round(3 * rel))
        _label_with_halo(ax, float(r["x"]), float(r["y"]), label, fontsize=fontsize, dx=1.8, dy=1.8)

    ball = sub[sub["is_ball"] & sub["visible"]]
    if not ball.empty:
        bx, by = float(ball["x"].iloc[0]), float(ball["y"].iloc[0])
        ax.scatter(bx, by, s=180, c="white", edgecolors="black", linewidths=1.5, marker="*", zorder=10)

    ax.set_title(
        f"Episode {record.boundary.episode_id} — peak at frame {peak_frame}  "
        f"(max OBSO {attribution.peak_obso:.3f})\n"
        f"Marker size scales with |Δ OBSO| contribution; gray X = defenders",
        fontsize=10,
    )
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(vmin=-max_abs, vmax=max_abs))
    fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02).set_label("Δ OBSO (contribution at peak)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="#0a4d28")
    plt.close(fig)


def render_ball_trajectory(record, narrative, attribution, decisive, strike, out_path: Path) -> None:
    states = record.state_trajectory
    if states.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=140)
    ax.plot(states["rel_time_s"], states["ball_x_oriented"], color="#2c7fb8", linewidth=2.0, marker="o", markersize=4)
    ax.axhline(70.0, color="orange", linestyle="--", linewidth=1.0, alpha=0.6, label="final-third (x=70)")
    ax.axhline(
        PENALTY_AREA_X_MIN,
        color="red",
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
        label=f"penalty area (x={PENALTY_AREA_X_MIN:.1f})",
    )
    if decisive is not None:
        ax.axvline(
            decisive.rel_time_s,
            color="orange",
            linewidth=2.0,
            label=f"decisive (t+{decisive.rel_time_s:.2f}s, OBSO={decisive.obso_at_decisive:.2f})",
        )
    if attribution is not None:
        peak_state = states[states["frame_id"] == attribution.peak_frame]
        if not peak_state.empty:
            t_peak = float(peak_state["rel_time_s"].iloc[0])
            ax.axvline(
                t_peak, color="red", linewidth=2.0, label=f"peak (t+{t_peak:.2f}s, OBSO={attribution.peak_obso:.2f})"
            )
    if strike is not None:
        ax.axvline(
            strike["strike_rel_time_s"],
            color="#ffd60a",
            linewidth=2.4,
            label=(
                f"strike (t+{strike['strike_rel_time_s']:.2f}s, by {strike['shot_taker']}, "
                f"{strike['ball_speed_at_strike']:.1f} m/s)"
            ),
        )
    ax.set_xlabel("seconds since episode start")
    ax.set_ylabel("ball x_oriented (m)  [forward is +x]")
    ax.set_title(f"Episode {record.boundary.episode_id} — ball trajectory")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_obso_timeseries(traj: pd.DataFrame, decisive, attribution, out_path: Path) -> None:
    if traj.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=140)
    ax.plot(
        traj["rel_time_s"],
        traj["obso_max"],
        color="#d62728",
        linewidth=2.2,
        marker="o",
        markersize=5,
        label="OBSO max (this frame)",
    )
    if decisive is not None:
        ax.axvline(
            decisive.rel_time_s, color="orange", linewidth=2.0, label=f"decisive moment (t+{decisive.rel_time_s:.2f}s)"
        )
        ax.axhline(
            decisive.threshold_pct * decisive.peak_obso,
            color="orange",
            linestyle=":",
            alpha=0.5,
            label=f"{int(decisive.threshold_pct * 100)}% of peak threshold",
        )
    if attribution is not None:
        peak_in_traj = traj[traj["frame_id"] == attribution.peak_frame]
        if not peak_in_traj.empty:
            ax.axvline(float(peak_in_traj["rel_time_s"].iloc[0]), color="red", linewidth=2.0, label="peak frame")
    ax.set_xlabel("seconds since episode start")
    ax.set_ylabel("OBSO max")
    ax.set_title("OBSO trajectory — when did the threat actually build?")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def find_shot_strike(record, tracking) -> dict | None:
    """Identify the most likely shot-strike inside the episode.

    Heuristic on tracking-only data: the strike is the frame where the ball's
    speed first crosses the shot-speed threshold (default 12 m/s). The
    shot-taker is the closest attacker to the ball at that frame.

    Falls back to the frame of *peak* ball-speed if no frame crosses the
    threshold — useful for episodes where the heuristic shot-detector tripped
    on a late-burst clearance instead of a clean strike.

    Returns ``None`` when there's no visible-ball frame in the episode.
    """
    in_ep = (tracking["frame_id"] >= record.boundary.start_frame) & (tracking["frame_id"] <= record.boundary.end_frame)
    ball = tracking[in_ep & tracking["is_ball"] & tracking["visible"]].copy()
    if ball.empty:
        return None
    ball["speed"] = np.hypot(ball["vx"], ball["vy"])

    fast = ball[ball["speed"] >= SHOT_SPEED_THRESHOLD_M_S].sort_values("frame_id")
    strike = fast.iloc[0] if not fast.empty else ball.loc[ball["speed"].idxmax()]

    strike_frame = int(strike["frame_id"])
    bx, by = float(strike["x"]), float(strike["y"])

    attackers = tracking[
        (tracking["frame_id"] == strike_frame)
        & (tracking["team_id"] == record.boundary.possession_team)
        & ~tracking["is_ball"]
        & tracking["visible"]
    ].copy()
    if attackers.empty:
        return None
    attackers["dist"] = np.hypot(attackers["x"] - bx, attackers["y"] - by)
    closest = attackers.sort_values("dist").iloc[0]

    return {
        "strike_frame": strike_frame,
        "strike_time_s": float(strike["time_seconds"]),
        "strike_rel_time_s": float(strike["time_seconds"] - record.boundary.start_time_s),
        "ball_speed_at_strike": float(strike["speed"]),
        "ball_x": bx,
        "ball_y": by,
        "shot_taker": str(closest["player_id"]),
        "shot_taker_distance_m": float(closest["dist"]),
        "above_threshold": not fast.empty,
    }


def _contrib_visual(d: float, max_abs: float) -> tuple[tuple, float, float]:
    """Map a contribution magnitude to (color, marker_size, line_width).

    Players with d ≈ 0 fade into the background so the actual contributor pops.
    """
    if max_abs <= 0:
        return plt.cm.RdBu_r(0.5), 24, 1.0
    rel = abs(d) / max_abs  # 0..1
    t = 0.5 + 0.5 * (d / max_abs)
    color = plt.cm.RdBu_r(t)
    # Marker: small (24) at noise floor, scaling to 110 at max-contributor.
    size = 24 + 86 * rel
    # Line: faint at noise floor, prominent at max-contributor.
    lw = 0.9 + 1.8 * rel
    return color, size, lw


def _label_with_halo(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    fontsize: int = 9,
    dx: float = 1.6,
    dy: float = 1.6,
    text_color: str = "white",
    halo: str = "black",
) -> None:
    """Render a number label OFFSET from (x, y) so it doesn't sit on the marker,
    with a stroke halo so it's readable against any pitch color."""
    ax.text(
        x + dx,
        y + dy,
        text,
        color=text_color,
        fontsize=fontsize,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=20,
        path_effects=[pe.withStroke(linewidth=2.4, foreground=halo)],
    )


def render_movement_traces(tracking, record, attribution, decisive, strike, out_path: Path) -> None:
    """Per-attacker movement paths over the episode, overlaid on the pitch.

    Visual hierarchy:
      - Attackers with non-trivial contribution: thick colored trace, larger marker,
        halo'd offset label.
      - Attackers at noise floor (|d| ≈ 0): faint thin trace, tiny marker,
        small offset label.
      - Defenders: dim gray X markers (no traces, no labels) — context only.
      - Ball: white dashed line, star at end.
    """
    fig, ax = plt.subplots(figsize=(13, 8.0), dpi=150)
    draw_pitch(ax)

    attacking_team = record.boundary.possession_team
    contribs = attribution.contributions if attribution else {}
    max_abs = max(0.001, max((abs(v) for v in contribs.values()), default=0.001))

    sub = tracking[
        (tracking["frame_id"] >= record.boundary.start_frame) & (tracking["frame_id"] <= record.boundary.end_frame)
    ]

    # Defenders first (background layer).
    for pid, group in sub[~sub["is_ball"] & (sub["team_id"] != attacking_team)].groupby("player_id"):
        gp = group.sort_values("frame_id")
        ax.plot(gp["x"], gp["y"], color="#666666", linewidth=0.6, alpha=0.25, zorder=2)
        ax.scatter(
            gp["x"].iloc[-1],
            gp["y"].iloc[-1],
            s=44,
            c="#9e9e9e",
            marker="x",
            linewidths=1.4,
            zorder=3,
        )

    # Attackers — order weakest-first so contributors render on top.
    attackers = list(sub[~sub["is_ball"] & (sub["team_id"] == attacking_team)].groupby("player_id"))
    attackers.sort(key=lambda kv: abs(contribs.get(str(kv[0]), 0.0)))
    for pid, group in attackers:
        gp = group.sort_values("frame_id")
        d = contribs.get(str(pid), 0.0)
        color, size, lw = _contrib_visual(d, max_abs)
        ax.plot(gp["x"], gp["y"], color=color, linewidth=lw, alpha=0.92, zorder=4)
        # Start: small open circle (just a hint of where they began).
        ax.scatter(
            gp["x"].iloc[0],
            gp["y"].iloc[0],
            s=22,
            facecolors="none",
            edgecolors=color,
            linewidths=1.3,
            alpha=0.7,
            zorder=5,
        )
        # End: filled marker, sized by contribution magnitude.
        ax.scatter(
            gp["x"].iloc[-1],
            gp["y"].iloc[-1],
            s=size,
            c=[color],
            edgecolors="white",
            linewidths=1.2,
            zorder=6,
        )
        # Label, offset and halo'd.
        label = str(pid).split("_")[-1] if "_" in str(pid) else str(pid)
        # Bigger / slightly offset for contributors so they pop.
        rel = abs(d) / max_abs
        fontsize = 8 + int(round(3 * rel))  # 8..11
        _label_with_halo(ax, float(gp["x"].iloc[-1]), float(gp["y"].iloc[-1]), label, fontsize=fontsize, dx=1.8, dy=1.8)

    # Ball path on top.
    ball = sub[sub["is_ball"] & sub["visible"]].sort_values("frame_id")
    if not ball.empty:
        ax.plot(
            ball["x"],
            ball["y"],
            color="#ffffff",
            linewidth=1.8,
            alpha=0.85,
            linestyle="--",
            zorder=8,
        )
        ax.scatter(
            ball["x"].iloc[-1],
            ball["y"].iloc[-1],
            s=180,
            c="white",
            edgecolors="black",
            linewidths=1.5,
            marker="*",
            zorder=10,
        )

    # Highlight the shot-taker with a yellow ring + strike-burst at the ball-strike position.
    if strike is not None:
        shot_taker_id = strike["shot_taker"]
        peak_frame = strike["strike_frame"]
        st_row = sub[(sub["frame_id"] == peak_frame) & (sub["player_id"] == shot_taker_id)]
        if not st_row.empty:
            stx = float(st_row["x"].iloc[0])
            sty = float(st_row["y"].iloc[0])
            ax.scatter(
                stx,
                sty,
                s=420,
                facecolors="none",
                edgecolors="#ffd60a",
                linewidths=2.6,
                zorder=11,
            )
        ax.scatter(
            strike["ball_x"],
            strike["ball_y"],
            s=420,
            c="#ffd60a",
            marker="*",
            edgecolors="black",
            linewidths=1.5,
            alpha=0.7,
            zorder=12,
        )

    title = f"Episode {record.boundary.episode_id} — movement traces over {record.boundary.duration_s:.2f}s\n"
    if decisive is not None and attribution is not None:
        title += (
            f"Decisive at t+{decisive.rel_time_s:.2f}s (OBSO {decisive.obso_at_decisive:.2f}). "
            f"Peak OBSO {attribution.peak_obso:.2f}. "
        )
    if strike is not None:
        title += (
            f"Shot taken by {strike['shot_taker']} at t+{strike['strike_rel_time_s']:.2f}s "
            f"({strike['ball_speed_at_strike']:.1f} m/s, yellow ring & burst)."
        )
    ax.set_title(title, fontsize=10)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(vmin=-max_abs, vmax=max_abs))
    fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02).set_label("Δ OBSO contribution at peak", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="#0a4d28")
    plt.close(fig)


def render_animation(tracking, record, attribution, decisive, strike, out_path: Path) -> None:
    """GIF of the episode with cumulative movement trails.

    Each frame draws every player's path *up to that frame* (not just their
    current position) so the trail grows over the playback. Same marker /
    label / color logic as the static traces — defenders are dim X markers,
    attackers are sized + colored by their contribution.
    """
    states = record.state_trajectory
    if states.empty:
        return
    snap_frames = states["frame_id"].astype(int).tolist()
    attacking_team = record.boundary.possession_team
    contribs = attribution.contributions if attribution else {}
    max_abs = max(0.001, max((abs(v) for v in contribs.values()), default=0.001))

    # Pre-slice the episode tracking once.
    sub_all = tracking[
        (tracking["frame_id"] >= record.boundary.start_frame) & (tracking["frame_id"] <= record.boundary.end_frame)
    ].sort_values("frame_id")

    fig, ax = plt.subplots(figsize=(13, 8.0), dpi=110)
    draw_pitch(ax)

    # Static colorbar (drawn once on the figure; ax.clear() preserves it).
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(vmin=-max_abs, vmax=max_abs))
    fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02).set_label("Δ OBSO contribution at peak", fontsize=9)

    def update(i: int):
        ax.clear()
        draw_pitch(ax)
        frame_id = snap_frames[i]
        upto = sub_all[sub_all["frame_id"] <= frame_id]
        current = sub_all[sub_all["frame_id"] == frame_id]

        # --- Defender trails + current X markers (background) ---
        defender_trails = upto[~upto["is_ball"] & (upto["team_id"] != attacking_team)]
        for _pid, group in defender_trails.groupby("player_id"):
            gp = group.sort_values("frame_id")
            ax.plot(gp["x"], gp["y"], color="#666666", linewidth=0.5, alpha=0.28, zorder=2)
        for _, r in current.iterrows():
            if r["is_ball"] or r["team_id"] == attacking_team:
                continue
            ax.scatter(r["x"], r["y"], s=42, c="#9e9e9e", marker="x", linewidths=1.3, zorder=3)

        # --- Attacker trails (weakest-first so contributors render on top) ---
        atk_trails = list(upto[~upto["is_ball"] & (upto["team_id"] == attacking_team)].groupby("player_id"))
        atk_trails.sort(key=lambda kv: abs(contribs.get(str(kv[0]), 0.0)))
        for pid, group in atk_trails:
            gp = group.sort_values("frame_id")
            d = contribs.get(str(pid), 0.0)
            color, _, lw = _contrib_visual(d, max_abs)
            ax.plot(gp["x"], gp["y"], color=color, linewidth=lw, alpha=0.88, zorder=4)
            # Small open circle at start so you can read direction of travel.
            ax.scatter(
                gp["x"].iloc[0],
                gp["y"].iloc[0],
                s=20,
                facecolors="none",
                edgecolors=color,
                linewidths=1.1,
                alpha=0.7,
                zorder=5,
            )

        # --- Current-frame attacker markers + labels (foreground) ---
        atk_current = [r for _, r in current.iterrows() if not r["is_ball"] and r["team_id"] == attacking_team]
        atk_current.sort(key=lambda r: abs(contribs.get(str(r["player_id"]), 0.0)))
        shot_taker_id = strike["shot_taker"] if strike else None
        for r in atk_current:
            pid = str(r["player_id"])
            d = contribs.get(pid, 0.0)
            color, size, _ = _contrib_visual(d, max_abs)
            ax.scatter(r["x"], r["y"], s=size, c=[color], edgecolors="white", linewidths=1.2, zorder=6)
            # Persistent yellow ring around the shot-taker so they're
            # findable in every frame.
            if pid == shot_taker_id:
                ax.scatter(
                    r["x"], r["y"], s=size + 380, facecolors="none", edgecolors="#ffd60a", linewidths=2.4, zorder=7
                )
            label = pid.split("_")[-1] if "_" in pid else pid
            rel = abs(d) / max_abs if max_abs > 0 else 0.0
            fontsize = 8 + int(round(3 * rel))
            _label_with_halo(ax, float(r["x"]), float(r["y"]), label, fontsize=fontsize, dx=1.8, dy=1.8)

        # --- Ball trail + current ---
        ball = upto[upto["is_ball"] & upto["visible"]].sort_values("frame_id")
        if not ball.empty:
            ax.plot(ball["x"], ball["y"], color="white", linewidth=1.6, alpha=0.85, linestyle="--", zorder=8)
            ax.scatter(
                ball["x"].iloc[-1],
                ball["y"].iloc[-1],
                s=170,
                c="white",
                edgecolors="black",
                linewidths=1.5,
                marker="*",
                zorder=10,
            )

        rel_t = float(states.iloc[i]["rel_time_s"])
        marker = ""
        # Multiple markers can stack — show all that fire on this frame.
        markers: list[str] = []
        if decisive is not None and frame_id == decisive.frame_id:
            markers.append("DECISIVE")
        if attribution is not None and frame_id == attribution.peak_frame:
            markers.append("PEAK")
        # Strike: title marker + on-pitch yellow burst at the ball position.
        if strike is not None and frame_id >= strike["strike_frame"]:
            # Draw burst at strike location. Persists for all frames after
            # the strike so you can see where it came from.
            ax.scatter(
                strike["ball_x"],
                strike["ball_y"],
                s=380,
                c="#ffd60a",
                marker="*",
                edgecolors="black",
                linewidths=1.4,
                alpha=0.55,
                zorder=9,
            )
            if frame_id == strike["strike_frame"]:
                markers.append(f"STRIKE by {strike['shot_taker']} ({strike['ball_speed_at_strike']:.1f} m/s)")
        if markers:
            marker = "  ← " + " · ".join(markers)
        ax.set_title(
            f"Episode {record.boundary.episode_id} — t+{rel_t:.2f}s (frame {frame_id}){marker}",
            fontsize=11,
        )
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(snap_frames), interval=500, blit=False)
    anim.save(out_path, writer=animation.PillowWriter(fps=2))
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--match", default="metrica:1")
    p.add_argument("--out-dir", type=Path, default=Path("data/features/phase4_complete"))
    p.add_argument(
        "--episode-id", type=int, default=None, help="Override which episode to walk through (default: auto-pick)"
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tracking = load_tracking(args.match)
    print(f"loaded tracking: {len(tracking):,} rows, {tracking['frame_id'].nunique():,} frames")

    # Slice A
    records = build_episodes(
        tracking,
        home_team_id="home",
        away_team_id="away",
        attacking_directions={"home": "right", "away": "left"},
    )
    print(f"Slice A: {len(records)} episodes built")

    # Pick the focus episode
    if args.episode_id is not None:
        chosen = next((r for r in records if r.boundary.episode_id == args.episode_id), None)
        if chosen is None:
            raise SystemExit(f"No episode {args.episode_id}")
        attr = compute_episode_attribution(chosen, tracking, "home", "away")
    else:
        shot_like = [r for r in records if r.outcome.shot_like]
        if not shot_like:
            raise SystemExit("No shot_like episodes")
        chosen, attr = None, None
        for r in shot_like:
            a = compute_episode_attribution(r, tracking, "home", "away")
            if a is None:
                continue
            mc = max((abs(c) for c in a.contributions.values()), default=0)
            if attr is None or mc > max((abs(c) for c in attr.contributions.values()), default=0):
                chosen, attr = r, a
    assert chosen is not None and attr is not None
    narrative = build_narrative(chosen, attr)
    print(f"Slice B: episode {chosen.boundary.episode_id} (peak_obso={attr.peak_obso:.3f})")

    # OBSO trajectory + decisive moment
    obso_traj = compute_obso_trajectory(chosen, tracking, "home", "away")
    decisive = find_decisive_moment(obso_traj, threshold_pct=0.5)
    if decisive is not None:
        print(
            f"  decisive moment: t+{decisive.rel_time_s:.2f}s, "
            f"OBSO={decisive.obso_at_decisive:.3f} (50% of peak {decisive.peak_obso:.3f})"
        )

    # Shot-strike detection (who actually struck the ball)
    strike = find_shot_strike(chosen, tracking)
    if strike:
        marker = "above 12 m/s threshold" if strike["above_threshold"] else "no threshold cross — used peak speed"
        print(
            f"  shot strike: frame {strike['strike_frame']} (t+{strike['strike_rel_time_s']:.2f}s), "
            f"by {strike['shot_taker']} (dist {strike['shot_taker_distance_m']:.2f}m, "
            f"ball speed {strike['ball_speed_at_strike']:.1f} m/s, {marker})"
        )

    # Slice C
    index = EpisodeIndex(k_default=3)
    index.fit(records)
    neighbors = index.query(chosen, k=3)
    prediction = index.predict_outcome(chosen, k=3)
    clusters = cluster_episodes(index, n_clusters=8)
    cluster_of_query = cluster_for_episode(clusters, chosen.boundary.episode_id)

    # Visualizations
    pitch_png = args.out_dir / "pitch_at_peak.png"
    traj_png = args.out_dir / "ball_trajectory.png"
    obso_png = args.out_dir / "obso_timeseries.png"
    traces_png = args.out_dir / "movement_traces.png"
    gif_path = args.out_dir / "play.gif"

    render_pitch_at_peak(tracking, chosen, attr, pitch_png)
    render_ball_trajectory(chosen, narrative, attr, decisive, strike, traj_png)
    render_obso_timeseries(obso_traj, decisive, attr, obso_png)
    render_movement_traces(tracking, chosen, attr, decisive, strike, traces_png)
    print("rendering animation...")
    render_animation(tracking, chosen, attr, decisive, strike, gif_path)
    print(f"  -> {gif_path}")

    # JSON bundle
    bundle = {
        "match": args.match,
        "n_episodes": len(records),
        "selected": {
            "episode_id": chosen.boundary.episode_id,
            "boundary": asdict(chosen.boundary),
            "outcome": asdict(chosen.outcome),
            "dominant_phase": chosen.dominant_phase,
            "n_snapshots": len(chosen.state_trajectory),
        },
        "obso_trajectory": [
            {
                "frame_id": int(r["frame_id"]),
                "rel_time_s": round(float(r["rel_time_s"]), 2),
                "obso_max": round(float(r["obso_max"]), 4),
            }
            for _, r in obso_traj.iterrows()
        ],
        "decisive_moment": (
            {
                "frame_id": decisive.frame_id,
                "rel_time_s": round(decisive.rel_time_s, 2),
                "obso_at_decisive": round(decisive.obso_at_decisive, 4),
                "peak_obso": round(decisive.peak_obso, 4),
                "threshold_pct": decisive.threshold_pct,
            }
            if decisive
            else None
        ),
        "shot_strike": (
            {
                "strike_frame": strike["strike_frame"],
                "rel_time_s": round(strike["strike_rel_time_s"], 2),
                "shot_taker": strike["shot_taker"],
                "shot_taker_distance_m": round(strike["shot_taker_distance_m"], 2),
                "ball_speed_at_strike": round(strike["ball_speed_at_strike"], 2),
                "ball_x": round(strike["ball_x"], 2),
                "ball_y": round(strike["ball_y"], 2),
                "above_threshold": strike["above_threshold"],
            }
            if strike
            else None
        ),
        "attribution": {
            "peak_frame": attr.peak_frame,
            "peak_obso": round(attr.peak_obso, 4),
            "contributions": {
                k: round(v, 4) for k, v in sorted(attr.contributions.items(), key=lambda kv: -abs(kv[1]))
            },
        },
        "narrative": {
            "trigger_frame": narrative.trigger_frame,
            "trigger_time_s": narrative.trigger_time_s,
            "top_contributors": [{"player_id": p, "contribution": round(c, 4)} for p, c in narrative.top_contributors],
            "text": narrative.text,
        },
        "retrieval": {
            "neighbors": [
                {
                    "rank": n.rank,
                    "episode_id": n.record.boundary.episode_id,
                    "distance": round(n.distance, 4),
                    "shot_like": n.record.outcome.shot_like,
                    "ended_in_box": n.record.outcome.ended_in_box,
                    "reached_final_third": n.record.outcome.reached_final_third,
                    "dominant_phase": n.record.dominant_phase,
                }
                for n in neighbors
            ],
            "prediction": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in prediction.items()},
        },
        "pattern": (
            {
                "cluster_id": cluster_of_query.cluster_id,
                "label": cluster_of_query.label,
                "n_episodes_in_cluster": cluster_of_query.n_episodes,
                "all_member_ids": cluster_of_query.episode_ids,
            }
            if cluster_of_query
            else None
        ),
        "all_clusters": [{"cluster_id": c.cluster_id, "label": c.label, "n_episodes": c.n_episodes} for c in clusters],
    }
    (args.out_dir / "episode.json").write_text(json.dumps(bundle, indent=2, default=str), encoding="utf-8")
    (args.out_dir / "STORY.md").write_text(
        _render_story(bundle, gif_path.name, traces_png.name, obso_png.name, pitch_png.name, traj_png.name),
        encoding="utf-8",
    )

    print(f"\nwrote {args.out_dir / 'STORY.md'}")
    print("=" * 70)
    print("DEMO READY")
    print("=" * 70)
    for f in [gif_path, traces_png, obso_png, pitch_png, traj_png]:
        print(f"  {f}")


def _render_story(bundle, gif_name, traces_name, obso_name, pitch_name, traj_name) -> str:
    sel = bundle["selected"]
    attr = bundle["attribution"]
    narr = bundle["narrative"]
    retr = bundle["retrieval"]
    pat = bundle["pattern"]
    dec = bundle["decisive_moment"]
    strike = bundle["shot_strike"]

    contrib_rows = "\n".join(f"| `{pid}` | {c:+.3f} |" for pid, c in attr["contributions"].items())
    neighbor_rows = "\n".join(
        f"| {n['rank']} | {n['episode_id']} | {n['distance']:.3f} | {n['dominant_phase']} | "
        f"{'YES' if n['shot_like'] else 'no'} | {'YES' if n['ended_in_box'] else 'no'} |"
        for n in retr["neighbors"]
    )
    cluster_rows = "\n".join(
        f"| {c['cluster_id']} | {c['n_episodes']} | {c['label']} |" for c in bundle["all_clusters"]
    )
    pred = retr["prediction"]
    obso_rows = "\n".join(
        f"| {r['frame_id']} | t+{r['rel_time_s']:.2f}s | {r['obso_max']:.3f} |" for r in bundle["obso_trajectory"]
    )

    decisive_block = (
        f"**Decisive moment:** at **t+{dec['rel_time_s']:.2f}s** (frame {dec['frame_id']}) "
        f"OBSO crossed {int(dec['threshold_pct'] * 100)}% of its eventual peak — rising from "
        f"the early-episode baseline to **{dec['obso_at_decisive']:.3f}** on its way to "
        f"the peak of **{dec['peak_obso']:.3f}** at the end of the episode."
        if dec
        else "**Decisive moment:** OBSO never crossed 50% of peak — the threat didn't really build."
    )

    if strike:
        threshold_note = (
            "(crossed the 12 m/s shot-speed threshold)"
            if strike["above_threshold"]
            else "(no frame crossed 12 m/s — used peak ball speed as fallback)"
        )
        strike_block = (
            f"**Shot taken by `{strike['shot_taker']}`** at **t+{strike['rel_time_s']:.2f}s** "
            f"(frame {strike['strike_frame']}). Ball speed at strike: "
            f"**{strike['ball_speed_at_strike']:.1f} m/s** {threshold_note}. "
            f"`{strike['shot_taker']}` was the closest attacker to the ball "
            f"({strike['shot_taker_distance_m']:.2f} m away), at "
            f"({strike['ball_x']:.1f}, {strike['ball_y']:.1f}) on the pitch."
        )
    else:
        strike_block = "**Shot taken by:** could not detect a strike (no visible ball frames in episode)."

    pattern_block = (
        (f"This episode lives in **cluster {pat['cluster_id']}**:\n\n> {pat['label']}\n")
        if pat
        else "(no cluster assignment)\n"
    )

    return f"""# Phase 4 Complete Demo — Episode {sel["episode_id"]} end to end

**Match:** `{bundle["match"]}` · {bundle["n_episodes"]} possession episodes total · this story focuses on episode {sel["episode_id"]}, picked because it had the largest single-player contribution magnitude across the match.

> {strike_block}

---

## The play, animated

![Play animation]({gif_name})

A 2 Hz playback of all 22 players + ball through the full {sel["boundary"]["duration_s"]:.2f}s. The shot-taker is ringed in **yellow** in every frame so you can track them; a yellow burst marks the ball's position at the strike moment. The DECISIVE / PEAK / STRIKE markers fire in the title at the right frames.

---

## When did the play start to work?

That's a different question from "when did it start." The episode *began* at t={sel["boundary"]["start_time_s"]:.2f}s with the ball already in the final third — the trigger fires at t+0 because the ball was deep when possession started. But the threat doesn't build linearly:

![OBSO over time]({obso_name})

| frame | t | OBSO max |
|---|---|---|
{obso_rows}

{decisive_block}

So the answer to "when did it start [to work]" is the **decisive moment**, not the trigger or the start. The decisive moment is the earliest frame where the eventual outcome's expected value first crossed half of its peak — the inflection where everything before is build-up and everything after is execution.

---

## What made the play successful?

![Movement traces]({traces_name})

Every attacker's path through the {sel["boundary"]["duration_s"]:.2f}s is plotted, colored by their Δ OBSO contribution at peak. **Open circles are starting positions, filled circles are where they ended.** Length and curvature of each line shows movement. The white dashed line is the ball path.

The attribution table at peak frame `{attr["peak_frame"]}`:

| player_id | Δ OBSO at peak |
|---|---|
{contrib_rows}

**The single decisive movement was `{narr["top_contributors"][0]["player_id"]}` with Δ OBSO {narr["top_contributors"][0]["contribution"]:+.3f}.** That's the difference between this attack and the average attack at the same moment from the same shape. Look at their trace on the movement-traces image — the colored line that diverges most from "stay near my average position."

The other attackers' contributions are at the noise floor (≈0). Their movement was within their normal positional envelope; they didn't add or subtract threat at the peak. This isn't saying they were useless — it's saying the *peak frame* attribution is dominated by one player. A different frame might surface a different contributor.

---

## Episode metadata (Slice A)

| Field | Value |
|---|---|
| episode_id | {sel["episode_id"]} |
| possession_team | `{sel["boundary"]["possession_team"]}` |
| start_time_s | {sel["boundary"]["start_time_s"]:.2f} |
| end_time_s | {sel["boundary"]["end_time_s"]:.2f} |
| duration_s | {sel["boundary"]["duration_s"]:.2f} |
| dominant_phase | `{sel["dominant_phase"]}` |
| end_reason | `{sel["outcome"]["end_reason"]}` |
| shot_like | **{sel["outcome"]["shot_like"]}** |
| ended_in_box | {sel["outcome"]["ended_in_box"]} |
| reached_final_third | {sel["outcome"]["reached_final_third"]} |
| end_ball_x | {sel["outcome"]["end_ball_x"]:.1f} |
| end_ball_speed | {sel["outcome"]["end_ball_speed"]:.2f} m/s |

The ball's x-progression over time:

![Ball trajectory]({traj_name})

The static peak-frame view (positions at the OBSO peak):

![Pitch at peak frame]({pitch_name})

### Auto-generated narrative

> {narr["text"]}

---

## What does this look like in the library? (Slice C)

Top 3 nearest neighbors among the other 75 episodes:

| rank | episode_id | distance | dominant_phase | shot_like | ended_in_box |
|---|---|---|---|---|---|
{neighbor_rows}

| metric | value |
|---|---|
| p(shot_like) | {pred["p_shot_like"]:.2f} |
| p(ended_in_box) | {pred["p_ended_in_box"]:.2f} |
| p(reached_final_third) | {pred["p_reached_final_third"]:.2f} |
| max_distance | {pred["max_distance"]:.3f} |

Note the asymmetry: this episode WAS shot_like, but its neighbors were not. The retrieval is telling you *this episode is unusual within its cluster* — a high-yield outcome from a state that usually doesn't yield one. Which is the whole point: the player's movement was the differentiator.

### Pattern membership

K-means surfaced **{len(bundle["all_clusters"])} recurring patterns**:

| cluster | n | label |
|---|---|---|
{cluster_rows}

{pattern_block}

---

## What this answers, end to end

- **When did it start?** t={sel["boundary"]["start_time_s"]:.2f}s (episode boundary).
- **When did it start to *work*?** The decisive moment{f" at t+{dec['rel_time_s']:.2f}s" if dec else " — OBSO never built meaningfully"}.
- **Who took the shot?** {f"`{strike['shot_taker']}` at t+{strike['rel_time_s']:.2f}s, {strike['ball_speed_at_strike']:.1f} m/s." if strike else "Could not detect a strike."}
- **What made it successful?** The single-player marginal: `{narr["top_contributors"][0]["player_id"]}` with Δ OBSO {narr["top_contributors"][0]["contribution"]:+.3f} at peak — the rest of the team was at their average positional envelope; this player exceeded it.
- **Why was the outcome unusual?** Retrieval found 3 neighbors (similar starting state, similar shape) and *none* were shot_like. The position similarity alone wouldn't have predicted a shot. The differentiator was movement, captured by the contribution.

> Note: shot-taker detection is heuristic on tracking-only data — frame of first ball-speed crossing 12 m/s, then the closest attacker. Without event-stream ground truth this is the best we can do; the yellow ring shows you the engine's best guess so you can sanity-check it against the GIF.

Open `play.gif` to watch the full {sel["boundary"]["duration_s"]:.2f}s end to end, then come back to `movement_traces.png` to see the same movement frozen as paths.
"""


if __name__ == "__main__":
    main()
