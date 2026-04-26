"""Benchmark numpy vs torch (CPU) vs torch (GPU) OBSO compute paths.

Run: ``python scripts/bench_obso.py`` — produces a small text report showing
wall-clock per-frame and total batch time at default grid resolution.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from football_analysis.analytics.pitch_control.obso import compute_obso_frame
from football_analysis.analytics.pitch_control.obso_gpu import (
    compute_obso_batch_torch,
)

ROWS, COLS = 68, 104  # default OBSO resolution


def _make_synthetic_frames(
    n_frames: int, n_per_team: int = 11
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
    list[pd.DataFrame],
]:
    """Same data in two formats: per-frame DataFrames (for numpy path) AND the
    list-of-arrays format the torch batched path consumes.
    """
    rng = np.random.default_rng(42)
    home_pos: list[np.ndarray] = []
    home_vel: list[np.ndarray] = []
    away_pos: list[np.ndarray] = []
    away_vel: list[np.ndarray] = []
    ball_xy = np.zeros((n_frames, 2), dtype=np.float64)
    dfs: list[pd.DataFrame] = []
    for f in range(n_frames):
        hp = rng.uniform([0, 0], [105, 68], size=(n_per_team, 2))
        hv = rng.uniform(-2, 2, size=(n_per_team, 2))
        ap = rng.uniform([0, 0], [105, 68], size=(n_per_team, 2))
        av = rng.uniform(-2, 2, size=(n_per_team, 2))
        bx, by = float(rng.uniform(20, 85)), float(rng.uniform(20, 48))
        home_pos.append(hp)
        home_vel.append(hv)
        away_pos.append(ap)
        away_vel.append(av)
        ball_xy[f] = (bx, by)

        rows = []
        for i in range(n_per_team):
            rows.append(
                {
                    "frame_id": f,
                    "period": 1,
                    "time_seconds": f * 0.04,
                    "player_id": f"h{i}",
                    "team_id": "home",
                    "x": hp[i, 0],
                    "y": hp[i, 1],
                    "vx": hv[i, 0],
                    "vy": hv[i, 1],
                    "is_ball": False,
                    "visible": True,
                }
            )
            rows.append(
                {
                    "frame_id": f,
                    "period": 1,
                    "time_seconds": f * 0.04,
                    "player_id": f"a{i}",
                    "team_id": "away",
                    "x": ap[i, 0],
                    "y": ap[i, 1],
                    "vx": av[i, 0],
                    "vy": av[i, 1],
                    "is_ball": False,
                    "visible": True,
                }
            )
        rows.append(
            {
                "frame_id": f,
                "period": 1,
                "time_seconds": f * 0.04,
                "player_id": "ball",
                "team_id": "home",
                "x": bx,
                "y": by,
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": True,
                "visible": True,
            }
        )
        dfs.append(pd.DataFrame(rows))
    return home_pos, home_vel, away_pos, away_vel, ball_xy, dfs


def main() -> None:
    print("OBSO benchmark @ rows=68, cols=104 (full default resolution)")
    print(f"GPU available: {torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()

    for n_frames in (10, 100, 1000):
        hp, hv, ap, av, ball, dfs = _make_synthetic_frames(n_frames)
        atk_is_home = np.ones(n_frames, dtype=np.bool_)

        # numpy path
        t0 = time.perf_counter()
        for f, df in enumerate(dfs):
            _ = compute_obso_frame(
                df, frame_id=f, attacking_team_id="home", defending_team_id="away", rows=ROWS, cols=COLS
            )
        np_total = time.perf_counter() - t0

        # torch CPU path
        t0 = time.perf_counter()
        _ = compute_obso_batch_torch(
            hp,
            hv,
            ap,
            av,
            ball,
            atk_is_home,
            rows=ROWS,
            cols=COLS,
            device=torch.device("cpu"),
        )
        torch_cpu_total = time.perf_counter() - t0

        # torch GPU path (warm-up first call to amortise CUDA context init)
        result_gpu_total = float("nan")
        if torch.cuda.is_available():
            # Warm-up
            _ = compute_obso_batch_torch(
                hp[:1],
                hv[:1],
                ap[:1],
                av[:1],
                ball[:1],
                atk_is_home[:1],
                rows=ROWS,
                cols=COLS,
                device=torch.device("cuda"),
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = compute_obso_batch_torch(
                hp,
                hv,
                ap,
                av,
                ball,
                atk_is_home,
                rows=ROWS,
                cols=COLS,
                device=torch.device("cuda"),
            )
            torch.cuda.synchronize()
            result_gpu_total = time.perf_counter() - t0

        print(
            f"\nN={n_frames:5d} frames: "
            f"numpy={np_total * 1000:7.1f}ms "
            f"({np_total * 1000 / n_frames:.2f}ms/frame), "
            f"torch_cpu={torch_cpu_total * 1000:7.1f}ms "
            f"({torch_cpu_total * 1000 / n_frames:.2f}ms/frame), "
            f"torch_gpu={result_gpu_total * 1000:7.1f}ms "
            f"({result_gpu_total * 1000 / n_frames:.2f}ms/frame)"
        )
        if torch.cuda.is_available():
            print(f"  speedup vs numpy:  cpu={np_total / torch_cpu_total:.1f}x, gpu={np_total / result_gpu_total:.1f}x")


if __name__ == "__main__":
    main()
