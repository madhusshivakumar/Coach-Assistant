"""Expected Threat (xT) — Karun Singh's 16x12 Markov value-of-possession model.

Reference: https://karun.in/blog/expected-threat.html

Discretise the pitch to a rows x cols grid (default 12 x 16). For each cell compute:
    s(c) = P(shoot | possession in c)
    m(c) = P(move  | possession in c)  = 1 - s(c)   (carry-or-pass, not turnover)
    g(c) = P(goal  | shot from c)
    T(c -> c') = P(move lands in c' | moving from c)

Then iterate the Bellman-style fixed point until converged:

    xT(c) = s(c) * g(c) + m(c) * sum_{c'} T(c -> c') * xT(c')

Each pass or carry is valued by `xt_delta = xT(end_cell) - xT(start_cell)`.

The implementation is deliberately free of external ML dependencies (no sklearn, no socceraction)
so it fits cleanly into the POC's lightweight analytics layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_analysis.analytics.pitch import PITCH_LENGTH_M, PITCH_WIDTH_M


@dataclass(frozen=True)
class XTGrid:
    """A fitted xT surface plus the underlying probabilities."""

    rows: int  # along pitch width (y axis)
    cols: int  # along pitch length (x axis)
    values: np.ndarray  # shape (rows, cols), float64
    shoot_prob: np.ndarray  # shape (rows, cols)
    goal_prob: np.ndarray  # shape (rows, cols)
    move_prob: np.ndarray  # shape (rows, cols)

    def xt_at(self, x: float, y: float) -> float:
        """Value of possessing the ball at the given canonical metric coordinate."""
        r, c = _cell_index(x, y, self.rows, self.cols)
        return float(self.values[r, c])


def _cell_index(x: float, y: float, rows: int, cols: int) -> tuple[int, int]:
    """Map a canonical-pitch (x, y) to a (row, col) cell index."""
    col = min(cols - 1, max(0, int(x / PITCH_LENGTH_M * cols)))
    row = min(rows - 1, max(0, int(y / PITCH_WIDTH_M * rows)))
    return row, col


def fit(events: pd.DataFrame, rows: int = 12, cols: int = 16, max_iter: int = 100, tol: float = 1e-6) -> XTGrid:
    """Fit an xT grid from SPADL-extended events.

    Events must contain columns: action_type ('pass' / 'shot' / 'dribble'), start_x, start_y,
    end_x, end_y, result ('success'/'fail'/...). Rows with missing coordinates are ignored.
    """
    if not {"action_type", "start_x", "start_y", "result"}.issubset(events.columns):
        raise ValueError("events must contain action_type, start_x, start_y, result columns")

    shoot_count = np.zeros((rows, cols), dtype=np.float64)
    goal_count = np.zeros((rows, cols), dtype=np.float64)
    move_count = np.zeros((rows, cols), dtype=np.float64)
    move_to = np.zeros((rows, cols, rows, cols), dtype=np.float64)

    for _, ev in events.iterrows():
        sx, sy = ev.get("start_x"), ev.get("start_y")
        if sx is None or sy is None or pd.isna(sx) or pd.isna(sy):
            continue
        r, c = _cell_index(float(sx), float(sy), rows, cols)

        action = ev["action_type"]
        result = ev.get("result")

        if action == "shot":
            shoot_count[r, c] += 1
            if result == "success":
                goal_count[r, c] += 1
        elif action in {"pass", "dribble"} and result == "success":
            ex, ey = ev.get("end_x"), ev.get("end_y")
            if ex is None or ey is None or pd.isna(ex) or pd.isna(ey):
                continue
            r2, c2 = _cell_index(float(ex), float(ey), rows, cols)
            move_count[r, c] += 1
            move_to[r, c, r2, c2] += 1

    total = shoot_count + move_count
    with np.errstate(divide="ignore", invalid="ignore"):
        shoot_prob = np.where(total > 0, shoot_count / total, 0.0)
        move_prob = np.where(total > 0, move_count / total, 0.0)
        goal_prob = np.where(shoot_count > 0, goal_count / shoot_count, 0.0)
        # Normalise transition counts per-source-cell to a probability distribution
        move_totals = move_to.sum(axis=(2, 3), keepdims=True)
        transition = np.where(move_totals > 0, move_to / move_totals, 0.0)

    xt = np.zeros((rows, cols), dtype=np.float64)
    for _ in range(max_iter):
        expected_move_value = np.einsum("rcij,ij->rc", transition, xt)
        xt_new = shoot_prob * goal_prob + move_prob * expected_move_value
        if np.max(np.abs(xt_new - xt)) < tol:
            xt = xt_new
            break
        xt = xt_new

    return XTGrid(
        rows=rows, cols=cols, values=xt, shoot_prob=shoot_prob, goal_prob=goal_prob, move_prob=move_prob
    )


def apply_xt(events: pd.DataFrame, grid: XTGrid) -> pd.DataFrame:
    """Return `events` with two added columns: `xt` (value at start cell) and `xt_delta`.

    `xt` is the start-cell value (for all on-ball actions with start coords).
    `xt_delta` is end_cell - start_cell for successful passes/carries, else 0.
    Shots get xt_delta = NaN (their value is captured by xG, not xT).
    """
    out = events.copy()
    xt_vals = np.full(len(out), np.nan, dtype=np.float64)
    xt_delta = np.full(len(out), np.nan, dtype=np.float64)

    for i, (_, ev) in enumerate(out.iterrows()):
        sx, sy = ev.get("start_x"), ev.get("start_y")
        if sx is None or sy is None or pd.isna(sx) or pd.isna(sy):
            continue
        r, c = _cell_index(float(sx), float(sy), grid.rows, grid.cols)
        xt_vals[i] = grid.values[r, c]

        action = ev["action_type"]
        if action in {"pass", "dribble"} and ev.get("result") == "success":
            ex, ey = ev.get("end_x"), ev.get("end_y")
            if ex is None or ey is None or pd.isna(ex) or pd.isna(ey):
                continue
            r2, c2 = _cell_index(float(ex), float(ey), grid.rows, grid.cols)
            xt_delta[i] = grid.values[r2, c2] - grid.values[r, c]

    out["xt"] = xt_vals
    out["xt_delta"] = xt_delta
    return out
