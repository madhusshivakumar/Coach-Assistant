"""Soft-DTW similarity for role-anchored episode trajectories.

Soft-DTW (Cuturi & Blondel, ICML 2017) replaces the hard ``min`` in the DTW
recurrence with a smoothed log-sum-exp ``softmin`` parameterised by ``γ > 0``.
This gives a continuous, sub-differentiable distance between two time series
that handles unequal-length sequences without hand-aligning them.

In this codebase the inputs are role-anchored player trajectories produced by
``episode_role_trajectory`` — for each episode we have ``(T_snapshots, 10
roles, 2 coords)``. Soft-DTW collapses the ``10 × 2`` per-snapshot vector into
a 20-dim feature, computes the pairwise squared-Euclidean cost between the
two episodes' snapshot sequences, and runs the soft-DTW recurrence on that
cost matrix. The output is a scalar distance suitable for HDBSCAN clustering
(M2) and pairwise retrieval.

Public API:
    soft_dtw(X, Y, gamma)                  — bare 2-D arrays
    soft_dtw_from_tensor(A, B, gamma)      — (T, n_roles, 2) tensors
    soft_dtw_from_trajectory(df_a, df_b)   — long-form role_trajectory frames

Implementation note: the recurrence is O(T₁·T₂) per call. Episode snapshot
counts are typically ≤ 30 at 2 Hz, so a pure-numpy double loop is fine even
when materialising a full pairwise distance matrix for ~10⁴ episodes. If
profiling later shows this is a bottleneck we can move to a numba ``@njit``
or batched cython kernel without changing the public API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A natural lower bound on γ — below this the log-sum-exp trick still works,
# but the distance is essentially hard DTW and the smoothness benefit is gone.
_GAMMA_MIN = 1e-6


def _softmin(a: float, b: float, c: float, gamma: float) -> float:
    """Smoothed minimum of three values.

    ``softmin_γ(a, b, c) = -γ · log(exp(-a/γ) + exp(-b/γ) + exp(-c/γ))``.
    Implemented with the log-sum-exp trick (subtract the running min before
    exponentiating) so we don't underflow when γ is small.
    """
    m = min(a, b, c)
    s = np.exp(-(a - m) / gamma) + np.exp(-(b - m) / gamma) + np.exp(-(c - m) / gamma)
    return float(-gamma * np.log(s) + m)


def soft_dtw(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """Compute soft-DTW distance between two 2-D time series.

    Args:
        X: shape ``(T1, D)`` — first sequence.
        Y: shape ``(T2, D)`` — second sequence with the same feature dim.
        gamma: smoothing parameter ``> 0``. Smaller γ → tighter smoothing
            (closer to hard DTW). Default 1.0 follows Cuturi & Blondel's
            recommendation for L2-cost data on the order of metres.

    Returns:
        Scalar soft-DTW value. Always ≥ 0; equals 0 iff ``X == Y`` and γ is
        small enough that the diagonal alignment dominates.

    Raises:
        ValueError: arrays not 2-D, mismatched feature dims, or γ ≤ 0.
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"soft_dtw expects 2-D arrays, got X.ndim={X.ndim}, Y.ndim={Y.ndim}")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"soft_dtw feature dim mismatch: X has {X.shape[1]}, Y has {Y.shape[1]}")
    if gamma < _GAMMA_MIN:
        raise ValueError(f"gamma must be > {_GAMMA_MIN} (got {gamma})")

    T1, _ = X.shape
    T2, _ = Y.shape

    # Cost matrix: pairwise squared-Euclidean.
    diff = X[:, None, :] - Y[None, :, :]
    cost = np.sum(diff * diff, axis=2)  # (T1, T2)

    # Sentinel-padded R matrix; R[0,0]=0 and the +inf borders block invalid moves.
    R = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float64)
    R[0, 0] = 0.0
    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            R[i, j] = cost[i - 1, j - 1] + _softmin(R[i - 1, j - 1], R[i - 1, j], R[i, j - 1], gamma)
    return float(R[T1, T2])


def soft_dtw_from_tensor(A: np.ndarray, B: np.ndarray, gamma: float = 1.0) -> float:
    """Soft-DTW between two role-anchored trajectory tensors.

    Args:
        A, B: shape ``(T, n_roles, 2)``. The two trajectories must share
            ``n_roles`` (M2 always pairs episodes built with the same template).
        gamma: smoothing parameter, forwarded to ``soft_dtw``.

    Returns:
        Scalar soft-DTW value on the flattened ``(T, n_roles · 2)`` view.

    Raises:
        ValueError: tensors not 3-D or role counts disagree.
    """
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError(f"soft_dtw_from_tensor expects 3-D arrays, got A.ndim={A.ndim}, B.ndim={B.ndim}")
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"role count mismatch: A has {A.shape[1]} roles, B has {B.shape[1]}")
    if A.shape[2] != 2 or B.shape[2] != 2:
        raise ValueError(f"trailing dim must be 2 (x,y); got A={A.shape[2]}, B={B.shape[2]}")

    T_a, n_roles, _ = A.shape
    T_b, _, _ = B.shape
    flat_a = A.reshape(T_a, n_roles * 2)
    flat_b = B.reshape(T_b, n_roles * 2)
    return soft_dtw(flat_a, flat_b, gamma=gamma)


def _trajectory_to_tensor(traj: pd.DataFrame) -> np.ndarray:
    """Pivot a role_trajectory long-form frame into ``(T, n_roles, 2)``.

    Sorts by ``(snapshot_idx, role)`` so the returned tensor is
    permutation-invariant — two trajectories with the same content but
    different row order produce identical tensors and hence identical
    soft-DTW values.
    """
    if traj.empty:
        raise ValueError("soft_dtw_from_trajectory got an empty role_trajectory frame")
    # Stable role ordering: alphabetical by role name. Both trajectories use the
    # same template (M2 contract), so this gives a consistent slot order.
    pivoted = (
        traj.sort_values(["snapshot_idx", "role"])
        .set_index(["snapshot_idx", "role"])[["x", "y"]]
    )
    snapshots = sorted(pivoted.index.get_level_values("snapshot_idx").unique())
    roles = sorted(pivoted.index.get_level_values("role").unique())
    arr = np.zeros((len(snapshots), len(roles), 2), dtype=np.float64)
    for i, snap in enumerate(snapshots):
        for j, role in enumerate(roles):
            try:
                row = pivoted.loc[(snap, role)]
            except KeyError:  # role missing at this snapshot — shouldn't happen for clean episodes
                continue
            arr[i, j, 0] = float(row["x"])
            arr[i, j, 1] = float(row["y"])
    return arr


def soft_dtw_from_trajectory(
    traj_a: pd.DataFrame,
    traj_b: pd.DataFrame,
    gamma: float = 1.0,
) -> float:
    """Soft-DTW directly from two ``episode_role_trajectory`` outputs.

    Args:
        traj_a, traj_b: long-form DataFrames as returned by
            ``episode_role_trajectory``. Must have non-empty content; both
            should share the same template's role set (caller's contract).
        gamma: smoothing parameter forwarded to soft-DTW.

    Returns:
        Scalar soft-DTW distance.
    """
    A = _trajectory_to_tensor(traj_a)
    B = _trajectory_to_tensor(traj_b)
    return soft_dtw_from_tensor(A, B, gamma=gamma)
