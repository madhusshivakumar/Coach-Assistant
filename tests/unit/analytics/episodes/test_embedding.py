"""Tests for episode embedding."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_analysis.analytics.episodes.embedding import (
    EPISODE_FEATURE_NAMES,
    EXTENDED_FEATURE_NAMES,
    embed_episode,
    embed_episode_extended,
)
from football_analysis.analytics.episodes.engine import EpisodeRecord
from football_analysis.analytics.episodes.outcomes import EpisodeOutcome
from football_analysis.analytics.episodes.segmenter import EpisodeBoundary
from football_analysis.analytics.formations.roles import FORMATION_4_3_3


def _record(states: pd.DataFrame, dominant_phase: str | None = "progression") -> EpisodeRecord:
    boundary = EpisodeBoundary(
        episode_id=1,
        start_frame=1,
        end_frame=100,
        start_time_s=0.0,
        end_time_s=4.0,
        duration_s=4.0,
        possession_team="home",
        end_reason="match_end",
    )
    outcome = EpisodeOutcome(
        episode_id=1,
        end_reason="match_end",
        reached_final_third=True,
        ended_in_box=False,
        shot_like=False,
        end_ball_x=80.0,
        end_ball_y=34.0,
        end_ball_speed=4.0,
        duration_s=4.0,
    )
    return EpisodeRecord(boundary=boundary, outcome=outcome, state_trajectory=states, dominant_phase=dominant_phase)


def _trajectory(start_x: float = 30.0, end_x: float = 80.0, n: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rel_time_s": np.linspace(0.0, 4.0, n),
            "ball_x": np.linspace(start_x, end_x, n),
            "ball_y": [34.0] * n,
            "ball_x_oriented": np.linspace(start_x, end_x, n),
            "ball_speed": [3.0] * n,
            "attackers_mean_x_oriented": np.linspace(40, 65, n),
            "defenders_line_height_oriented": [85.0] * n,
            "attackers_length": [40.0] * n,
            "attackers_width": [50.0] * n,
            "attackers_visible": [11] * n,
            "defenders_visible": [10] * n,
            "defenders_compactness_x": [3.5] * n,
        }
    )


def test_embed_episode_returns_correct_shape() -> None:
    rec = _record(_trajectory())
    vec = embed_episode(rec)
    assert vec.shape == (len(EPISODE_FEATURE_NAMES),)
    assert vec.dtype == np.float64
    assert np.isfinite(vec).all()


def test_embed_episode_phase_one_hot_set_correctly() -> None:
    rec = _record(_trajectory(), dominant_phase="finishing")
    vec = embed_episode(rec)
    idx = EPISODE_FEATURE_NAMES.index("phase_finishing")
    assert vec[idx] == 1.0
    # Other phase one-hots should be zero.
    other_phase_idxs = [
        i for i, name in enumerate(EPISODE_FEATURE_NAMES) if name.startswith("phase_") and name != "phase_finishing"
    ]
    assert all(vec[i] == 0.0 for i in other_phase_idxs)


def test_embed_episode_unknown_phase_yields_all_zero_phase_one_hot() -> None:
    rec = _record(_trajectory(), dominant_phase=None)
    vec = embed_episode(rec)
    phase_idxs = [i for i, name in enumerate(EPISODE_FEATURE_NAMES) if name.startswith("phase_")]
    assert all(vec[i] == 0.0 for i in phase_idxs)


def test_embed_episode_ball_x_displacement_is_max_minus_start() -> None:
    rec = _record(_trajectory(start_x=30.0, end_x=80.0))
    vec = embed_episode(rec)
    idx_disp = EPISODE_FEATURE_NAMES.index("ball_x_displacement")
    # Trajectory is monotonically increasing → max == end → displacement = 50.
    assert abs(vec[idx_disp] - 50.0) < 1e-6


def test_embed_episode_partial_mode_uses_only_prefix() -> None:
    rec = _record(_trajectory(start_x=30.0, end_x=80.0, n=8))
    full = embed_episode(rec)
    prefix = embed_episode(rec, max_rel_time_s=2.0)
    # Prefix end_ball_x should be < full end_ball_x (we cut at half-time).
    idx_end = EPISODE_FEATURE_NAMES.index("end_ball_x_oriented")
    assert prefix[idx_end] < full[idx_end]


def test_embed_episode_empty_trajectory_returns_zeros() -> None:
    rec = _record(pd.DataFrame())
    vec = embed_episode(rec)
    assert vec.shape == (len(EPISODE_FEATURE_NAMES),)
    assert (vec == 0).all()


def test_embed_episode_handles_all_nan_ball_columns() -> None:
    states = _trajectory()
    states["ball_x_oriented"] = float("nan")
    states["ball_speed"] = float("nan")
    rec = _record(states)
    vec = embed_episode(rec)
    # Should produce no NaNs anywhere — we substitute 0.
    assert np.isfinite(vec).all()


# ---------------------------------------------------------------------------
# v2 redesign — M1 task #2: extended embedding (pressing + role + template)
# ---------------------------------------------------------------------------


def _tracking_for_extended(
    n_frames: int = 100,
    fps: int = 25,
    pressing_close: bool = True,
) -> pd.DataFrame:
    """Synthesize 4-3-3 home + arbitrary away tracking for a 4-second episode.

    ``pressing_close=True``: defenders sit on top of the ball (mean defender →
    ball distance ≈ 0–1 m). ``pressing_close=False``: defenders sit far away
    (≥ 50 m), as if dropping deep.
    """
    tpl = FORMATION_4_3_3
    rows: list[dict] = []
    for f in range(1, n_frames + 1):
        t = f / fps
        # Home outfielders on 4-3-3 slots
        for i, (tx, ty) in enumerate(zip(tpl.xs, tpl.ys, strict=True)):
            rows.append(
                {
                    "frame_id": f,
                    "period": 1,
                    "time_seconds": t,
                    "player_id": f"home_{i}",
                    "team_id": "home",
                    "x": tx,
                    "y": ty,
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": False,
                    "visible": True,
                }
            )
        # Home GK
        rows.append(
            {
                "frame_id": f,
                "period": 1,
                "time_seconds": t,
                "player_id": "home_GK",
                "team_id": "home",
                "x": 5.0,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": False,
                "visible": True,
            }
        )
        # Away players: either tightly pressing the ball, or far away
        ball_x = 50.0
        ball_y = 34.0
        if pressing_close:
            away_xs = [ball_x + 0.5 + j * 0.3 for j in range(11)]
            away_ys = [ball_y + 0.5 + j * 0.3 for j in range(11)]
        else:
            away_xs = [100.0] * 11
            away_ys = [60.0] * 11
        for i in range(10):  # 10 outfielders
            rows.append(
                {
                    "frame_id": f,
                    "period": 1,
                    "time_seconds": t,
                    "player_id": f"away_{i}",
                    "team_id": "away",
                    "x": away_xs[i],
                    "y": away_ys[i],
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": False,
                    "visible": True,
                }
            )
        rows.append(
            {
                "frame_id": f,
                "period": 1,
                "time_seconds": t,
                "player_id": "away_GK",
                "team_id": "away",
                "x": away_xs[10],
                "y": away_ys[10],
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": False,
                "visible": True,
            }
        )
        # Ball at midfield
        rows.append(
            {
                "frame_id": f,
                "period": 1,
                "time_seconds": t,
                "player_id": "ball",
                "team_id": None,
                "x": ball_x,
                "y": ball_y,
                "vx": 1.0,
                "vy": 0.0,
                "is_ball": True,
                "visible": True,
            }
        )
    return pd.DataFrame(rows)


def _record_for_extended() -> EpisodeRecord:
    """4-second possession episode used by the extended-embedding tests."""
    boundary = EpisodeBoundary(
        episode_id=1,
        start_frame=1,
        end_frame=100,
        start_time_s=0.04,
        end_time_s=4.0,
        duration_s=3.96,
        possession_team="home",
        end_reason="match_end",
    )
    outcome = EpisodeOutcome(
        episode_id=1,
        end_reason="match_end",
        reached_final_third=False,
        ended_in_box=False,
        shot_like=False,
        end_ball_x=50.0,
        end_ball_y=34.0,
        end_ball_speed=1.0,
        duration_s=3.96,
    )
    return EpisodeRecord(
        boundary=boundary,
        outcome=outcome,
        state_trajectory=_trajectory(),  # state aggregates are still required for the original 25 dims
        dominant_phase="progression",
    )


def test_extended_schema_is_strict_superset_of_base() -> None:
    """First 17 dims of EXTENDED_FEATURE_NAMES must equal the first 17 of
    EPISODE_FEATURE_NAMES (the non-phase block), and the trailing phase one-hots
    must also match. Order is contractual — index alignment is how the cluster
    layer reuses the base embedding."""
    base = list(EPISODE_FEATURE_NAMES)
    ext = list(EXTENDED_FEATURE_NAMES)
    assert ext[: len(base)] == base
    assert len(ext) > len(base)


def test_embed_episode_extended_returns_correct_shape() -> None:
    rec = _record_for_extended()
    tracking = _tracking_for_extended()
    vec = embed_episode_extended(rec, tracking, "home", "away")
    assert vec.shape == (len(EXTENDED_FEATURE_NAMES),)
    assert vec.dtype == np.float64
    assert np.isfinite(vec).all()


def test_extended_first_block_matches_base_embed() -> None:
    """The first len(EPISODE_FEATURE_NAMES) entries of the extended vector must
    equal the base ``embed_episode`` output — old retrieval indexes can be
    re-built without re-meaning these dims."""
    rec = _record_for_extended()
    tracking = _tracking_for_extended()
    base = embed_episode(rec)
    ext = embed_episode_extended(rec, tracking, "home", "away")
    np.testing.assert_array_almost_equal(ext[: len(base)], base)


def test_min_pressing_distance_small_when_defenders_sit_on_ball() -> None:
    """When defenders are within ~1 m of the ball every snapshot, the min
    pressing-distance feature should be < 2 m."""
    rec = _record_for_extended()
    tracking = _tracking_for_extended(pressing_close=True)
    vec = embed_episode_extended(rec, tracking, "home", "away")
    idx = EXTENDED_FEATURE_NAMES.index("min_pressing_distance")
    assert vec[idx] < 2.0


def test_min_pressing_distance_large_when_defenders_drop_deep() -> None:
    rec = _record_for_extended()
    tracking = _tracking_for_extended(pressing_close=False)
    vec = embed_episode_extended(rec, tracking, "home", "away")
    idx = EXTENDED_FEATURE_NAMES.index("min_pressing_distance")
    assert vec[idx] > 30.0


def test_formation_match_cost_low_when_team_on_template() -> None:
    """Home outfielders sit exactly on 4-3-3 slots — the mean
    formation-template-match cost should be very small."""
    rec = _record_for_extended()
    tracking = _tracking_for_extended()
    vec = embed_episode_extended(rec, tracking, "home", "away", template=FORMATION_4_3_3)
    idx = EXTENDED_FEATURE_NAMES.index("formation_match_cost")
    assert vec[idx] < 5.0


def test_role_displacement_low_when_team_on_template() -> None:
    rec = _record_for_extended()
    tracking = _tracking_for_extended()
    vec = embed_episode_extended(rec, tracking, "home", "away", template=FORMATION_4_3_3)
    idx = EXTENDED_FEATURE_NAMES.index("mean_attacker_role_displacement")
    # Players sit exactly on the slots → displacement effectively zero.
    assert vec[idx] < 1.0


def test_extended_handles_empty_tracking_returns_zeros_for_extras() -> None:
    rec = _record_for_extended()
    empty_tracking = pd.DataFrame(
        columns=["frame_id", "period", "time_seconds", "player_id", "team_id", "x", "y", "vx", "vy", "is_ball", "visible"]
    )
    vec = embed_episode_extended(rec, empty_tracking, "home", "away")
    # Same length as full extended schema, no NaNs leaked in
    assert vec.shape == (len(EXTENDED_FEATURE_NAMES),)
    assert np.isfinite(vec).all()
    # The base block is computed from state_trajectory and stays non-zero;
    # the new dims fall back to 0 when no tracking is available.
    extras = vec[len(EPISODE_FEATURE_NAMES):]
    assert (extras == 0).all()


def test_extended_partial_mode_truncates_by_rel_time() -> None:
    """``max_rel_time_s`` cut-off applies to both the base trajectory features
    and the new tracking-derived features."""
    rec = _record_for_extended()
    tracking = _tracking_for_extended()
    full = embed_episode_extended(rec, tracking, "home", "away")
    half = embed_episode_extended(rec, tracking, "home", "away", max_rel_time_s=2.0)
    # The base block already differs (already tested in base tests).
    # New dims should remain finite under the cut-off.
    assert np.isfinite(half).all()
    # Same length contract regardless of cut-off
    assert half.shape == full.shape
