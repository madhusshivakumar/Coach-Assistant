"""Tests for baseline + z-score + strengths/weaknesses classification."""

from __future__ import annotations

import pytest

from football_analysis.analytics.formations.profile import FormationProfile
from football_analysis.analytics.formations.strengths import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    NEUTRAL,
    Baseline,
    adjusted_z,
    classify_strengths_weaknesses,
    compute_baseline,
    score_profile,
)


def _profile(team_id: str, **metrics: float) -> FormationProfile:
    return FormationProfile(
        team_id=team_id,
        phase=None,
        sample="events",
        metrics=metrics,
        n_frames=1,
    )


def test_compute_baseline_produces_per_metric_mean_and_std() -> None:
    p1 = _profile("A", xt_generated=1.0, ppda=10.0)
    p2 = _profile("B", xt_generated=3.0, ppda=20.0)
    b = compute_baseline([p1, p2])
    assert b.n_samples == 2
    assert b.means["xt_generated"] == pytest.approx(2.0)
    assert b.means["ppda"] == pytest.approx(15.0)
    # stdev with n=2: sqrt(2)
    assert b.stds["xt_generated"] == pytest.approx(2**0.5)


def test_compute_baseline_with_single_profile_has_zero_std() -> None:
    p = _profile("A", xt_generated=2.0)
    b = compute_baseline([p])
    assert b.means["xt_generated"] == 2.0
    assert b.stds["xt_generated"] == 0.0


def test_compute_baseline_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one profile"):
        compute_baseline([])


def test_compute_baseline_handles_disjoint_metric_keys() -> None:
    p1 = _profile("A", xt_generated=1.0, ppda=10.0)
    p2 = _profile("B", xt_generated=3.0, field_tilt=0.6)
    b = compute_baseline([p1, p2])
    # xt_generated is in both
    assert b.means["xt_generated"] == 2.0
    # ppda is only in p1, std should be 0
    assert b.means["ppda"] == 10.0
    assert b.stds["ppda"] == 0.0
    # field_tilt only in p2
    assert b.means["field_tilt"] == 0.6


def test_score_profile_returns_zero_when_baseline_std_is_zero() -> None:
    b = Baseline(means={"x": 5.0}, stds={"x": 0.0}, n_samples=1)
    p = _profile("A", x=10.0)
    z = score_profile(p, b)
    assert z["x"] == 0.0


def test_score_profile_z_score_is_correct() -> None:
    b = Baseline(means={"x": 5.0}, stds={"x": 2.0}, n_samples=10)
    p = _profile("A", x=9.0)
    assert score_profile(p, b)["x"] == pytest.approx(2.0)


def test_score_profile_skips_metrics_not_in_baseline() -> None:
    b = Baseline(means={"x": 5.0}, stds={"x": 2.0}, n_samples=10)
    p = _profile("A", x=7.0, y=99.0)
    z = score_profile(p, b)
    assert "x" in z
    assert "y" not in z


def test_adjusted_z_flips_sign_for_lower_is_better() -> None:
    assert adjusted_z("ppda", -1.0) == 1.0  # low PPDA = strength
    assert adjusted_z("xt_generated", 1.5) == 1.5  # high xT = strength
    assert adjusted_z("length_mean", 1.0) == 1.0  # neutral metric: pass through


def test_classify_strengths_weaknesses_with_threshold() -> None:
    z = {
        "xt_generated": 1.5,
        "ppda": -1.0,  # low PPDA → strength
        "field_tilt": -0.6,
        "length_mean": 5.0,  # neutral metric never marked strong/weak
        "shots": 0.2,  # below threshold → neutral
    }
    cls = classify_strengths_weaknesses(z, threshold_z=0.5)
    assert cls.strengths == sorted(["xt_generated", "ppda"])
    assert cls.weaknesses == ["field_tilt"]
    assert cls.neutral == sorted(["length_mean", "shots"])


def test_metric_groups_partition_known_keys() -> None:
    """HIGHER/LOWER/NEUTRAL must be disjoint."""
    assert set() == HIGHER_IS_BETTER & LOWER_IS_BETTER
    assert set() == HIGHER_IS_BETTER & NEUTRAL
    assert set() == LOWER_IS_BETTER & NEUTRAL
