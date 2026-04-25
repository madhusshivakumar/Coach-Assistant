"""Smoke tests for the polar formation radar."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

from football_analysis.analytics.formations.profile import FormationProfile
from football_analysis.analytics.formations.strengths import compute_baseline
from football_analysis.viz.static.formation_radar import plot_formation_radar


@pytest.fixture(autouse=True)
def close_figs():
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


def _profile(team: str, **metrics: float) -> FormationProfile:
    return FormationProfile(team_id=team, phase=None, sample="events", metrics=metrics, n_frames=1)


def test_radar_returns_figure_and_writes_png(tmp_path: Path) -> None:
    p1 = _profile("home", xt_generated=1.5, ppda=8.0, field_tilt=0.55)
    p2 = _profile("away", xt_generated=0.8, ppda=12.0, field_tilt=0.45)
    baseline = compute_baseline([p1, p2])
    fig = plot_formation_radar([p1, p2], baseline, title="test radar")
    assert isinstance(fig, Figure)
    out = tmp_path / "radar.png"
    fig.savefig(out)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_radar_rejects_empty_profile_list() -> None:
    baseline = compute_baseline([_profile("a", x=1.0)])
    with pytest.raises(ValueError, match="at least one"):
        plot_formation_radar([], baseline)


def test_radar_uses_phase_in_label_when_present(tmp_path: Path) -> None:
    p = FormationProfile(
        team_id="home",
        phase="build_up",
        sample="tracking",
        metrics={"length_mean": 30.0, "width_mean": 40.0},
        n_frames=10,
    )
    baseline = compute_baseline([p])
    fig = plot_formation_radar([p], baseline)
    legend = fig.axes[0].get_legend()
    assert legend is not None
    label_text = legend.get_texts()[0].get_text()
    assert "home" in label_text
    assert "build_up" in label_text


def test_radar_respects_explicit_metric_order(tmp_path: Path) -> None:
    p1 = _profile("a", xt_generated=1.0, ppda=10.0, field_tilt=0.4)
    p2 = _profile("b", xt_generated=2.0, ppda=15.0, field_tilt=0.6)
    baseline = compute_baseline([p1, p2])
    metric_order = ("ppda", "xt_generated", "field_tilt")
    fig = plot_formation_radar([p1, p2], baseline, metric_order=metric_order)
    # X-tick labels should match metric_order length
    xticklabels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert len(xticklabels) == len(metric_order)


def test_radar_clips_extreme_z_scores() -> None:
    """An outlier profile (~10σ) should still render — values are clipped to ±z_clip."""
    p_normal = _profile("normal", xt_generated=1.0)
    p_outlier = _profile("outlier", xt_generated=100.0)
    baseline = compute_baseline([p_normal, p_outlier])
    # No assertion failure means clipping handled the extreme value.
    fig = plot_formation_radar([p_outlier], baseline, z_clip=2.0)
    assert isinstance(fig, Figure)


def test_radar_rejects_baseline_with_no_metrics() -> None:
    from football_analysis.analytics.formations.strengths import Baseline

    p = _profile("a", x=1.0)
    empty_baseline = Baseline(means={}, stds={}, n_samples=0)
    with pytest.raises(ValueError, match="metric_order"):
        plot_formation_radar([p], empty_baseline)
