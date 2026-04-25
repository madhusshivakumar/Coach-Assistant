"""Polar radar plot for ``FormationProfile`` slices.

Each axis is one metric. Radial position is the sign-adjusted z-score against a
baseline (so outward always means "strength"). Multiple profiles drawn as overlapping
polygons make team-vs-team or phase-vs-phase comparison immediate.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from football_analysis.analytics.formations.profile import FormationProfile
from football_analysis.analytics.formations.strengths import (
    LOWER_IS_BETTER,
    Baseline,
    score_profile,
)

# Default colour rotation for overlapping polygons.
_PALETTE: tuple[str, ...] = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#8e44ad")

# Friendly axis labels with arrow hints showing optimisation direction.
_LABEL_ALIASES: dict[str, str] = {
    "xt_generated": "xT generated ↑",
    "xt_conceded": "xT conceded ↓",
    "ppda": "PPDA ↓",
    "field_tilt": "field tilt ↑",
    "shots": "shots ↑",
    "goals": "goals ↑",
    "length_mean": "length",
    "width_mean": "width",
    "hull_area_mean": "hull area ↑",
    "def_line_mean": "def. line",
    "off_line_mean": "off. line ↑",
    "vertical_compactness_mean": "vert. compact. ↓",
}


def _adjusted_value(metric: str, raw_z: float, clip: float) -> float:
    z = -raw_z if metric in LOWER_IS_BETTER else raw_z
    return max(-clip, min(clip, z))


def plot_formation_radar(
    profiles: Sequence[FormationProfile],
    baseline: Baseline,
    title: str | None = None,
    metric_order: Sequence[str] | None = None,
    z_clip: float = 2.0,
    label_fn: object = None,
) -> Figure:
    """Render a polar radar with one polygon per profile.

    Args:
        profiles: 1–N profiles to overlay.
        baseline: produced by ``compute_baseline``; defines the comparison population.
        title: figure title (e.g. ``"Tracking — build_up phase"``).
        metric_order: explicit axis order; defaults to sorted(baseline.means).
        z_clip: clip adjusted z-scores to ±this for plot legibility.
        label_fn: callable ``FormationProfile -> str`` for legend labels; defaults to
            ``"{team_id}"`` plus phase if present.
    """
    if not profiles:
        raise ValueError("plot_formation_radar needs at least one profile")
    if metric_order is None:
        metric_order = sorted(baseline.means.keys())
    if not metric_order:
        raise ValueError("metric_order is empty — baseline has no metrics")

    angles = np.linspace(0.0, 2.0 * np.pi, len(metric_order), endpoint=False).tolist()
    closed_angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(7.6, 7.4), subplot_kw={"projection": "polar"})

    def default_label(p: FormationProfile) -> str:
        return f"{p.team_id}" + (f"  ({p.phase})" if p.phase else "")

    label_callable = label_fn if callable(label_fn) else default_label

    for i, profile in enumerate(profiles):
        z_scores = score_profile(profile, baseline)
        values = [_adjusted_value(m, z_scores.get(m, 0.0), z_clip) for m in metric_order]
        values_closed = values + values[:1]
        color = _PALETTE[i % len(_PALETTE)]
        ax.plot(closed_angles, values_closed, color=color, linewidth=1.8, label=label_callable(profile))
        ax.fill(closed_angles, values_closed, color=color, alpha=0.18)

    axis_labels = [_LABEL_ALIASES.get(m, m) for m in metric_order]
    ax.set_xticks(angles)
    ax.set_xticklabels(axis_labels, fontsize=8)
    ax.set_yticks([-z_clip, 0.0, z_clip])
    ax.set_yticklabels([f"−{z_clip:.0f}σ", "avg", f"+{z_clip:.0f}σ"], fontsize=7)
    ax.set_ylim(-z_clip - 0.15, z_clip + 0.15)
    ax.grid(True, alpha=0.4)
    ax.set_theta_zero_location("N")  # type: ignore[attr-defined]
    ax.set_theta_direction(-1)  # type: ignore[attr-defined]

    if title:
        ax.set_title(title, fontsize=11, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.05), fontsize=9, frameon=False)
    fig.tight_layout()
    return fig
