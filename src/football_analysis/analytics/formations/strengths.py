"""Score ``FormationProfile`` slices against a baseline distribution.

Three steps, all pure functions over ``FormationProfile``:

1. ``compute_baseline(profiles)`` — per-metric mean + std across a population.
2. ``score_profile(profile, baseline)`` — z-score each metric vs baseline.
3. ``classify_strengths_weaknesses(z_scores)`` — sort metrics into strengths / weaknesses /
   neutral using a sign-aware threshold (``LOWER_IS_BETTER`` metrics flip sign so a
   positive adjusted z always reads "strength").
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass

from football_analysis.analytics.formations.profile import FormationProfile

# Metrics where larger raw values are better for the team being profiled.
HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {
        "xt_generated",
        "field_tilt",
        "shots",
        "goals",
        "off_line_mean",
        "hull_area_mean",
    }
)

# Metrics where smaller raw values are better.
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "ppda",
        "xt_conceded",
        "vertical_compactness_mean",
    }
)

# Metrics that are neither universally good nor bad (high or low both context-dependent).
NEUTRAL: frozenset[str] = frozenset(
    {
        "length_mean",
        "width_mean",
        "def_line_mean",
    }
)


@dataclass(frozen=True)
class Baseline:
    """Per-metric mean and std over a profile population."""

    means: dict[str, float]
    stds: dict[str, float]
    n_samples: int


@dataclass(frozen=True)
class StrengthClassification:
    """Three-way bucketing of metrics by sign-adjusted z-score."""

    strengths: list[str]
    weaknesses: list[str]
    neutral: list[str]


def compute_baseline(profiles: Iterable[FormationProfile]) -> Baseline:
    """Compute per-metric (mean, std) across the union of metrics seen in ``profiles``.

    Profiles with disjoint metric keys are tolerated — each metric's stats are computed
    over only the profiles that actually contain it.
    """
    profile_list = list(profiles)
    if not profile_list:
        raise ValueError("compute_baseline requires at least one profile")

    all_metric_keys: set[str] = set()
    for p in profile_list:
        all_metric_keys.update(p.metrics)

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for key in all_metric_keys:
        values = [p.metrics[key] for p in profile_list if key in p.metrics]
        if not values:
            continue
        means[key] = float(statistics.mean(values))
        stds[key] = float(statistics.stdev(values)) if len(values) >= 2 else 0.0

    return Baseline(means=means, stds=stds, n_samples=len(profile_list))


def score_profile(profile: FormationProfile, baseline: Baseline) -> dict[str, float]:
    """Per-metric z-score = ``(value - mean) / std``. Returns 0.0 when std is 0."""
    z: dict[str, float] = {}
    for metric, value in profile.metrics.items():
        if metric not in baseline.means:
            continue
        std = baseline.stds.get(metric, 0.0)
        z[metric] = (value - baseline.means[metric]) / std if std > 0.0 else 0.0
    return z


def adjusted_z(metric: str, raw_z: float) -> float:
    """Sign-adjust a raw z-score so positive always means "strength" for the metric.

    For ``LOWER_IS_BETTER`` metrics, flip sign. For ``NEUTRAL`` metrics, return raw.
    """
    if metric in LOWER_IS_BETTER:
        return -raw_z
    return raw_z


def classify_strengths_weaknesses(
    z_scores: dict[str, float],
    threshold_z: float = 0.5,
) -> StrengthClassification:
    """Bucket metrics by sign-adjusted z-score against ``threshold_z``.

    A metric goes to:
    - ``strengths`` if adjusted z >= threshold_z
    - ``weaknesses`` if adjusted z <= -threshold_z
    - ``neutral`` otherwise (including all NEUTRAL metrics regardless of magnitude,
      since high/low isn't intrinsically good for them)
    """
    strengths: list[str] = []
    weaknesses: list[str] = []
    neutral: list[str] = []
    for metric, raw_z in z_scores.items():
        if metric in NEUTRAL:
            neutral.append(metric)
            continue
        adj = adjusted_z(metric, raw_z)
        if adj >= threshold_z:
            strengths.append(metric)
        elif adj <= -threshold_z:
            weaknesses.append(metric)
        else:
            neutral.append(metric)
    return StrengthClassification(
        strengths=sorted(strengths),
        weaknesses=sorted(weaknesses),
        neutral=sorted(neutral),
    )
