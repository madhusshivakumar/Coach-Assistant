"""Pitch-coordinate rescaling at the data boundary.

kloppy hands us events with `Point` start/end coordinates in its `PitchDimensions`.
We always convert to the canonical 105x68 metric pitch (see football_analysis.analytics.pitch).
"""

from __future__ import annotations

from collections.abc import Iterable

from football_analysis.analytics.pitch import Pitch, rescale_point


def rescale_series(
    xs: Iterable[float],
    ys: Iterable[float],
    source_length: float,
    source_width: float,
    source_origin: str = "bottom_left",
    target: Pitch | None = None,
) -> tuple[list[float], list[float]]:
    """Vectorised-by-loop rescale. Numpy-friendly wrapper lives in events_spadl."""
    out_x: list[float] = []
    out_y: list[float] = []
    for x, y in zip(xs, ys, strict=True):
        nx, ny = rescale_point(x, y, source_length, source_width, source_origin, target)
        out_x.append(nx)
        out_y.append(ny)
    return out_x, out_y
