"""Tests for the data-layer pitch rescale helper (list form)."""

from __future__ import annotations

from football_analysis.data.normalize.pitch import rescale_series


def test_rescale_series_roundtrip() -> None:
    xs = [0.0, 60.0, 120.0]
    ys = [0.0, 40.0, 80.0]
    out_x, out_y = rescale_series(xs, ys, 120.0, 80.0, source_origin="top_left")
    assert len(out_x) == 3
    # (60, 40) is centre -> (52.5, 34.0) on canonical
    assert abs(out_x[1] - 52.5) < 1e-9
    assert abs(out_y[1] - 34.0) < 1e-9


def test_rescale_series_empty() -> None:
    out_x, out_y = rescale_series([], [], 120.0, 80.0)
    assert out_x == []
    assert out_y == []
