"""Canonical pitch geometry.

The canonical coordinate system for the project:

- Metric units (metres).
- Pitch 105.0 x 68.0 m.
- Origin at bottom-left corner (0, 0); top-right corner at (105, 68).
- Home team attacks left-to-right in the first half, flipped at halftime.

All downstream code assumes this. Ingest is the only layer that negotiates
provider-specific coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

PITCH_LENGTH_M: float = 105.0
PITCH_WIDTH_M: float = 68.0


@dataclass(frozen=True)
class Pitch:
    """Immutable pitch dimensions. Kept as an object so alternate sizes can be threaded through later."""

    length: float = PITCH_LENGTH_M
    width: float = PITCH_WIDTH_M

    @property
    def center(self) -> tuple[float, float]:
        return (self.length / 2.0, self.width / 2.0)

    def in_bounds(self, x: float, y: float) -> bool:
        """Closed interval — the pitch corners are in bounds."""
        return 0.0 <= x <= self.length and 0.0 <= y <= self.width


def rescale_point(
    x: float,
    y: float,
    source_length: float,
    source_width: float,
    source_origin: str = "bottom_left",
    target: Pitch | None = None,
) -> tuple[float, float]:
    """Rescale a single point from an arbitrary pitch into the canonical pitch.

    `source_origin` is one of:
    - "bottom_left" — (0, 0) at home-left corner (StatsBomb 120x80 uses top-left in y but
      kloppy normalises this for us, so most loaders hand us bottom-left already).
    - "top_left" — (0, 0) at home-left/top corner (y flipped).
    - "center" — (0, 0) at pitch center (Metrica normalised, Tracab).
    """
    pitch = target or Pitch()
    if source_origin == "bottom_left":
        nx = x / source_length
        ny = y / source_width
    elif source_origin == "top_left":
        nx = x / source_length
        ny = 1.0 - y / source_width
    elif source_origin == "center":
        nx = x / source_length + 0.5
        ny = y / source_width + 0.5
    else:
        raise ValueError(f"unknown source_origin: {source_origin!r}")
    return nx * pitch.length, ny * pitch.width


def flip_horizontal(x: float, y: float, pitch: Pitch | None = None) -> tuple[float, float]:
    """Mirror a point across the centre line (x axis). Used for halftime orientation flip."""
    p = pitch or Pitch()
    return p.length - x, p.width - y
