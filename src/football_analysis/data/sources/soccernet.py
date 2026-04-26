"""SoccerNet GameState 2024 source — clips with pitch-projected tracking.

Each ``SNGS-XXX`` directory in the test/train zip is a 30-second clip at 25 Hz
around a known action (Corner, Shot, Goal, Free kick, etc.). The
``Labels-GameState.json`` carries per-frame, per-object annotations including:

- ``bbox_image``: pixel-space bounding box (we ignore — we don't process video)
- ``bbox_pitch``: PITCH-PROJECTED bottom-edge coordinates (the foot positions).
  This is what makes SoccerNet drop-in usable for our engine — no CV step needed.
- ``attributes.role``: player / goalkeeper / referee / ball
- ``attributes.team``: ``left`` or ``right`` for players (referees have null)
- ``track_id``: persistent within a clip; reused across clips so don't merge globally

We work directly off the zip (don't unzip 36 k JPEG frames we'll never use).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from football_analysis.config import get_settings
from football_analysis.logging import get_logger

_log = get_logger(__name__)


def _raw_cache_dir(raw_dir: Path | None = None) -> Path:
    root = raw_dir if raw_dir is not None else get_settings().raw_dir
    return root / "soccernet" / "gamestate-2024"


def list_clips(split: str = "test", raw_dir: Path | None = None) -> list[str]:
    """List clip names (e.g. ``"SNGS-116"``) inside the ``split.zip``.

    Returns empty list if the zip isn't present (caller should fetch first).
    """
    zip_path = _raw_cache_dir(raw_dir) / f"{split}.zip"
    if not zip_path.exists():
        return []
    with zipfile.ZipFile(zip_path) as z:
        clip_names = sorted({n.split("/")[0] for n in z.namelist() if "/" in n and n.startswith("SNGS-")})
    return clip_names


def load_clip(clip_name: str, split: str = "test", raw_dir: Path | None = None) -> dict[str, Any]:
    """Load one clip's ``Labels-GameState.json`` directly from the zip.

    Args:
        clip_name: e.g. ``"SNGS-116"``.
        split: ``"test"`` / ``"train"`` / ``"valid"`` / ``"challenge"``.
        raw_dir: override the default raw-data dir.

    Returns:
        Parsed JSON with ``info``, ``images``, ``annotations``, ``categories``.
    """
    zip_path = _raw_cache_dir(raw_dir) / f"{split}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"SoccerNet split {split!r} not fetched yet (expected {zip_path})")
    target = f"{clip_name}/Labels-GameState.json"
    with zipfile.ZipFile(zip_path) as z:
        try:
            with z.open(target) as f:
                payload: dict[str, Any] = json.load(f)
                return payload
        except KeyError as e:
            raise FileNotFoundError(f"clip {clip_name!r} not found in {zip_path}") from e
