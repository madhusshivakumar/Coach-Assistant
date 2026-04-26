"""SkillCorner Open Data source — broadcast-tracking for ~10 free matches.

SkillCorner publishes ``match.json`` (metadata + player rosters) and
``tracking_extrapolated.jsonl`` (10 Hz tracking, one JSON object per frame). The
tracking files are stored as Git LFS pointers in the repo; the actual content
must be fetched from ``media.githubusercontent.com``.

We mirror both files to ``data/raw/skillcorner/match-{id}/`` and write a
sentinel so ``fa-data status`` knows what's on disk without reparsing.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from football_analysis.config import get_settings
from football_analysis.logging import get_logger

_log = get_logger(__name__)

# These were the 10 match IDs in the public repo as of 2025. The list can be
# refreshed via the GitHub contents API; we hardcode for offline reproducibility.
DEFAULT_MATCH_IDS: tuple[str, ...] = (
    "1886347",
    "1899585",
    "1925299",
    "1953632",
    "1996435",
    "2006229",
    "2011166",
    "2013725",
    "2015213",
    "2017461",
)

_RAW_BASE = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches"
_LFS_BASE = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches"


def _raw_cache_dir(raw_dir: Path | None = None) -> Path:
    root = raw_dir if raw_dir is not None else get_settings().raw_dir
    return root / "skillcorner"


def list_available(timeout: float = 30.0) -> list[str]:
    """Query the GitHub contents API to list current match IDs in the repo.

    Returns ``DEFAULT_MATCH_IDS`` on any failure (network / API limit) so
    offline use still works.
    """
    try:
        r = httpx.get(
            "https://api.github.com/repos/SkillCorner/opendata/contents/data/matches",
            timeout=timeout,
        )
        if r.status_code != 200:
            return list(DEFAULT_MATCH_IDS)
        return sorted(d["name"] for d in r.json() if d.get("type") == "dir")
    except Exception:  # pragma: no cover - network failure path
        return list(DEFAULT_MATCH_IDS)


def fetch_match(
    match_id: str,
    raw_dir: Path | None = None,
    force: bool = False,
    timeout: float = 120.0,
) -> Path:
    """Download a single SkillCorner match's metadata + tracking JSONL.

    Args:
        match_id: SkillCorner match ID (string, e.g. ``"1886347"``).
        raw_dir: override the default raw-data dir.
        force: re-download even if the sentinel file is present.
        timeout: per-HTTP-request timeout in seconds. The tracking file can be
            ~90 MB so a generous timeout is appropriate.

    Returns:
        The match directory under ``raw_dir/skillcorner/``.
    """
    cache = _raw_cache_dir(raw_dir) / f"match-{match_id}"
    cache.mkdir(parents=True, exist_ok=True)
    sentinel = cache / "fetched"
    if sentinel.exists() and not force:
        _log.info("skillcorner_cached", match_id=match_id)
        return cache

    match_url = f"{_RAW_BASE}/{match_id}/{match_id}_match.json"
    tracking_url = f"{_LFS_BASE}/{match_id}/{match_id}_tracking_extrapolated.jsonl"
    _log.info("skillcorner_fetch", match_id=match_id)

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        match_resp = client.get(match_url)
        match_resp.raise_for_status()
        (cache / "match.json").write_text(match_resp.text, encoding="utf-8")

        tracking_resp = client.get(tracking_url)
        tracking_resp.raise_for_status()
        # The LFS-resolved file is ASCII-friendly JSONL; safe to write as text.
        (cache / "tracking.jsonl").write_text(tracking_resp.text, encoding="utf-8")

    sentinel.write_text(
        json.dumps(
            {"match_id": match_id, "match_bytes": len(match_resp.text), "tracking_bytes": len(tracking_resp.text)}
        ),
        encoding="utf-8",
    )
    return cache


def list_fetched(raw_dir: Path | None = None) -> list[str]:
    cache = _raw_cache_dir(raw_dir)
    if not cache.exists():
        return []
    return sorted(p.name.removeprefix("match-") for p in cache.glob("match-*"))


def load_match(match_id: str, raw_dir: Path | None = None) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Read a fetched SkillCorner match into ``(match_metadata, tracking_frames)``.

    Tracking frames are returned as a list of dicts (one per JSONL line) — the
    normaliser flattens them into the canonical long-form schema.
    """
    cache = _raw_cache_dir(raw_dir) / f"match-{match_id}"
    if not (cache / "fetched").exists():
        raise FileNotFoundError(f"SkillCorner match {match_id!r} not fetched yet")
    match_data = json.loads((cache / "match.json").read_text(encoding="utf-8"))
    tracking = [
        json.loads(line) for line in (cache / "tracking.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    return match_data, tracking
