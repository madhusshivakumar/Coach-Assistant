"""StatsBomb Open Data source.

Uses kloppy's StatsBomb loader as the spine. We wrap it so that:
- HTTP calls route through httpx (testable via `responses`).
- Raw payloads are cached byte-for-byte in `data/raw/statsbomb/` keyed by URL.
- A uniform `list_matches` / `load_match` API is exposed to the rest of the data layer.

For the Open Data, kloppy itself fetches over the network — we simply point it at
local cached JSON when available and fall back to HTTP otherwise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx

from football_analysis.config import get_settings
from football_analysis.logging import get_logger

_log = get_logger(__name__)

OPEN_DATA_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


@dataclass(frozen=True)
class StatsBombMatch:
    """Identifying tuple for an open-data match."""

    competition_id: int
    season_id: int
    match_id: int

    @property
    def match_key(self) -> str:
        return f"statsbomb:{self.match_id}"


def _cache_path(raw_dir: Path, kind: str, *parts: str | int) -> Path:
    """Cache key for a StatsBomb asset (competitions / matches / events / lineups / three-sixty)."""
    if not parts:
        return raw_dir / "statsbomb" / f"{kind}.json"
    return raw_dir / "statsbomb" / kind / Path(*[str(p) for p in parts]).with_suffix(".json")


def _fetch_json(
    url: str,
    cache_path: Path,
    client: httpx.Client | None = None,
    force: bool = False,
) -> Any:
    """GET a URL, cache the body as JSON, return parsed payload."""
    if cache_path.exists() and not force:
        _log.debug("statsbomb_cache_hit", url=url, path=str(cache_path))
        return json.loads(cache_path.read_text(encoding="utf-8"))
    _log.info("statsbomb_fetch", url=url)
    close = client is None
    c = client or httpx.Client(timeout=30.0)
    try:
        resp = c.get(url)
        resp.raise_for_status()
        body = resp.text
    finally:
        if close:
            c.close()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(body, encoding="utf-8")
    return json.loads(body)


def list_competitions(
    raw_dir: Path | None = None, client: httpx.Client | None = None, force: bool = False
) -> list[dict[str, Any]]:
    """Return the competitions manifest."""
    raw = raw_dir or get_settings().raw_dir
    path = _cache_path(raw, "competitions")
    return cast(
        "list[dict[str, Any]]",
        _fetch_json(f"{OPEN_DATA_BASE}/competitions.json", path, client=client, force=force),
    )


def list_matches(
    competition_id: int,
    season_id: int,
    raw_dir: Path | None = None,
    client: httpx.Client | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Return the match manifest for a competition/season."""
    raw = raw_dir or get_settings().raw_dir
    path = _cache_path(raw, "matches", competition_id, season_id)
    return cast(
        "list[dict[str, Any]]",
        _fetch_json(
            f"{OPEN_DATA_BASE}/matches/{competition_id}/{season_id}.json",
            path,
            client=client,
            force=force,
        ),
    )


def load_match_events(
    match_id: int,
    raw_dir: Path | None = None,
    client: httpx.Client | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Return the raw event stream for a match."""
    raw = raw_dir or get_settings().raw_dir
    path = _cache_path(raw, "events", match_id)
    return cast(
        "list[dict[str, Any]]",
        _fetch_json(f"{OPEN_DATA_BASE}/events/{match_id}.json", path, client=client, force=force),
    )


def load_match_lineups(
    match_id: int,
    raw_dir: Path | None = None,
    client: httpx.Client | None = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Return lineups for a match."""
    raw = raw_dir or get_settings().raw_dir
    path = _cache_path(raw, "lineups", match_id)
    return cast(
        "list[dict[str, Any]]",
        _fetch_json(f"{OPEN_DATA_BASE}/lineups/{match_id}.json", path, client=client, force=force),
    )
