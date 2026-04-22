"""DuckDB catalog over the Parquet tree.

Single `catalog.duckdb` file at the data root. Exposes:
- Views over `processed/events/**/*.parquet` and `processed/tracking/**/*.parquet`.
- A small `ingest_runs` dimension table for incremental-update bookkeeping.

Everything downstream (dashboard, notebooks) reads via this catalog only.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from football_analysis.config import get_settings
from football_analysis.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

_log = get_logger(__name__)


def processed_events_path(competition: str, season: str, match_id: str, processed_dir: Path | None = None) -> Path:
    """Canonical Parquet path for processed events."""
    pd = processed_dir or get_settings().processed_dir
    return pd / "events" / f"competition={competition}" / f"season={season}" / f"{match_id}.parquet"


def processed_tracking_path(
    competition: str, season: str, match_id: str, period: int, processed_dir: Path | None = None
) -> Path:
    """Canonical Parquet path for processed tracking (per period)."""
    pd = processed_dir or get_settings().processed_dir
    return (
        pd
        / "tracking"
        / f"competition={competition}"
        / f"season={season}"
        / f"match_id={match_id}"
        / f"period={period}.parquet"
    )


@contextmanager
def connect(catalog_path: Path | None = None, read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a DuckDB connection to the project catalog."""
    path = catalog_path or get_settings().catalog_path
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def init_schema(catalog_path: Path | None = None) -> None:
    """Create the small bookkeeping tables. Idempotent."""
    with connect(catalog_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS ingest_runs (
                source        VARCHAR NOT NULL,
                match_id      VARCHAR NOT NULL,
                source_hash   VARCHAR,
                ingested_at   TIMESTAMP NOT NULL,
                schema_version VARCHAR NOT NULL,
                PRIMARY KEY (source, match_id)
            )
            """
        )


def record_ingest(
    source: str,
    match_id: str,
    source_hash: str | None,
    schema_version: str,
    catalog_path: Path | None = None,
) -> None:
    """Mark a match as ingested. Overwrites any previous record."""
    with connect(catalog_path) as con:
        con.execute("DELETE FROM ingest_runs WHERE source = ? AND match_id = ?", [source, match_id])
        con.execute(
            "INSERT INTO ingest_runs VALUES (?, ?, ?, ?, ?)",
            [source, match_id, source_hash, datetime.now(UTC), schema_version],
        )


def ingested_matches(source: str | None = None, catalog_path: Path | None = None) -> list[dict[str, Any]]:
    """Return the ingested-match log. Filter by source if provided."""
    with connect(catalog_path, read_only=True) as con:
        if source is None:
            cur = con.execute("SELECT * FROM ingest_runs ORDER BY ingested_at DESC")
        else:
            cur = con.execute("SELECT * FROM ingest_runs WHERE source = ? ORDER BY ingested_at DESC", [source])
        cols = [d[0] for d in cur.description or []]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def rebuild_event_views(catalog_path: Path | None = None, processed_dir: Path | None = None) -> None:
    """(Re)build DuckDB views over the event Parquet tree.

    Silent no-op if no event parquet files exist yet (a first ingest may be tracking-only).
    """
    pd = processed_dir or get_settings().processed_dir
    events_root = pd / "events"
    if not events_root.exists() or not any(events_root.rglob("*.parquet")):
        _log.info("catalog_rebuild_events_skipped", reason="no parquet files")
        return
    events_glob = str(events_root / "**" / "*.parquet")
    with connect(catalog_path) as con:
        con.execute("DROP VIEW IF EXISTS events")
        # `hive_partitioning=1` ensures `competition=` and `season=` are exposed as columns.
        con.execute(f"CREATE VIEW events AS SELECT * FROM read_parquet('{events_glob}', hive_partitioning=1)")
    _log.info("catalog_rebuild_events", glob=events_glob)


def rebuild_tracking_views(catalog_path: Path | None = None, processed_dir: Path | None = None) -> None:
    """(Re)build DuckDB views for tracking Parquets. Silent no-op if none exist yet."""
    pd = processed_dir or get_settings().processed_dir
    tracking_root = pd / "tracking"
    if not tracking_root.exists() or not any(tracking_root.rglob("*.parquet")):
        _log.info("catalog_rebuild_tracking_skipped", reason="no parquet files")
        return
    tracking_glob = str(tracking_root / "**" / "*.parquet")
    with connect(catalog_path) as con:
        con.execute("DROP VIEW IF EXISTS tracking")
        con.execute(f"CREATE VIEW tracking AS SELECT * FROM read_parquet('{tracking_glob}', hive_partitioning=1)")
    _log.info("catalog_rebuild_tracking", glob=tracking_glob)
