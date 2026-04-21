"""Tests for the DuckDB catalog module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from football_analysis.data import catalog as c


def test_processed_paths_shape(tmp_path: Path) -> None:
    e = c.processed_events_path("WC2022", "2022", "statsbomb-1", processed_dir=tmp_path / "proc")
    assert e.name == "statsbomb-1.parquet"
    assert "competition=WC2022" in e.parts
    assert "season=2022" in e.parts

    t = c.processed_tracking_path("WC2022", "2022", "statsbomb-1", period=1, processed_dir=tmp_path / "proc")
    assert t.name == "period=1.parquet"
    assert "match_id=statsbomb-1" in t.parts


def test_init_and_record(tmp_path: Path) -> None:
    db = tmp_path / "catalog.duckdb"
    c.init_schema(catalog_path=db)
    c.record_ingest("statsbomb", "statsbomb:1", "abc123", "0.1.0", catalog_path=db)
    rows = c.ingested_matches(catalog_path=db)
    assert len(rows) == 1
    assert rows[0]["match_id"] == "statsbomb:1"
    assert rows[0]["source_hash"] == "abc123"


def test_record_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "catalog.duckdb"
    c.init_schema(catalog_path=db)
    c.record_ingest("statsbomb", "statsbomb:1", "abc", "0.1.0", catalog_path=db)
    c.record_ingest("statsbomb", "statsbomb:1", "def", "0.1.0", catalog_path=db)
    rows = c.ingested_matches(source="statsbomb", catalog_path=db)
    assert len(rows) == 1
    assert rows[0]["source_hash"] == "def"


def test_rebuild_view_over_parquet_tree(tmp_path: Path) -> None:
    db = tmp_path / "catalog.duckdb"
    processed = tmp_path / "processed"
    out = c.processed_events_path("WC2022", "2022", "statsbomb-1", processed_dir=processed)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({"match_id": ["statsbomb:1"], "period": [1], "time_seconds": [0.0], "action_type": ["pass"]})
    pq.write_table(pa.Table.from_pandas(df), out)

    c.rebuild_event_views(catalog_path=db, processed_dir=processed)

    with c.connect(catalog_path=db, read_only=True) as con:
        out_rows = con.execute(
            "SELECT match_id, action_type, competition, CAST(season AS VARCHAR) FROM events"
        ).fetchall()
    assert out_rows == [("statsbomb:1", "pass", "WC2022", "2022")]
