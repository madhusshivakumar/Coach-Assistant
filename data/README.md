# data/

Local data tree. **Entirely gitignored** except this README and `.gitkeep` files.

```
data/
├── raw/         # vendor drops, pinned by source hash
│   ├── statsbomb/
│   ├── pff_wc2022/
│   ├── metrica/
│   └── skillcorner/
├── interim/     # kloppy-normalized caches
├── processed/   # analytics-ready parquet (events, tracking)
├── features/    # derived: xT/VAEP/pitch-control/roles
├── external/    # crosswalks, reference CSVs
└── catalog.duckdb
```

Location is configurable via `FA_DATA_DIR` (defaults to `./data`). Override in `.env`:

```
FA_DATA_DIR=D:/football-data
```

Test fixtures live under `tests/fixtures/` — never here.
