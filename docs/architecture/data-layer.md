# Data Layer — Design (POC)

**Status:** proposed
**Owner:** data ingestion
**Last updated:** 2026-04-20
**Scope:** free/open data only. No commercial keys. Single-user research workstation.

The four project goals (tactical top-down, player movement, formation strengths/weaknesses, movement-relative-to-position) all collapse to the same requirement at this layer: **synchronised, normalised events + tracking, keyed to a canonical match id, with a single pitch coordinate convention**. Everything below serves that.

---

## 1. Data sources chosen

Four sources, in priority order:

1. **StatsBomb Open Data** — events + 360 freeze frames. Breadth (WC18/22, Euro20/24, WWC19/23, WEuro25, NWSL, Messi career) is unmatched for free. Drives goals 1 and 3.
2. **PFF FC WC2022** (Gradient Sports release) — 64 matches with broadcast tracking + events + grades. The single richest free tracking corpus; it is our main tracking substrate for formation and movement work (goals 3 and 4).
3. **Metrica sample data** — 3 matches, 25 Hz, full-pitch optical, anonymised. Canonical teaching set for pitch control and the only free source that is not broadcast-limited. Used for ground-truth method validation.
4. **SkillCorner open data** — 10 matches of broadcast tracking at 10 Hz. A second tracking provider is important for testing that the normalisation layer really is provider-agnostic.

**Deferred / excluded**:
- Understat — may add later for shot-model cross-checks, not needed for tactical POC.
- FBref advanced stats — **explicitly excluded** (feed terminated Jan 2026).
- Wyscout Pappalardo 2017/18 — events-only, older schema, redundant given StatsBomb coverage.
- DFL/IDSSE Bundesliga 2025 release — excellent but large; park behind a feature flag for a later phase.
- Transfermarkt / Sofascore / FotMob — scouting metadata, out of scope for the tactical layer.

Rationale: StatsBomb + PFF covers the entire 2022 World Cup with both events and tracking, synchronised — that single competition is enough to demonstrate all four project goals end-to-end. Metrica and SkillCorner exist to keep the pipeline honest across providers.

---

## 2. Normalisation strategy

**All data flows through `kloppy` as the single entry point.** kloppy already has first-party loaders for every source chosen (StatsBomb, Metrica, SkillCorner, PFF). This is non-negotiable — writing a second normaliser is how this project dies.

Target schemas (two tables, one per modality):

**Events — SPADL-extended.** SPADL (via `socceraction`) is the academic standard and is what the downstream VAEP/xT models in the metrics layer consume. We extend SPADL rather than replace it:

| column | type | notes |
|---|---|---|
| `match_id` | str | canonical, `{provider}:{native_id}` |
| `period_id` | int8 | 1/2/ET1/ET2/PEN |
| `time_seconds` | float32 | from period start |
| `team_id`, `player_id` | str | canonical (see §cross-ref) |
| `start_x`, `start_y`, `end_x`, `end_y` | float32 | metres, **always L→R attack, 105×68 pitch** |
| `action_type`, `result`, `bodypart` | categorical | SPADL enum |
| `xg`, `xt`, `vaep` | float32 nullable | populated by features layer, not ingest |
| `raw_event_id` | str | back-pointer to provider event |
| `freeze_frame` | struct[] nullable | StatsBomb 360, normalised coords |

**Tracking — long-form frame table.**

| column | type | notes |
|---|---|---|
| `match_id`, `period_id`, `frame_id` | | frame_id monotonic within period |
| `timestamp` | float32 | seconds since period start |
| `player_id` | str | nullable for ball |
| `x`, `y` | float32 | metres, L→R attack |
| `vx`, `vy`, `speed` | float32 | computed, Savitzky-Golay smoothed |
| `is_ball` | bool | |
| `visible` | bool | false = off-camera (SkillCorner / PFF) |

**Orientation.** kloppy hands back provider-native orientation; we immediately transform to **"attacking left→right for the home team in the first half, flipped at halftime"**. This is the single most common source of silent bugs in this domain and must be asserted in ingestion, not trusted.

**Pitch.** All coords rescaled to metres on a 105×68 m pitch. Metrica's [0,1] normalisation is multiplied out; StatsBomb's 120×80 yards is converted.

**Cross-provider IDs.** A small `entities` table maps `(provider, native_id) → canonical_id` for teams and players. For the WC2022 overlap between StatsBomb and PFF this is seeded manually (32 teams, ~800 players) from a YAML crosswalk we hand-maintain. Matches are keyed by `(competition, season, date, home, away)`.

---

## 3. Storage

**Parquet on disk, queried via DuckDB.** No server, no ORM.

- Parquet because event and tracking data are append-mostly, columnar, and 10× smaller than CSV/JSON. Tracking at 25 Hz × 22 players × 90 min is ~3 M rows/match — Parquet + zstd handles this comfortably.
- DuckDB because it reads Parquet natively, does window functions and joins fast on a single machine, and needs zero setup. SQLite is ruled out (no columnar, poor analytics perf); Postgres is overkill for a single user.
- Partitioning:
  - `events/competition=.../season=.../match_id=....parquet`
  - `tracking/competition=.../season=.../match_id=.../period=....parquet` (split by period to keep files <200 MB)
- A single `catalog.duckdb` file holds views over the Parquet tree plus the small dimension tables (`matches`, `teams`, `players`, `entity_crosswalk`, `ingest_runs`).

**Caching.** Raw payloads are cached byte-for-byte under `data/raw/` keyed by source URL + ETag. `statsbombpy`'s on-disk cache is disabled in favour of ours so we own the eviction policy. Processed Parquet under `data/processed/` is the analytical layer; features (xT, VAEP, pitch control tensors) live under `data/features/`. All three layers are reproducible from `data/raw/`.

**Incremental updates.** An `ingest_runs` table records `(source, match_id, source_hash, ingested_at, kloppy_version)`. A match is re-ingested only if the source hash changes or the kloppy/schema version advances. StatsBomb Open Data is a git repo — we track the commit SHA per pull.

---

## 4. Directory layout

```
football-analysis/
├── data/                         # git-ignored
│   ├── raw/
│   │   ├── statsbomb/            # mirror of open-data repo, pinned commit
│   │   ├── pff_wc2022/
│   │   ├── metrica/
│   │   └── skillcorner/
│   ├── processed/
│   │   ├── events/competition=.../season=.../*.parquet
│   │   └── tracking/competition=.../season=.../match_id=.../*.parquet
│   ├── features/                 # xT grids, VAEP, pitch-control tensors
│   └── catalog.duckdb
├── src/
│   └── data/
│       ├── __init__.py
│       ├── sources/              # one module per provider
│       │   ├── statsbomb.py
│       │   ├── pff.py
│       │   ├── metrica.py
│       │   └── skillcorner.py
│       ├── normalize/
│       │   ├── events_spadl.py
│       │   ├── tracking.py
│       │   ├── orientation.py
│       │   └── pitch.py
│       ├── catalog.py            # DuckDB views, ingest_runs
│       ├── crosswalk/            # YAML id mappings
│       │   └── wc2022.yaml
│       ├── validation.py         # pandera/pydantic schemas
│       └── cli.py                # Typer entrypoint
├── tests/
│   └── data/
│       ├── fixtures/             # <5 MB each, see §testing
│       ├── test_statsbomb.py
│       ├── test_orientation.py
│       └── ...
└── docs/architecture/data-layer.md
```

Dashboard code never reads from `data/raw/` or from provider SDKs. It reads only from `catalog.duckdb`. This firewall is how we keep sample/test data out of the UI.

---

## 5. CLI interface

Single Typer app, installed as `fa-data`:

```bash
# Mirror a source into data/raw/ (idempotent)
fa-data fetch statsbomb --competition "FIFA World Cup" --season 2022
fa-data fetch pff --competition wc2022
fa-data fetch metrica --match all
fa-data fetch skillcorner --match all

# Normalise raw → processed Parquet
fa-data ingest statsbomb --competition WC2022
fa-data ingest pff --match 3869685
fa-data ingest all --since 2026-01-01

# Rebuild DuckDB views / crosswalk
fa-data catalog rebuild

# Validate
fa-data validate --layer processed
fa-data validate --match 3869685 --strict

# Maintenance
fa-data status                  # coverage report
fa-data clean --layer features  # drop derived data, keep raw
```

`fetch` is cache-aware and network-bound; `ingest` is CPU-bound and re-runnable. They are deliberately separate steps so CI can unit-test `ingest` against fixtures without a network.

---

## 6. Dependencies

Python 3.11+, managed with `uv` and pinned in `pyproject.toml`:

- `kloppy >=3.15,<4` — normalisation spine
- `statsbombpy >=1.14` — StatsBomb client
- `socceraction >=1.5` — SPADL conversion
- `duckdb >=1.1`
- `pyarrow >=16`
- `pandas >=2.2`, `polars >=1.5` (polars for tracking hot paths)
- `pandera >=0.20` — schema validation
- `pydantic >=2.7` — config + crosswalk models
- `typer >=0.12`, `rich >=13`
- `httpx >=0.27` — PFF / SkillCorner direct downloads
- `pyyaml`, `platformdirs`

Test-only: `pytest`, `pytest-cov`, `pytest-xdist`, `hypothesis`, `responses`.

---

## 7. Open questions / tradeoffs

- **SPADL vs raw kloppy event model.** SPADL loses some StatsBomb richness (technique enums, pressure chains). We keep `raw_event_id` so the full payload is always one join away in `data/raw/`. Acceptable.
- **Tracking frequency harmonisation.** Metrica is 25 Hz, SkillCorner is 10 Hz, PFF ~10 Hz. We store at native rate; downsample on read. Do not upsample — pitch control derivatives are garbage on interpolated frames.
- **StatsBomb 360 vs real tracking.** 360 is event-triggered and broadcast-limited. It lives in the `freeze_frame` column of the events table, not the tracking table, to prevent accidentally treating it as continuous.
- **Entity crosswalk maintenance.** Manual YAML scales to WC2022. If we add a league, this becomes a real problem — revisit with fuzzy name matching + review queue.
- **Polars vs pandas.** Use polars inside normalisation (tracking is the bottleneck), return pandas at the API boundary for compatibility with `socceraction` and `mplsoccer`.

---

## 8. Testing approach

Target >90% line coverage on `src/data/`. Strategy:

1. **Tiny real fixtures**, not synthetic. One StatsBomb match JSON (~3 MB), one Metrica match (head-truncated to the first 5 minutes, ~2 MB), one PFF match (first 10 min), one SkillCorner match (first 10 min). All under `tests/data/fixtures/`, all under 5 MB each, all with a LICENCE note. These never enter `data/`.
2. **Schema tests** via `pandera` — every processed Parquet must pass the events or tracking schema. Run in CI and as `fa-data validate`.
3. **Invariants** — property tests with `hypothesis`:
   - All `x` in [0, 105], all `y` in [0, 68].
   - For any match, every event's `team_id` is one of the two match teams.
   - Orientation: the home team's average x in the first half is < 52.5 iff attacking L→R is correctly applied (check on real fixtures, not synthetic).
   - Round-trip: ingest → read → assert frame counts and event counts equal provider-reported totals.
4. **HTTP isolation** — `sources/` modules mock at the `httpx` / `statsbombpy` boundary using `responses`. No test ever hits a real network.
5. **Coverage gates** — `pytest --cov=src/data --cov-fail-under=90` in CI.
6. **Fixture regeneration script** under `scripts/regen_fixtures.py` so truncated samples stay reproducible when upstream data moves.

Explicit non-goal: do not test kloppy's or statsbombpy's internals. Test only our normalisation, orientation, crosswalk, catalog, and CLI.
