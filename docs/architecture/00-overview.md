# POC Architecture — Unified Overview

**Status:** approved-for-scaffold · **Date:** 2026-04-20 · **Designed via ruflo swarm (4 architects)**

This is the reconciled design. Each sibling doc (`data-layer.md`, `analytics-layer.md`, `viz-layer.md`, `infra-layer.md`) owns its own scope; this file resolves boundary decisions and captures conflicts that were reconciled.

---

## 1. Guiding principles

1. **Metric 105×68 pitch, attacking-left-to-right, home team right after kickoff** is the canonical coordinate system. Enforced at ingest, never re-negotiated downstream.
2. **Tracking is the clock.** Events are joined to frames via a ±200 ms window.
3. **Phase-conditioned, role-relative.** Every structural/movement metric is `(phase, role)`-keyed. Bialkowski role assignment + phase classifier are the two non-optional primitives.
4. **One-way dependency chain.** `data → analytics → viz`. No back-writes, no cross-imports.
5. **`src/football_analysis/viz/` is Streamlit-free.** Streamlit lives only in `src/football_analysis/app/`. Keeps plots testable + notebook-friendly.
6. **Dashboard reads only from the catalog.** No sample/test data ever wired to the UI.

---

## 2. Reconciled repository layout

```
football-analysis/
├── .github/workflows/ci.yml
├── .pre-commit-config.yaml
├── .gitattributes
├── .gitignore
├── Makefile
├── make.bat
├── pyproject.toml
├── uv.lock
├── README.md
├── .env.example
│
├── config/
│   └── default.yaml
│
├── data/                              # gitignored except .gitkeep + READMEs
│   ├── raw/       {statsbomb, pff_wc2022, metrica, skillcorner}
│   ├── interim/   # kloppy-normalized caches
│   ├── processed/ # analytics-ready parquet (events, tracking)
│   ├── features/  # derived: xT/VAEP values, pitch-control integrals, role assignments
│   ├── external/  # crosswalks, schemas, reference CSVs
│   └── catalog.duckdb
│
├── docs/
│   ├── 00-executive-summary.md ... 05-tools-and-community.md
│   └── architecture/
│       ├── 00-overview.md   (this file)
│       ├── data-layer.md
│       ├── analytics-layer.md
│       ├── viz-layer.md
│       └── infra-layer.md
│
├── notebooks/
│   ├── exploration/   # scratch, outputs stripped by nbstripout
│   └── analysis/      # curated, reproducible
│
├── scripts/
│   ├── fetch_statsbomb.py
│   ├── fetch_pff.py
│   └── regen_fixtures.py
│
├── src/
│   └── football_analysis/             # single installable package
│       ├── __init__.py
│       ├── config.py                  # pydantic-settings
│       ├── logging.py                 # structlog
│       ├── data/                      # = data architect's src/data
│       │   ├── sources/               # statsbomb.py, pff.py, metrica.py, skillcorner.py
│       │   ├── normalize/             # events_spadl.py, tracking.py, orientation.py, pitch.py
│       │   ├── crosswalk/             # YAML id mappings
│       │   ├── catalog.py
│       │   ├── validation.py
│       │   └── cli.py                 # → `fa-data` entrypoint
│       ├── analytics/                 # = analytics architect's src/analytics
│       │   ├── types.py
│       │   ├── pitch.py               # canonical geometry helpers (shared)
│       │   ├── xg/
│       │   ├── possession_value/      # xt.py, vaep.py
│       │   ├── pitch_control/         # spearman.py, motion.py, obso.py
│       │   ├── formations/            # roles.py, detect.py, shape.py, strengths.py
│       │   ├── phases/                # classifier.py (rules v1), segments.py
│       │   ├── movement/              # role_relative.py, off_ball.py, combinations.py
│       │   ├── team/                  # ppda.py, field_tilt.py, compactness.py
│       │   └── pipeline/              # runner.py, cache.py, schemas.py
│       ├── viz/                       # NO streamlit imports
│       │   ├── pitch.py
│       │   ├── theme.py
│       │   ├── static/                # shot_map, pass_network, heatmap, pass_sonar, radar
│       │   ├── interactive/           # tactical_view.py (Plotly), pitch_control.py
│       │   ├── overlays/              # roles, phase, events
│       │   └── export/                # png.py, mp4.py, html.py
│       ├── app/                       # Streamlit only
│       │   ├── main.py
│       │   ├── state.py
│       │   ├── data.py                # @st.cache_data loaders from catalog.duckdb
│       │   ├── components/            # playback, match_picker, player_picker
│       │   └── pages/                 # 01_match_replay, 02_player_profile, 03_formation_compare, 04_team_style, 05_pitch_control
│       └── cv/                        # Phase 4 placeholder (opt-in extra)
│
└── tests/
    ├── unit/                          # mirrors src/football_analysis/
    │   ├── data/
    │   ├── analytics/
    │   ├── viz/
    │   └── app/
    ├── integration/
    ├── fixtures/                      # tiny real data (<5 MB each), committed
    │   ├── statsbomb/
    │   ├── pff/
    │   ├── metrica/
    │   └── viz/baseline/              # pytest-mpl baseline PNGs
    └── conftest.py
```

---

## 3. Conflict resolutions

| Conflict | Options | Resolution |
|---|---|---|
| Package name | `src/data/` (flat) vs `src/football_analysis/io/` | **`src/football_analysis/` single package**, data architect's subpackage names preserved under it |
| Data subdirs | raw/processed/features vs raw/interim/processed/external | **Merged: raw + interim + processed + features + external** |
| StatsBomb 360 | nested column (data) vs separate `freeze_frames` DataFrame (analytics) | **Stored nested in events Parquet; exposed as flattened DataFrame via an adapter at analytics boundary** |
| Fixtures path | `tests/data/fixtures/` vs `tests/fixtures/` | **`tests/fixtures/` central**, subdirs per provider |
| Tests layout | Mirror-src vs unit/integration split | **Both: `tests/unit/` mirrors src, `tests/integration/` for round-trips, `tests/fixtures/` shared** |
| floodlight | Analytics defers, infra includes | **Deferred** — removed from pyproject defaults |
| Python version | 3.11 vs 3.11+ | **`>=3.11,<3.12`** per infra rationale |
| CLI entrypoints | `fa-data` proposed | Adopt `fa-*` convention: **`fa-data`, `fa-analytics`, `fa`** (umbrella) |

---

## 4. Interfaces (layer contracts)

### data → analytics

Three strictly-validated DataFrames, all in metric 105×68 coords, home attacks L→R:

- **`events_spadl`** (always): `match_id, period, time_seconds, team_id, player_id, start_x/y, end_x/y, action_type, result, bodypart, original_event_id, freeze_frame (nullable struct[])`
- **`tracking`** (optional, 10–25 Hz): `match_id, period, frame_id, time_seconds, player_id, team_id, x, y, vx, vy, speed, is_ball, visible`
- **`metadata`**: pitch dims, lineup, kickoff direction

When `tracking` is absent and `freeze_frame` is present, analytics uses freeze-frames as **degraded spatial fallback** (zero-velocity intercept model) for event-time OBSO/control only.

### analytics → viz

Parquet + Arrow frames with fixed schema:

| Frame | Key cols | Producer |
|---|---|---|
| `tracking` | `frame_id, player_id, x, y, vx, vy, possession_team, possession_player` | analytics pipeline |
| `events` | `event_id, frame_id, type, x, y, end_x, end_y, xg, xt, vaep` | analytics pipeline |
| `roles` | `frame_id, player_id, role_id, role_label` | `formations/roles` |
| `phases` | `frame_start, frame_end, phase` | `phases/segments` |
| `pitch_control` | `frame_id, grid` (104×68 float32) | `pitch_control/spearman` |
| `formations` | `team_id, phase, formation_code, avg_positions` | `formations/detect` |
| `player_season` | `player_id, metric, value, percentile` | aggregates |

---

## 5. Dependency groups (consolidated pyproject)

| Group | Purpose | Key libs |
|---|---|---|
| `core` (required) | Ingestion + data model | kloppy, statsbombpy, numpy, pandas, polars, pyarrow, duckdb, pandera, pydantic, pydantic-settings, structlog, typer, rich, httpx |
| `analytics` | Compute | socceraction, scikit-learn, scipy, shapely |
| `viz` | Plotting + dashboard | mplsoccer, matplotlib, plotly, streamlit, soccerplots, imageio-ffmpeg |
| `cv` | Phase 4 opt-in | opencv-python, torch, ultralytics, supervision |
| `dev` | Tooling | pytest, pytest-cov, pytest-xdist, pytest-randomly, pytest-mpl, hypothesis, syrupy, responses, ruff, mypy, pre-commit, ipykernel, jupyterlab, nbstripout |
| `docs` | Site | mkdocs-material, mkdocstrings[python] |

`uv sync --all-extras --group dev` installs everything except `cv`; `cv` is opt-in.

---

## 6. Implementation phasing (linked to exec summary)

| Phase | Deliverables | Layers touched |
|---|---|---|
| **Phase 0** (orientation, ~1 week) | Repo scaffold, fixtures, CI, `fa-data fetch statsbomb --competition WC2022` works end-to-end, first shot-map notebook | infra, data |
| **Phase 1** (event analysis, 2–4 wks) | Full ingestion through StatsBomb + Metrica; xG, xT, VAEP; Streamlit Match Replay page (events only, no tracking) | data, analytics.{xg,possession_value,phases,team}, viz.static, app |
| **Phase 2** (tracking + pitch control, 3–5 wks) | PFF FC WC2022 ingested; pitch control + OBSO implemented; tactical view Plotly animated; pitch-control explorer page | analytics.{pitch_control,movement}, viz.interactive |
| **Phase 3** (formation deep dive, 3–4 wks) | Bialkowski role assignment, formation detection, strengths/weaknesses radars, movement combinations detectors | analytics.{formations,movement}, viz.static.radar, app pages 03-04 |
| **Phase 4** (optional CV, 4–8+ wks) | Broadcast → tracking via sn-gamestate; custom match support | cv |

---

## 7. Testing & quality bar

- **>90% line + branch coverage** enforced in CI (`--cov-fail-under=90`).
- **pytest-mpl + syrupy** for plot regression tests.
- **Hypothesis** for pure-function invariants (pitch bounds, permutation of role assignment, idempotent orientation).
- **Tiny real fixtures** (<5 MB each), regenerable via `scripts/regen_fixtures.py`.
- **Integration tests** marked `@pytest.mark.integration`; run nightly in CI.
- **CI matrix**: ubuntu-latest + windows-latest on Python 3.11.

---

## 8. Deferred decisions (tracked, non-blocking)

1. DVC for `data/` once Phase 2+ grows past ~20 GB.
2. `cv-gpu` extra with CUDA torch wheels in Phase 4.
3. Upgrade Plotly tactical view → FastAPI + React micro-frontend if ≥25 Hz smooth playback is needed.
4. `nox` for multi-env testing.
5. Entity crosswalk fuzzy-matching when we add a second competition beyond WC2022.

---

## 9. Non-goals for POC

- No real-time / live ingestion.
- No auth, no multi-user.
- No commercial APIs (no StatsBomb IQ, no Wyscout, no Opta).
- No broadcast video processing (Phase 4 only if pursued).
- No betting / predictive match outcome models.
- No scouting / similarity search / player-vectors.
