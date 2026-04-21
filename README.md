# football-analysis

Top-down tactical analysis of 11v11 association football: single POV of any match, player
movement, formation strengths/weaknesses, movement relative to gameplay.

## Quick start

```bash
make install        # uv sync + pre-commit hooks
cp .env.example .env
make test           # run pytest
make app            # launch Streamlit dashboard
```

## Layout

See [`docs/architecture/00-overview.md`](docs/architecture/00-overview.md) for the full map.

- `src/football_analysis/` — installable package (`data`, `analytics`, `viz`, `app`, `cv`)
- `tests/` — mirrors src, plus `fixtures/` and `integration/`
- `data/` — gitignored; vendor data lives here, never in git
- `docs/` — research (00–05) + architecture (`architecture/*.md`)
- `notebooks/` — `exploration/` (scratch) and `analysis/` (curated)

## Phases

| Phase | Focus | Status |
|---|---|---|
| 0 | Repo scaffold, fixtures, CI | ⏳ current |
| 1 | Events: xG, xT, VAEP + Match Replay page | pending |
| 2 | Tracking + pitch control + Plotly tactical view | pending |
| 3 | Formation detection + strengths/weaknesses | pending |
| 4 | Optional CV pipeline (broadcast → tracking) | pending |

See [`docs/00-executive-summary.md`](docs/00-executive-summary.md).

## Data sources (POC, free only)

- StatsBomb Open Data (WC, Euro, NWSL, Messi)
- PFF FC World Cup 2022 (tracking + events, 64 matches)
- Metrica Sports (3 sample matches, 25 Hz)
- SkillCorner open data (10 broadcast-tracking matches)

FBref advanced stats are not used — the Opta feed was terminated in Jan 2026.

## Development

```bash
make fmt     # ruff format + autofix
make lint    # ruff check + format check
make type    # mypy
make test    # pytest parallel
make cov     # pytest + HTML coverage report
make ci      # lint + type + test (matches GitHub Actions)
```

Coverage target: >90% line + branch. Streamlit code under `app/` is excluded from coverage
(smoke-tested via `streamlit.testing.v1.AppTest` instead).

## License

MIT (code). Data retains its provider licenses — see each `data/raw/*/README.md`.
