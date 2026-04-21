# Test fixtures

Tiny, stable, hand-trimmed samples used by the unit test suite.

**Rules:**
- Every fixture <5 MB. Committed to git.
- Real data, not synthetic (unless clearly named `_synthetic`).
- Regeneration tracked in `scripts/regen_fixtures.py`.
- Include a `LICENCE.txt` note per subdirectory referencing the upstream source.
- Never wired into the running dashboard (per the no-test-data rule).

Subdirs (populated in Phase 0):

- `statsbomb/` — one match JSON trimmed to first half
- `pff/` — one match trimmed to first 10 min
- `metrica/` — one match trimmed to first 5 min
- `skillcorner/` — one match trimmed to first 10 min
- `viz/baseline/` — pytest-mpl baseline PNGs
