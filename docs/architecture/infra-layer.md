# Infrastructure Layer — Design

Status: draft (2026-04-20)
Scope: repository layout, Python environment, dev toolchain, CI, data handling, Windows support. Decisions only — no analytics logic.

The project is a Windows-11 POC with four goals: top-down POV of matches, player movement, formation analysis, movement-vs-gameplay. Stack will eventually include heavy deps (OpenCV, PyTorch, socceraction, kloppy, mplsoccer), so the infra has to handle fat, platform-sensitive wheels cleanly.

---

## 1. Repository Layout

```
football-analysis/
├── .github/workflows/        # CI (lint + test on push)
├── .vscode/                  # shared editor settings (optional, committed)
├── config/                   # declarative config (YAML), pitch coords, team maps
│   └── default.yaml
├── data/                     # GITIGNORED — see section 5
│   ├── raw/                  # vendor drops (StatsBomb, Metrica, SkillCorner, video)
│   ├── interim/              # kloppy-normalized, cached parses
│   ├── processed/            # analytics-ready parquet
│   └── external/             # schemas, reference CSVs
├── docs/
│   ├── architecture/         # this file + sibling design docs
│   ├── 00-executive-summary.md
│   └── ...
├── notebooks/
│   ├── exploration/          # scratch, committed but pre-stripped
│   └── analysis/             # curated, reproducible, committed
├── scripts/                  # one-off CLIs (fetch_statsbomb.py, render_pitch.py)
├── src/
│   └── football_analysis/    # the installable package
│       ├── __init__.py
│       ├── io/               # loaders (statsbombpy, kloppy wrappers)
│       ├── metrics/          # xT, VAEP, pitch control
│       ├── formations/       # role assignment, shape detection
│       ├── cv/               # broadcast → top-down
│       ├── viz/              # mplsoccer builders
│       ├── config.py         # pydantic-settings entry point
│       └── logging.py        # structlog setup
├── tests/
│   ├── unit/                 # mirror src/ tree
│   ├── integration/          # real loader round-trips on fixtures
│   ├── fixtures/             # tiny sample data (≤ 1 MB each)
│   └── conftest.py
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock
```

Rationale: src-layout (not flat) so `import football_analysis` always routes through the installed package — prevents the classic "tests pass locally but fail after install" bug. One package, multiple subpackages by bounded context (aligns with the DDD note in project CLAUDE.md).

---

## 2. Python Environment

**Pick: `uv` + single `pyproject.toml`, Python 3.11.**

Why uv over poetry/pip-tools/conda:
- 10–100× faster resolution; matters once torch + opencv-python + statsbombpy + kloppy + mplsoccer are all in one env.
- Native PEP 621 `pyproject.toml` — no poetry-only `[tool.poetry]` lock-in.
- Built-in `uv run`, `uv tool`, workspace support.
- Excellent Windows support (prebuilt binary, no Rust toolchain required).
- Lockfile (`uv.lock`) is cross-platform and deterministic.
- Conda's only real win is binary channels; uv already pulls prebuilt wheels for opencv/torch on Windows, so conda buys us nothing and costs us a second package system.

**Python 3.11, not 3.12.** PyTorch, some CV wheels, and a handful of analytics libs (notably older socceraction tags) still have rockier 3.12 stories on Windows. 3.11 is the current "everything just works" tier and has a 3-year support runway. Revisit at Python 3.13 LTS feel.

Dependency groups (PEP 735):
- `core` — kloppy, statsbombpy, soccerdata, numpy, pandas, pyarrow, pydantic, pydantic-settings, structlog.
- `analytics` — socceraction, floodlight, scikit-learn, scipy.
- `viz` — mplsoccer, matplotlib, plotly, streamlit.
- `cv` — opencv-python, torch, ultralytics, supervision (roboflow), pillow.
- `dev` — pytest, pytest-cov, pytest-xdist, pytest-randomly, hypothesis, ruff, mypy, pre-commit, ipykernel, jupyterlab, nbstripout.
- `docs` — mkdocs-material, mkdocstrings[python].

Install everything: `uv sync --all-extras --group dev`. Install minimal: `uv sync` (just `core`).

---

## 3. Dev Toolchain

- **Lint + format: ruff only.** No black. Ruff's formatter is black-compatible and one binary is better than two. `ruff check --fix && ruff format`.
- **Type checking: mypy, `strict = true`**, but with `ignore_missing_imports` for the fuzzy analytics deps (kloppy/mplsoccer type stubs are uneven). Pyright is also fine, but mypy integrates better with pre-commit on Windows.
- **Testing: pytest** + `pytest-cov` (coverage), `pytest-xdist` (parallel — `-n auto`), `pytest-randomly` (catch order-dependent tests), `hypothesis` (property tests on geometry/formation code where algebraic invariants matter). Coverage target: **>90% line, branch reporting on.** Fail the build below 90.
- **Pre-commit hooks** (`.pre-commit-config.yaml`):
  - `ruff` (check + format)
  - `mypy` (src only, not tests)
  - `nbstripout` (strip notebook outputs on commit — mandatory for `notebooks/exploration/`)
  - trailing-whitespace, end-of-file-fixer, check-yaml, check-toml, check-added-large-files (max 1 MB)
- **CI: GitHub Actions, one workflow, two jobs** (`lint`, `test`) matrixed on `ubuntu-latest` + `windows-latest`, Python 3.11 only. Windows-in-CI is non-negotiable because the developer runs Windows — we must catch path-separator and encoding issues in PRs, not after. Skip the heavy `cv` group in CI unless a PR touches `src/football_analysis/cv/` (path filter).
- **Task runner: Makefile**, with a `make.bat` shim for cmd users. Git-bash on Windows 11 handles a Makefile natively via the bundled `make` (install `make` via scoop/winget once). Everything goes through `uv run` so nothing depends on an activated venv.
- **Docs: mkdocs-material + mkdocstrings.** Markdown-first, builds from `docs/`. Sphinx is overkill here; pure-markdown loses auto-API. mkdocs strikes the right balance.

---

## 4. Dependency Groups (rationale)

Splitting `cv` from `core`/`analytics` is the key decision. Torch + CUDA pulls are slow and optional — an analyst who wants to run xT pipelines should not wait for torch. The `cv` group stays opt-in until Phase 4 (per the exec summary's phased plan).

`viz` is separate so headless pipelines (CI, batch jobs) don't drag in Streamlit/Plotly.

`docs` is separate so CI's lint/test jobs skip mkdocs.

---

## 5. Data Directory

**Rule: data lives in `./data/` at repo root, and the entire `data/` tree is gitignored** except a `data/.gitkeep` and per-subdir `README.md` describing what belongs there.

Raw vendor data is never committed — even small StatsBomb JSONs — because:
1. Licensing on "open" data still has attribution terms we'd rather handle once.
2. Keeps repo clones fast.
3. Forces every ingestion to go through a script (`scripts/fetch_*.py`) that is reproducible.

Test fixtures (`tests/fixtures/`) are the one committed exception: tiny, stable, hand-trimmed samples (one match, ~100 events), explicitly not for analysis.

Raw video (broadcast clips for CV work) must live outside the repo — configurable via `FA_DATA_DIR` env var (default `./data`). Pydantic-settings reads `.env`:

```
FA_DATA_DIR=D:/football-data
FA_LOG_LEVEL=INFO
```

This keeps the repo portable across a laptop SSD and an external drive.

---

## 6. Sample Files

### `pyproject.toml`

```toml
[project]
name = "football-analysis"
version = "0.1.0"
description = "Top-down tactical analysis of 11v11 soccer: movement, formations, CV pipeline."
requires-python = ">=3.11,<3.12"
readme = "README.md"
license = { text = "MIT" }
dependencies = [
    "kloppy>=3.15",
    "statsbombpy>=1.14",
    "soccerdata>=1.8",
    "numpy>=1.26",
    "pandas>=2.2",
    "pyarrow>=15",
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "structlog>=24.1",
    "typer>=0.12",
]

[project.optional-dependencies]
analytics = ["socceraction>=1.5", "floodlight>=0.4", "scikit-learn>=1.4", "scipy>=1.12"]
viz       = ["mplsoccer>=1.4", "matplotlib>=3.8", "plotly>=5.20", "streamlit>=1.32"]
cv        = ["opencv-python>=4.9", "torch>=2.2", "ultralytics>=8.1", "supervision>=0.19", "pillow>=10.2"]

[dependency-groups]
dev = [
    "pytest>=8.0", "pytest-cov>=5.0", "pytest-xdist>=3.5",
    "pytest-randomly>=3.15", "hypothesis>=6.98",
    "ruff>=0.4", "mypy>=1.10", "pre-commit>=3.7",
    "ipykernel>=6.29", "jupyterlab>=4.1", "nbstripout>=0.7",
]
docs = ["mkdocs-material>=9.5", "mkdocstrings[python]>=0.25"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]
[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "C4", "SIM", "PL", "RUF"]
ignore = ["PLR0913"]  # allow many args in analytics functions

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true
files = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers --cov=football_analysis --cov-branch --cov-report=term-missing --cov-fail-under=90"
markers = ["slow: takes >1s", "integration: hits disk/network fixtures"]

[tool.coverage.run]
source = ["src/football_analysis"]
branch = true
```

### `Makefile`

```makefile
.PHONY: install sync test lint fmt type cov notebook clean ci

install: ; uv sync --all-extras --group dev && uv run pre-commit install
sync:    ; uv sync --all-extras --group dev
fmt:     ; uv run ruff format . && uv run ruff check --fix .
lint:    ; uv run ruff check . && uv run ruff format --check .
type:    ; uv run mypy src
test:    ; uv run pytest -n auto
cov:     ; uv run pytest -n auto --cov-report=html
notebook:; uv run jupyter lab
clean:   ; rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov dist build .coverage
ci:      lint type test
```

### `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.env
.env.*
!.env.example

# Tooling caches
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/
htmlcov/
.coverage
.coverage.*
coverage.xml
dist/
build/

# Editors / OS
.idea/
*.swp
.DS_Store
Thumbs.db

# Data (hard rule — never commit)
data/**
!data/.gitkeep
!data/**/README.md

# Notebooks — keep sources, not outputs
# (outputs stripped by nbstripout pre-commit)

# Models / large binaries
*.pt
*.onnx
*.mp4
*.mov
*.avi
```

### `README.md` (skeleton)

```markdown
# football-analysis
Top-down tactical analysis of 11v11 soccer matches: movement, formations, CV.

## Quick start
    make install
    make test

## Layout
See docs/architecture/infra-layer.md.

## Phases
Phase 0 orientation → Phase 1 events → Phase 2 tracking → Phase 3 formations → Phase 4 CV.
See docs/00-executive-summary.md.

## Data
Set `FA_DATA_DIR` in `.env`. Nothing under `data/` is tracked.
```

---

## 7. Windows Notes

- User runs git-bash. Make sure `make` is installed (`winget install GnuWin32.Make` or via scoop); ship a `make.bat` shim that forwards to `uv run` equivalents for cmd/PowerShell users.
- Always use `pathlib.Path`, never string concatenation with `/`. Enforce via ruff rule `PTH*`.
- UTF-8 everywhere: set `PYTHONUTF8=1` in `.env.example`, and open files with `encoding="utf-8"` explicitly.
- Long-path support: CV asset trees get deep. Enable Win10+ long paths via registry note in `docs/architecture/windows-setup.md` (follow-up doc).
- Line endings: `.gitattributes` with `* text=auto eol=lf` plus `*.ps1 eol=crlf` to avoid ruff complaining about CRLF in pipeline files.
- Avoid `os.fork`-based parallelism; `pytest-xdist` uses spawn on Windows — fine.
- Prebuilt wheels only: pin `opencv-python` (not `opencv-contrib-python-headless` — has GUI issues on Windows display scaling).

---

## 8. Testing Approach

- **Mirror layout**: `tests/unit/metrics/test_xt.py` tests `src/football_analysis/metrics/xt.py`.
- **TDD London school** (per project CLAUDE.md): mock collaborators at module boundaries; unit tests assert on interactions for the IO/config seams, assert on values for the math seams.
- **Hypothesis** for pure-function layers: pitch coordinate transforms, Hungarian role assignment (permutation invariance), pitch control (probabilities sum to 1).
- **Fixtures under `tests/fixtures/`**: one StatsBomb match JSON (~1 MB, trimmed), one Metrica 30-second tracking clip, one 5-frame video snippet for CV smoke tests. Anything bigger goes through `scripts/fetch_*.py` and is gitignored.
- **Integration tests** marked `@pytest.mark.integration`, run by default locally, skipped in the fast CI lane, run in a nightly CI job.
- **Coverage >90% line + branch**, `--cov-fail-under=90` in `pyproject.toml`. Exclude `src/football_analysis/viz/` interactive Streamlit code from coverage (separate smoke test instead).
- **Notebooks are NOT tested**. Code that needs tests moves into `src/` first. Notebooks can import from the installed package and stay thin.

---

## Open questions (tracked, not blockers)

1. Do we want `dvc` for the `data/` tree once Phase 2 tracking data arrives? Probably yes, deferred.
2. GPU torch vs CPU torch — default to CPU wheel in `[cv]`, document a `[cv-gpu]` extra in Phase 4.
3. Whether to adopt `nox` for multi-env testing later. Not now — one supported Python, one supported OS pair.
