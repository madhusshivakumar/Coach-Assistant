.PHONY: install sync test lint fmt type cov app notebook clean ci help

help:
	@echo "install   - uv sync + install pre-commit hooks"
	@echo "sync      - uv sync all extras + dev group"
	@echo "fmt       - ruff format + autofix"
	@echo "lint      - ruff check (no fix)"
	@echo "type      - mypy src"
	@echo "test      - pytest (parallel)"
	@echo "cov       - pytest with HTML coverage report"
	@echo "app       - launch Streamlit dashboard"
	@echo "notebook  - launch JupyterLab"
	@echo "clean     - remove caches"
	@echo "ci        - lint + type + test (matches CI)"

install:
	uv sync --all-extras --group dev
	uv run pre-commit install

sync:
	uv sync --all-extras --group dev

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .

type:
	uv run mypy src

test:
	uv run pytest -n auto

cov:
	uv run pytest -n auto --cov-report=html

app:
	uv run streamlit run src/football_analysis/app/main.py

notebook:
	uv run jupyter lab

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov dist build .coverage .coverage.*

ci: lint type test
