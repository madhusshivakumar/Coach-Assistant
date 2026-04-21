@echo off
REM Windows cmd/PowerShell shim for the Makefile targets.
REM Keeps parity with the bash-friendly Makefile for non-bash users.

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="install" goto install
if "%1"=="sync" goto sync
if "%1"=="fmt" goto fmt
if "%1"=="lint" goto lint
if "%1"=="type" goto type
if "%1"=="test" goto test
if "%1"=="cov" goto cov
if "%1"=="app" goto app
if "%1"=="notebook" goto notebook
if "%1"=="ci" goto ci

echo Unknown target: %1
goto help

:help
echo install   - uv sync + install pre-commit hooks
echo sync      - uv sync all extras + dev group
echo fmt       - ruff format + autofix
echo lint      - ruff check
echo type      - mypy src
echo test      - pytest parallel
echo cov       - pytest with HTML coverage
echo app       - launch Streamlit dashboard
echo notebook  - launch JupyterLab
echo ci        - lint + type + test
goto end

:install
uv sync --all-extras --group dev
uv run pre-commit install
goto end

:sync
uv sync --all-extras --group dev
goto end

:fmt
uv run ruff format .
uv run ruff check --fix .
goto end

:lint
uv run ruff check .
uv run ruff format --check .
goto end

:type
uv run mypy src
goto end

:test
uv run pytest -n auto
goto end

:cov
uv run pytest -n auto --cov-report=html
goto end

:app
uv run streamlit run src/football_analysis/app/main.py
goto end

:notebook
uv run jupyter lab
goto end

:ci
call :lint
if errorlevel 1 exit /b 1
call :type
if errorlevel 1 exit /b 1
call :test
goto end

:end
