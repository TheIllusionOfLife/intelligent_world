# Repository Guidelines

## Project Structure & Module Organization
This repository now includes both planning docs and a Python runtime scaffold.
- Root docs: `minimal_alife_project_overview.md`, `unified_review.md`, `technical_design_spec.md`
- Runtime package: `src/alife_core/`
- Tests: `tests/`
- Config: `configs/default.yaml`
- Spikes/benchmarks: `scripts/`

Follow `technical_design_spec.md` when adding new runtime modules (`agent/`, `evaluator/`, `mutation/`, `tasks/`, `logging/`).

## Build, Test, and Development Commands
Use `uv` for Python workflows (no global installs).
- `uv run pytest`: run unit tests
- `uv run ruff check .`: lint
- `uv run ruff format --check .`: formatting check
- `uv run alife run --task two_sum_sorted --seed 7`: run CLI entrypoint
- `uv run alife spike docker-latency`: run Docker latency spike
- `uv run alife spike ast-feasibility`: run AST feasibility spike

Useful exploration commands:
- `rg --files`
- `rg -n "pattern" src tests`

## Coding Style & Naming Conventions
- Python: 4-space indentation, type hints, focused modules
- Naming: `snake_case` for functions/files, `PascalCase` for classes, `UPPER_CASE` for constants
- Keep modules cohesive and deterministic where possible
- Use `ruff` as the formatter/linter baseline

## Testing Guidelines
With `tests/` present:
- Use `pytest`
- Name test files `test_*.py`
- Keep tests deterministic and fast
- Cover evaluator logic, mutation validation gates, task handling, and lifecycle rules
- Add regression tests for every bug fix before implementation changes

## Commit & Pull Request Guidelines
Commit style in history is concise, imperative, and sentence-case (example: `Implement Phase 1 MVP core runtime and tooling`).

Before opening a PR:
- Create a topic branch (`docs/...`, `feat/...`, `fix/...`)
- Keep commits logically scoped
- Ensure lint + format check + tests are green
- In PR body, include purpose, key files changed, validation commands, and scope boundaries
