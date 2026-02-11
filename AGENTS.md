# Repository Guidelines

## Project Structure & Module Organization
This repository is currently documentation-first. Core files at the root:
- `minimal_alife_project_overview.md`: initial concept and vision
- `unified_review.md`: consolidated critique and gap analysis
- `technical_design_spec.md`: implementation-ready system design

When implementation begins, follow the structure defined in `technical_design_spec.md`: `src/alife_core/` for runtime code, `tests/` for framework tests, `configs/` for YAML settings, and `logs/` for run outputs.

## Build, Test, and Development Commands
There is no build pipeline committed yet. Use these commands for current workflows:
- `rg --files`: quick repository file index
- `rg -n "pattern" *.md`: search design decisions across docs
- `git log --oneline`: review historical decisions and commit style

When Python code is added, use `uv` for environment and execution (not global installs), and add explicit project commands here (for example `uv run pytest`).

## Coding Style & Naming Conventions
For documentation updates:
- Use clear, sectioned Markdown with concise paragraphs
- Prefer relative links (example: `./technical_design_spec.md`)
- Keep terminology consistent (`fitness`, `pass_ratio`, `hidden tests`, `mutation`)

For future Python modules (per design spec):
- 4-space indentation, type hints, and focused modules
- `snake_case` for functions/files, `PascalCase` for classes, `UPPER_CASE` for constants
- Use `ruff` for linting/formatting once configured

## Testing Guidelines
No automated test suite is committed yet. Until code exists, validate by consistency checks across documents (scope, metrics, and directory references).

Once `tests/` is added:
- Use `pytest`
- Name files `test_*.py`
- Keep unit tests deterministic and fast
- Cover evaluator logic, mutation validation gates, and task definitions from the design spec

## Commit & Pull Request Guidelines
Commit style in history is concise, imperative, and sentence-case (example: `Add unified review of minimal ALife project overview`).

Follow these rules:
- Create a topic branch (`docs/...`, `feat/...`, `fix/...`) before changes
- Keep commits logically scoped
- In PRs, include: purpose, key file changes, and any spec sections affected
- Link related issues/tasks and include screenshots only when visual artifacts are introduced
