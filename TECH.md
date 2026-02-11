# TECH.md

## Language and runtime
- Python 3.12+

## Package/dependency management
- `uv` for dependency sync and command execution.
- Dependencies declared in `pyproject.toml`; lockfile is `uv.lock`.

## Core libraries/tools
- `pyyaml`: config loading
- `pytest`: tests
- `ruff`: lint + formatting checks

## Application entrypoints
- CLI script: `alife` (configured in `pyproject.toml`)
- Main CLI module: `src/alife_core/cli.py`
- Metrics summary spike: `scripts/metrics_report.py` (`alife spike metrics-report --log-path ...`)

## Execution and safety constraints
- Candidate code evaluation backends:
  - `docker` (default, safety boundary)
  - `process` (local fallback, explicitly gated in unsafe paths)
- Docker runner is configured with restricted flags (network disabled, caps dropped, resource limits).

## Configuration
- Default runtime config: `configs/default.yaml`
- Config schema/data model: `src/alife_core/models.py` (`RunConfig`)
- Population metrics/convergence knobs:
  - `novelty_k`
  - `convergence_patience`
  - `convergence_entropy_floor`
  - `convergence_fitness_delta_floor`

## Quality gates
- Tests: `uv run --group dev pytest`
- Lint: `uv run --group dev ruff check .`
- Format check: `uv run --group dev ruff format --check .`
- CI workflow: `.github/workflows/ci.yml`

## Technical constraints
- Keep code deterministic wherever possible (seed-driven flows).
- Prefer standard library + existing dependencies.
- Do not bypass safety checks around sandbox backend selection.
- Keep spike scripts runnable both directly and via CLI wrappers.
- JSONL logs use schema version `2` envelopes with event-specific data nested in `payload`.
