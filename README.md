# intelligent_world

Phase 1 scaffold for a minimal ALife runtime focused on deterministic evaluation, mutation safety gates, and structured observability.

## Quickstart
- Install and run checks via `uv`.
- Run tests: `uv run pytest`
- Lint: `uv run ruff check .`
- Format check: `uv run ruff format --check .`

## CLI
- `uv run alife run --task two_sum_sorted --seed 7`
- `uv run alife spike docker-latency`
- `uv run alife spike ast-feasibility`

## Sandbox execution
Evaluator supports two backends:
- `docker` (default): containerized execution boundary for untrusted candidate code
- `process`: local subprocess fallback, mainly for development/testing

Configure via `configs/default.yaml` or `RunConfig`.
