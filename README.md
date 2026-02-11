# intelligent_world

Phase 1 scaffold for a minimal ALife runtime focused on deterministic evaluation, mutation safety gates, and structured observability.

## Quickstart
- Sync dependencies: `uv sync --group dev`
- Run tests: `uv run --group dev pytest`
- Lint: `uv run --group dev ruff check .`
- Format check: `uv run --group dev ruff format --check .`

## CLI
- `uv run alife run --task two_sum_sorted --seed 7`
- `uv run alife run --task two_sum_sorted --seed 7 --bootstrap-backend ollama --ollama-model gpt-oss:20b`
- `uv run alife spike docker-latency`
- `uv run alife spike ast-feasibility`
- `uv run alife spike parameter-sweep`
- `uv run alife spike parameter-sweep --sweep-output sweep_summary.json`

## Sandbox execution
Evaluator supports two backends:
- `docker` (default): containerized execution boundary for untrusted candidate code
- `process`: local subprocess fallback, mainly for development/testing

Configure via `configs/default.yaml` or `RunConfig`.
