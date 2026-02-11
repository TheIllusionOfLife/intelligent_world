# AGENTS.md

This file defines repository-specific instructions for coding agents and human contributors.

## Scope
These rules apply to the entire repository unless a deeper `AGENTS.md` overrides them.

## Fast command reference
Use `uv` for all Python workflows.

```bash
uv sync --group dev
uv run --group dev pytest
uv run --group dev ruff check .
uv run --group dev ruff format --check .
uv run alife run --task two_sum_sorted --seed 7
uv run alife spike docker-latency
uv run alife spike ast-feasibility
uv run alife spike schedule-curve
uv run alife spike parameter-sweep
```

Commands agents often miss:
- Population mode run:
```bash
uv run alife run \
  --task two_sum_sorted \
  --population \
  --population-size 12 \
  --elite-count 3 \
  --max-generations 30 \
  --population-workers 4
```
- Unsafe process backend opt-in (must be explicit):
```bash
uv run alife run --task two_sum_sorted --unsafe-process-backend
uv run alife spike parameter-sweep --unsafe-process-backend
```
- Sweep artifact output:
```bash
uv run alife spike parameter-sweep --sweep-output sweep_summary.json
```

## Code style and implementation rules
- Python 3.12+, 4-space indentation, full type hints for new/edited code.
- Keep modules small and cohesive; prefer pure/deterministic functions.
- Naming:
  - `snake_case` for modules/functions/variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants
- Keep runtime behavior fail-fast and explicit; avoid silent fallbacks except where already designed.
- Do not add new dependencies unless clearly necessary.

## Testing instructions
- Primary test runner: `pytest`
- Always run before finishing changes:
  - `uv run --group dev pytest`
  - `uv run --group dev ruff check .`
  - `uv run --group dev ruff format --check .`
- Bug fixes must include a regression test first when practical.
- Keep tests deterministic and fast. Prefer `sandbox_backend="process"` in tests unless validating Docker-specific behavior.

## Repository etiquette
- Branch naming:
  - `feat/<short-description>`
  - `fix/<short-description>`
  - `docs/<short-description>`
  - `chore/<short-description>`
- Never push directly to `main`.
- Keep commits logically scoped and message style imperative sentence case.
- PRs should include:
  - purpose
  - key files changed
  - validation commands and results
  - explicit out-of-scope notes

## Architecture decisions to preserve
- Runtime package lives under `src/alife_core/`.
- Evaluation supports `docker` (default) and `process` backends.
- `docker` is the default safety boundary for untrusted code evaluation.
- Bootstrapping supports `static` and `ollama`; `ollama` is optional and must degrade safely.
- JSONL event logging in `logs/` is part of runtime observability and reproducibility.
- Canonical technical design history is in `docs/legacy/technical_design_spec.md`.

## Environment and tooling quirks
- Required tools: `uv`, Python 3.12+, Docker daemon.
- Optional tool: `ollama` CLI for model-based seed generation.
- No required environment variables for core local execution.
- If Docker is unavailable, default Docker-backed runs fail fast by design.

## Common gotchas
- `process` backend is intentionally gated for safety in some flows.
- `max_generations` counts reproduction rounds; evaluation includes generation `0` baseline.
- `population_workers` is capped at runtime by CPU count and `population_size`.
- Use repository-relative output paths for sweep artifacts to avoid path validation errors.

## Directory intent
- Root: active project entry docs and build metadata.
- `docs/legacy/`: historical planning/review/spec documents (not day-to-day onboarding docs).
- `scripts/`: research spikes and benchmarks callable through the CLI.
- `tests/`: deterministic unit/regression tests.
