# intelligent_world

A minimal Artificial Life (ALife) runtime for evolving Python solutions on small coding tasks under explicit safety constraints.

## What this project does
- Evaluates candidate code against train/hidden tests.
- Mutates and selects candidates in either single-agent or population mode.
- Runs untrusted code in Docker by default (`sandbox_backend: docker`).
- Logs experiment events as JSONL for reproducibility and analysis.

## Quickstart
### Prerequisites
- Python 3.12+
- `uv`
- Docker daemon (required for default sandbox)
- Optional: `ollama` CLI when using `bootstrap_backend: ollama`

### Setup
```bash
uv sync --group dev
```

### Validate
```bash
uv run --group dev pytest
uv run --group dev ruff check .
uv run --group dev ruff format --check .
```

## Usage
### Run an experiment
```bash
uv run alife run --task two_sum_sorted --seed 7
```

### Run population mode
```bash
uv run alife run \
  --task two_sum_sorted \
  --seed 7 \
  --population \
  --population-size 12 \
  --elite-count 3 \
  --max-generations 30 \
  --population-workers 4
```

### Use Ollama bootstrap
```bash
uv run alife run --task two_sum_sorted --seed 7 --bootstrap-backend ollama --ollama-model gpt-oss:20b
```

### Spikes
```bash
uv run alife spike docker-latency
uv run alife spike ast-feasibility
uv run alife spike schedule-curve
uv run alife spike parameter-sweep
uv run alife spike parameter-sweep --sweep-output sweep_summary.json
uv run alife spike metrics-report --log-path logs/<run-id>.jsonl
```

## Event schema
- Runtime logs use JSONL schema v2 envelopes with:
  - `schema_version`
  - `event_type`
  - `timestamp`
  - `run_id`
  - `mode`
  - `task`
  - `step`
  - `payload`
- Population runs emit `generation.metrics` events with diversity, entropy, novelty, lineage depth, and AST complexity fields.

## Safety model
- Default boundary is Docker with restricted runtime settings.
- `process` backend is available for local development/testing only.
- Explicit opt-in is required for unsafe combinations:
```bash
uv run alife run --task two_sum_sorted --unsafe-process-backend
uv run alife spike parameter-sweep --unsafe-process-backend
```

## Repository docs
- `AGENTS.md`: instructions for coding agents and contributors.
- `PRODUCT.md`: product purpose and objectives.
- `TECH.md`: stack and technical constraints.
- `STRUCTURE.md`: file/module organization and conventions.
- `docs/legacy/`: historical planning/research docs.

## CI
GitHub Actions (`.github/workflows/ci.yml`) runs:
- tests (`pytest`)
- lint (`ruff check`)
- format check (`ruff format --check`)
