# STRUCTURE.md

## Top-level layout
- `src/alife_core/`: runtime package source
- `tests/`: unit and regression tests
- `scripts/`: spikes/benchmarks invoked by CLI
- `configs/`: runtime configuration files
- `docs/legacy/`: historical planning and design docs
- `.github/workflows/`: CI and repository automation workflows

## Source package structure (`src/alife_core/`)
- `cli.py`: argument parsing and command dispatch
- `runtime.py`: experiment orchestration and lifecycle execution
- `models.py`: typed config and shared data structures
- `tasks/`: built-in task definitions/loading
- `evaluator/`: candidate execution and scoring
- `mutation/`: candidate validation gates
- `agent/`: progression/curriculum lifecycle logic
- `logging/`: structured run/event logging helpers
- `bootstrap/`: initial candidate generation (`static`/`ollama`)

## Naming conventions
- Files/modules: `snake_case`
- Functions/variables: `snake_case`
- Classes/dataclasses: `PascalCase`
- Constants: `UPPER_CASE`

## Import patterns
- Use absolute imports within package: `from alife_core...`
- Avoid circular imports by keeping leaf concerns isolated (`models`, `tasks`, `logging`).

## File placement rules
- New runtime logic goes under `src/alife_core/<domain>/`.
- New tests mirror runtime concerns under `tests/test_<domain>.py`.
- Non-production experiments belong in `scripts/`, not in runtime modules.
- Long-form historical or exploratory docs belong in `docs/legacy/`.

## Architectural boundaries
- `cli.py` should remain thin: parse args + dispatch only.
- `runtime.py` owns experiment flow, lifecycle transitions, and high-level policy checks.
- `evaluator/` owns sandbox execution and correctness scoring.
- `mutation/` owns validation gates; do not embed gate logic in CLI.

## Generated artifacts
- Runtime artifacts:
  - `logs/`
  - `organisms/`
  - `sweep_runs/`
  - optional sweep outputs (for example `sweep_summary.json`)
- These should not be committed and are ignored via `.gitignore`.
