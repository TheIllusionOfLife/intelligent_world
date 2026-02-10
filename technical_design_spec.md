# Technical Design Spec: Minimal ALife System

> **Document role**: Actionable design spec — concrete decisions made after considering the initial proposition and the research review.
>
> **Inputs**:
> - [`minimal_alife_project_overview.md`](./minimal_alife_project_overview.md) — The founding vision and conceptual design
> - [`unified_review.md`](./unified_review.md) — Research review that identified critical gaps and design contradictions
>
> **How this was produced**: A structured interview clarified all open questions raised by the review (mutation engine, fitness function, sandboxing, population model, etc.). Each decision is documented below with rationale.

---

## 1. Project Scope & Goals

- **Type**: Engineering demo (working system first, academic write-up later)
- **Success criteria**: Emergent strategy diversity — different runs produce qualitatively different solutions to the same problem
- **Phased approach**:
  - **Phase 1**: Single-agent evolution loop, working end-to-end
  - **Phase 2**: Population of competing agents with selection/diversity dynamics

---

## 2. Mutation Engine

### Phase 1: AST-based programmatic mutations

The mutation engine operates on Python AST nodes. No LLM API calls during evolution.

- **Adaptive mutation size**: Start with 1–2 node changes; increase mutation scope when fitness stagnates over a configurable window
- **Mutation types**: Constant tweaks, operator swaps, guard insertion/removal, statement reordering, expression substitution
- **No interpretability requirement**: Mutations are not logged with rationale — only the code diff and fitness delta are recorded

### Bootstrap: Local LLM (one-time)

- **Tool**: Ollama (already installed)
- **Model**: User-specified model via Ollama
- **Purpose**: Generate initial seed implementation from blank function signature
- **Scope**: Called once per task to produce the starting organism. Not called during evolution loop.

---

## 3. Evaluation Function

### Fitness formula

```
Fitness = w1 * pass_ratio − w2 * ast_edit_cost
```

- **pass_ratio**: Fraction of tests passed (train set only visible to agent)
- **ast_edit_cost**: Number of AST nodes added/removed/changed from previous generation
- **No time component**: `time_seconds` dropped entirely to satisfy determinism requirement

Weights `w1`, `w2` are configurable. Initial values TBD via sensitivity analysis in early experiments.

### Anti-Goodhart: Train/Hidden test split

- Each task has two test sets:
  - **Train tests**: Visible to the evaluation loop. Used for fitness calculation.
  - **Hidden tests**: Never seen during evolution. Used for post-hoc generalization assessment.
- Hidden test results are logged but do NOT affect fitness or energy.

### Determinism guarantee

- No wall-clock time in fitness
- AST diff is deterministic
- Test execution is deterministic (pure functions, no I/O, no randomness in tasks)

---

## 4. Agent Life Cycle

### Energy model

- Initial energy: `1.0` (scale TBD via experimentation)
- Per-step: base survival cost reduces energy; fitness improvement increases energy; fitness degradation reduces energy
- **Death**: Energy ≤ 0, or no improvement for N steps (configurable), or resource budget exceeded

### Death is permanent (Phase 1)

- When an agent dies, the experiment run ends
- No respawn. Restart is manual.
- In Phase 2 (population), other agents continue when one dies.

### Acceptance policy: Simulated annealing

- Better mutations: always accepted
- Worse mutations: accepted with probability `exp(-Δfitness / temperature)`
- Temperature decreases on a configurable schedule (linear or exponential decay)

---

## 5. Task Design

### Sequential curriculum

Tasks are presented in order of increasing difficulty. Agent must reach a fitness threshold on the current task before unlocking the next.

### Initial task set

1. **two_sum_sorted** — Find indices of two numbers summing to target (sorted input)
2. **run_length_encode** — Compress string via run-length encoding
3. **slugify** — Convert text to URL-safe slug

### Initial state: Blank slate + LLM bootstrap

- Agent starts with only the function signature
- Local LLM generates the first seed implementation
- All subsequent evolution is AST-mutation only

---

## 6. Execution Sandbox

### Phase 1: Lightweight Docker wrapper

- Each evaluation runs via `docker run` subprocess call
- Resource limits: CPU time, memory, no network access, read-only filesystem (except working dir)
- Timeout enforced at both Docker and subprocess level
- No direct `docker-py` dependency — plain subprocess calls to `docker` CLI

### Constraints on organism code

- No I/O operations (file, network, stdin)
- No imports beyond a whitelist (e.g., `math`, `collections`, `itertools`)
- No `exec`, `eval`, `__import__`, `open`, `os`, `sys`
- Violations detected via AST inspection before execution

---

## 7. Observability

### Structured JSON event log

Each step emits a JSON event:

```json
{
  "step": 42,
  "timestamp": "2025-02-08T12:34:56Z",
  "energy": 0.73,
  "fitness": 0.85,
  "fitness_delta": 0.02,
  "mutation_type": "operator_swap",
  "mutation_nodes_changed": 2,
  "accepted": true,
  "train_pass_ratio": 0.85,
  "hidden_pass_ratio": 0.80,
  "code_hash": "abc123",
  "task": "two_sum_sorted"
}
```

- Log file: `logs/{run_id}.jsonl`
- Post-hoc analysis via Jupyter notebooks or scripts
- No live dashboard in Phase 1

### Archive

- Every accepted mutation's code is saved to `organisms/archive/{step}.py`
- Current best is always at `organisms/current/{task_name}.py`

---

## 8. Project Structure

```
alife_min/
  pyproject.toml
  README.md
  Dockerfile              # Sandbox image for organism evaluation

  src/
    alife/
      tasks/              # Task definitions + train/hidden test sets
        two_sum_sorted/
        run_length_encode/
        slugify/
      evaluator/          # Fitness calculation, test runner, sandbox wrapper
      evolution/          # AST mutation engine, simulated annealing, energy model
      bootstrap/          # Ollama integration for initial seed generation
      logging/            # JSON event logger

  tests/                  # Framework tests (not organism tests)

  organisms/
    current/              # Current best solutions per task
    archive/              # Historical generations

  logs/                   # JSONL run logs

  notebooks/              # Post-hoc analysis
```

---

## 9. Decisions Deferred to Phase 2

| Topic | Phase 1 Status | Phase 2 Plan |
|-------|---------------|--------------|
| Population dynamics | Single agent | Multiple agents with selection, crossover, diversity maintenance |
| Diversity maintenance | Not applicable | Novelty Search, MAP-Elites, or entropy regularization |
| Respawn on death | No respawn | Respawn from best ancestor or crossover of survivors |
| Property-based testing | Train/hidden split | Add Hypothesis-style generative tests |
| Live dashboard | JSON logs only | Real-time web dashboard |
| Academic rigor | Demo-level | Formal hypotheses, baselines, statistical tests, prior work comparison |

---

## 10. Open Parameters (To Be Determined Empirically)

| Parameter | Description | Initial Guess | Method |
|-----------|-------------|---------------|--------|
| `w1` (pass_ratio weight) | Importance of test passing | 0.9 | Sensitivity analysis |
| `w2` (edit_cost weight) | Penalty for large code changes | 0.1 | Sensitivity analysis |
| `base_survival_cost` | Energy drain per step | 0.01 | Tune to target ~100-step runs |
| `improvement_reward` | Energy gain on fitness increase | proportional to Δfitness | Experiment |
| `N_stagnation` | Steps without improvement before death | 20 | Experiment |
| `initial_temperature` | SA starting temperature | 1.0 | Standard SA tuning |
| `cooling_rate` | SA temperature decay | 0.995 per step | Standard SA tuning |
| `fitness_threshold` | Score needed to unlock next task | 0.9 pass_ratio | Experiment |
| `mutation_stagnation_window` | Steps before increasing mutation size | 10 | Experiment |
