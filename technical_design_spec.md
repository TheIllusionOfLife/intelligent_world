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
- **Success criteria**: Emergent strategy diversity — different runs produce qualitatively different solutions to the same problem. Measured by:
  - **Solution cluster count**: Number of distinct solution strategies across N runs (clustered by AST structural similarity). Target: ≥ 3 clusters across 10 runs.
  - **Hidden pass ratio distribution**: Mean and variance of hidden test performance across runs. Healthy diversity shows spread, not collapse to one mode.
  - **Lifespan distribution**: Mean and variance of agent survival steps. Target: at least 50% of runs survive ≥ 100 steps.
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

**Tiebreak rule**: When two candidates have identical fitness, prefer the one with fewer AST nodes (simpler code).

**Weight scaling**: `w2` (edit_cost penalty) decays over time to avoid suppressing beneficial large mutations in later stages:

```
w2_effective = w2 * decay_factor ^ step
```

`decay_factor` (e.g., 0.999) is configurable. This ensures early evolution favors small safe changes while later evolution permits larger structural shifts.

### Anti-Goodhart: Train/Hidden test split

- Each task has two test sets:
  - **Train tests**: Visible to the evaluation loop. Used for fitness calculation.
  - **Hidden tests**: Never seen during evolution. Used for post-hoc generalization assessment.
- Hidden test results are logged but do NOT affect fitness or energy in Phase 1.
- **Goodhart detection (monitoring only)**: If `hidden_pass_ratio` drops below `train_pass_ratio` by more than a configurable threshold (e.g., 0.2), the event is flagged as `"goodhart_warning": true` in the log. No automated intervention in Phase 1.
- **Phase 2 escalation**: Introduce energy penalty or forced rollback when hidden degradation exceeds threshold. Design TBD based on Phase 1 observations.

### Determinism guarantee

- No wall-clock time in fitness
- AST diff is deterministic
- Test execution is deterministic (pure functions, no I/O, no randomness in tasks)

### Reproducibility requirements

Each run MUST record the following metadata in the first JSONL event (`type: "run_start"`):

- `random_seed`: Fixed seed for all stochastic operations (mutation selection, SA acceptance)
- `docker_image_digest`: SHA256 digest of the sandbox Docker image
- `python_version`: Exact Python version inside the container (e.g., `3.12.1`)
- `cpu_architecture`: Host CPU architecture (e.g., `arm64`, `x86_64`)
- `framework_git_sha`: Git commit hash of the alife framework
- `parameters`: Full snapshot of all configurable parameters (weights, thresholds, cooling rate, etc.)

Without these fields, a run is considered non-reproducible and should be excluded from comparative analysis.

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

- Better mutations (`new_fitness >= old_fitness`): always accepted
- Worse mutations: accepted with probability `exp(-loss / temperature)` where:
  ```
  loss = max(0, old_fitness - new_fitness)
  ```
  This ensures `loss` is always non-negative, so the exponent is always ≤ 0, and acceptance probability is always in (0, 1]. Larger degradations yield lower acceptance probability.
- Temperature decreases on a configurable schedule (linear or exponential decay)

---

## 5. Task Design

### Sequential curriculum

Tasks are presented in order of increasing difficulty. Agent must satisfy **both** `pass_ratio_threshold` (test correctness gate) and `fitness_threshold` (overall quality gate) on the current task before unlocking the next. This prevents advancing with high pass rate but degenerate code structure.

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

**Layer 1: Static analysis (AST inspection before execution)**

- No I/O operations (file, network, stdin)
- No imports beyond a whitelist (e.g., `math`, `collections`, `itertools`)
- No `exec`, `eval`, `__import__`, `open`, `os`, `sys`

AST inspection alone is insufficient — obfuscation paths (attribute access chains, existing object traversal, exception-based escapes) can bypass static checks.

**Layer 2: Runtime containment (Docker-level enforcement)**

These are the primary security boundary. Even if static analysis is bypassed, runtime limits contain damage:

- `--network=none`: No network access
- `--read-only` with explicit tmpdir mount for working directory only
- `--memory` / `--cpus`: Resource caps
- `--pids-limit`: Prevent fork bombs
- `--security-opt=no-new-privileges`: Prevent privilege escalation
- Timeout enforced at subprocess level (kill container on exceed)

**Threat model**: Organism code is treated as untrusted input. Static analysis is a fast-reject filter; Docker runtime isolation is the actual security boundary.

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
| `pass_ratio_threshold` | Minimum train pass_ratio to unlock next task | 0.9 | Experiment |
| `fitness_threshold` | Minimum overall fitness to unlock next task (pass_ratio - edit_cost) | 0.8 | Experiment |
| `mutation_stagnation_window` | Steps before increasing mutation size | 10 | Experiment |
