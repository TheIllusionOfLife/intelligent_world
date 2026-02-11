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
  - **Phase 1**: Single-agent evolution loop, working end-to-end. **Honest scope**: Phase 1 is a deterministic code optimizer with survival pressure — not yet an ALife system. Without population dynamics, crossover, or semantic mutation, it cannot exhibit true evolutionary behavior. Its purpose is to validate the infrastructure (sandbox, evaluation, logging, energy model) that Phase 2 builds upon.
  - **Phase 2**: Population of competing agents with selection, crossover, and diversity maintenance. This is where ALife-like dynamics (strategy divergence, niche formation, arms races) become possible.

---

## 2. Mutation Engine

### Phase 1: AST-based programmatic mutations

The mutation engine operates on Python AST nodes. No LLM API calls during evolution.

- **Implementation**: Use Python's built-in `ast` module for parsing/unparsing and `LibCST` for concrete syntax tree transformations that preserve formatting. `ast` handles structural analysis; `LibCST` handles mutations that need to produce readable output.
- **Adaptive mutation size**: Start with 1–2 node changes; increase mutation scope when fitness stagnates over a configurable window
- **Mutation types**: Constant tweaks, operator swaps, guard insertion/removal, statement reordering, expression substitution
- **No interpretability requirement**: Mutations are not logged with rationale — only the code diff and fitness delta are recorded

**Mutation validation gate**: Every mutation MUST pass through a validation pipeline before reaching the evaluator:
1. `ast.parse()` — reject if syntactically invalid (zero-cost check)
2. `compile()` — reject if not compilable
3. AST static analysis — reject if violates sandbox constraints (Section 6)

Only mutations surviving all three gates are sent to the Docker evaluator. Failed mutations are logged as `"mutation_rejected": true` with the rejection stage, but do NOT cost energy or count toward stagnation.

**Feasibility risk & fallback**: AST mutations on Python code have an unverified viability rate. If the viable-mutation-to-improvement ratio is too low (<5%), agents will die before meaningful evolution occurs.

**Required pre-implementation spike**: Before building the mutation engine, run a standalone feasibility test — apply 100 random AST mutations to a simple function and measure:
- (a) Syntactic validity rate
- (b) Semantic difference rate (output changes)
- (c) Fitness improvement rate

**Fallback strategy** (activate if viability rate <5%):
- **Tier 1**: Add a syntax repair pass (re-parse, auto-fix common errors) between mutation and validation
- **Tier 2**: Supplement with template-based mutations — a library of parameterized, guaranteed-valid code transformations (e.g., "replace `for` loop with `while`", "add boundary check for empty input")
- **Tier 3**: Reintroduce local LLM for mutation (not just bootstrap) if programmatic approaches prove insufficient

### Bootstrap: Local LLM (one-time)

- **Tool**: Ollama (already installed)
- **Model**: User-specified model via Ollama
- **Purpose**: Generate initial seed implementation from blank function signature
- **Scope**: Called once per task to produce the starting organism. Not called during evolution loop.

**Bootstrap simplicity constraint**: The LLM prompt MUST request a naive, minimal implementation (e.g., brute-force, no clever optimizations, simple control flow). Sophisticated LLM output creates a "brain damage" risk — AST mutations are blind local edits that will likely destroy complex semantic structure immediately. The seed should be code that AST mutations can plausibly improve, not code that is already near-optimal.

Example prompt template:
```
Write the simplest possible Python implementation of {function_signature}.
Use only basic loops and conditionals. No helper functions, no imports,
no optimizations. Prioritize readability over efficiency.
```

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

**Weight scaling**: `w2` (edit_cost penalty) decays over time to avoid suppressing beneficial large mutations in later stages, but never below a floor to prevent code bloat (a classic Genetic Programming problem where junk code accumulates unchecked):

```
w2_effective = max(w2_floor, w2 * decay_factor ^ step)
```

- `decay_factor` (e.g., 0.999) is configurable.
- `w2_floor` (e.g., 0.02) ensures a minimum edit cost penalty always applies, preventing unbounded code growth.

This ensures early evolution favors small safe changes, later evolution permits larger structural shifts, but junk accumulation is always penalized.

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
- Per-step energy update (additive):
  ```
  energy -= base_survival_cost                          # always drained
  energy += improvement_multiplier * max(0, Δfitness)   # reward on improvement
  energy -= degradation_multiplier * max(0, -Δfitness)  # penalty on regression
  ```
  Where `Δfitness = new_fitness - old_fitness`. Both multipliers are configurable. `improvement_multiplier` and `degradation_multiplier` default to `1.0` — meaning a +0.05 fitness gain restores 0.05 energy, and a -0.05 regression costs 0.05 energy, independent of current energy level.
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

**Stability constraint (SA cooling vs w2 decay interaction)**: SA temperature decay makes the agent increasingly intolerant of fitness regression, while w2 decay encourages larger structural changes. These forces move in opposite directions — the system becomes simultaneously more accepting of big edits AND less forgiving of bad outcomes. This creates a potential "death cliff" where one bad large mutation is fatal.

To mitigate:
- **Coupled schedules**: w2 decay MUST NOT outpace SA cooling. Formally: at any step, if w2 has decayed by X%, temperature should have decayed by at most X%. This ensures the agent never permits large mutations it cannot survive failing.
- **Pre-implementation visualization**: Before running experiments, plot both curves (w2_effective and temperature) over 500 steps with the chosen parameters. Verify visually that the w2 curve never leads the temperature curve by more than one phase.

---

## 5. Task Design

### Sequential curriculum

Tasks are presented in order of increasing difficulty. Agent must satisfy **both** `pass_ratio_threshold` (test correctness gate) and `fitness_threshold` (overall quality gate) on the current task before unlocking the next. This prevents advancing with high pass rate but degenerate code structure.

### Initial task set

#### Task 1: two_sum_sorted

```python
def two_sum_sorted(numbers: list[int], target: int) -> tuple[int, int]:
    """Return 1-based indices of two numbers that sum to target. Input is sorted ascending."""
```

Train tests:
```python
assert two_sum_sorted([2, 7, 11, 15], 9) == (1, 2)
assert two_sum_sorted([1, 2, 3, 4, 5], 8) == (3, 5)
assert two_sum_sorted([-3, -1, 0, 4, 7], 3) == (1, 5)
assert two_sum_sorted([1, 1, 1, 1, 5], 6) == (1, 5)
assert two_sum_sorted([1, 2], 3) == (1, 2)
assert two_sum_sorted([-5, -3, 0, 2, 8], -8) == (1, 2)
assert two_sum_sorted([1, 3, 5, 7, 9, 11], 12) == (1, 6)
assert two_sum_sorted([0, 0, 3, 4], 0) == (1, 2)
assert two_sum_sorted([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 19) == (9, 10)
assert two_sum_sorted([10, 20, 30, 40, 50], 70) == (2, 5)
```

Hidden tests:
```python
assert two_sum_sorted([1, 5, 8, 11, 14], 19) == (2, 5)
assert two_sum_sorted([-10, -5, 0, 5, 10], 0) == (1, 5)
assert two_sum_sorted([3, 3], 6) == (1, 2)
assert two_sum_sorted([1, 4, 6, 8, 10, 12], 14) == (2, 5)
assert two_sum_sorted(list(range(1, 101)), 101) == (1, 100)
```

#### Task 2: run_length_encode

```python
def run_length_encode(s: str) -> str:
    """Compress string using run-length encoding. Single chars have no count prefix."""
```

Train tests:
```python
assert run_length_encode("aabbbcccc") == "a2b3c4"
assert run_length_encode("abc") == "abc"
assert run_length_encode("") == ""
assert run_length_encode("aaaa") == "a4"
assert run_length_encode("a") == "a"
assert run_length_encode("aabb") == "a2b2"
assert run_length_encode("aaabba") == "a3b2a"
assert run_length_encode("zzzzzzzzzz") == "z10"
assert run_length_encode("abababab") == "abababab"
assert run_length_encode("aaaaabbbbbccccc") == "a5b5c5"
```

Hidden tests:
```python
assert run_length_encode("xxyyxxyyxx") == "x2y2x2y2x2"
assert run_length_encode("aaaaaaaaaaaa") == "a12"
assert run_length_encode("abcabcabc") == "abcabcabc"
assert run_length_encode("aaabbbccc") == "a3b3c3"
assert run_length_encode("z") == "z"
```

#### Task 3: slugify

```python
def slugify(text: str) -> str:
    """Convert text to URL-safe slug: lowercase, alphanumeric + hyphens, no leading/trailing/double hyphens."""
```

Train tests:
```python
assert slugify("Hello World") == "hello-world"
assert slugify("  Hello   World  ") == "hello-world"
assert slugify("Python 3.12 Released!") == "python-312-released"
assert slugify("") == ""
assert slugify("already-slugified") == "already-slugified"
assert slugify("UPPERCASE") == "uppercase"
assert slugify("foo---bar") == "foo-bar"
assert slugify("...leading and trailing...") == "leading-and-trailing"
assert slugify("café résumé") == "caf-rsum"
assert slugify("one") == "one"
```

Hidden tests:
```python
assert slugify("  ") == ""
assert slugify("---") == ""
assert slugify("Hello, World! How's it going?") == "hello-world-hows-it-going"
assert slugify("under_score_test") == "under-score-test"
assert slugify("MiXeD CaSe 123") == "mixed-case-123"
```

### Initial state: Blank slate + LLM bootstrap

- Agent starts with only the function signature
- Local LLM generates the first seed implementation
- All subsequent evolution is AST-mutation only

**Bootstrap validation gate**: The LLM-generated seed MUST pass a minimum quality check before evolution begins:
1. Must be syntactically valid (`ast.parse()` succeeds)
2. Must pass sandbox static analysis (Section 6)
3. Must pass at least 1 train test (non-zero `pass_ratio`)

If the seed fails validation:
- **Retry**: Re-prompt the LLM up to 3 times with the error message appended to the prompt
- **Fallback**: If all retries fail, use a hardcoded minimal stub that returns a type-correct default (e.g., `return (1, 2)` for two_sum_sorted). This guarantees evolution can start, even from a very low fitness baseline

---

## 6. Execution Sandbox

### Phase 1: Lightweight Docker wrapper

- Each evaluation runs via `docker run` subprocess call
- Resource limits: CPU time, memory, no network access, read-only filesystem (except working dir)
- Timeout enforced at both Docker and subprocess level
- No direct `docker-py` dependency — plain subprocess calls to `docker` CLI

**Required pre-implementation benchmark**: Time 100 sequential `docker run` calls with a trivial Python script (`print("ok")`) on the target macOS machine. Docker Desktop on macOS has significant startup latency per container.

**Latency mitigation strategy** (activate based on benchmark results):

| Per-call latency | Strategy |
|-----------------|----------|
| < 200ms | Proceed with `docker run` per evaluation. No changes needed. |
| 200ms–1s | **Container reuse**: Keep a long-running container alive, copy organism code in via `docker cp`, execute via `docker exec`. Same security constraints apply (re-create container every N evals to prevent state leakage). |
| > 1s | **Phase 1 fallback**: Use `subprocess` with `resource` module limits (ulimit) + AST static analysis as primary sandbox. Accept marginally lower isolation for Phase 1. Migrate to Docker in Phase 2 when iteration speed is less critical. |

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

## 9. Pre-Implementation Spikes (Required)

These spikes MUST be completed before writing the main system. They validate critical assumptions and produce data needed for parameter calibration.

| Spike | Duration | Deliverable | Go/No-Go Criterion |
|-------|----------|-------------|---------------------|
| **AST mutation feasibility** | 1–2 days | Standalone script: 100 random mutations on a simple function. Report syntactic validity rate, semantic difference rate, fitness improvement rate. | Viable mutation rate ≥ 5%. If below, activate fallback tiers (Section 2). |
| **Task definition validation** | 0.5 day | All 3 tasks: function signatures, train tests, hidden tests committed and passing with a hand-written reference solution. | All train + hidden tests pass against reference solutions. |
| **Docker latency benchmark** | 0.5 day | Benchmark script: 100 sequential `docker run` calls on target macOS machine. Report mean/p95 latency. | Latency ≤ 200ms/call for `docker run` path. Otherwise activate mitigation (Section 6). |
| **SA/w2 curve visualization** | 0.5 day | Plot temperature and w2_effective over 500 steps with initial parameter guesses. Verify stability constraint (Section 4). | w2 curve never leads temperature curve by more than one phase. |

**Recommended order**: Task definitions → Docker benchmark → AST feasibility → Curve visualization.

---

## 10. Decisions Deferred to Phase 2

| Topic | Phase 1 Status | Phase 2 Plan |
|-------|---------------|--------------|
| Population dynamics | Single agent | Multiple agents with selection, crossover, diversity maintenance |
| Diversity maintenance | Not applicable | Novelty Search, MAP-Elites, or entropy regularization |
| Respawn on death | No respawn | Respawn from best ancestor or crossover of survivors |
| Property-based testing | Train/hidden split | Add Hypothesis-style generative tests |
| Live dashboard | JSON logs only | Real-time web dashboard |
| Academic rigor | Demo-level | Formal hypotheses, baselines, statistical tests, prior work comparison |

---

## 11. Open Parameters (To Be Determined Empirically)

| Parameter | Description | Initial Guess | Method |
|-----------|-------------|---------------|--------|
| `w1` (pass_ratio weight) | Importance of test passing | 0.9 | Sensitivity analysis |
| `w2` (edit_cost weight) | Penalty for large code changes | 0.1 | Sensitivity analysis |
| `base_survival_cost` | Energy drain per step | 0.01 | Tune to target ~100-step runs |
| `improvement_multiplier` | Energy gain per unit of positive Δfitness (additive) | 1.0 | Experiment |
| `degradation_multiplier` | Energy loss per unit of negative Δfitness (additive) | 1.0 | Experiment |
| `w2_floor` | Minimum edit cost penalty (prevents code bloat) | 0.02 | Experiment |
| `N_stagnation` | Steps without improvement before death | 100 | Experiment |
| `initial_temperature` | SA starting temperature | 1.0 | Standard SA tuning |
| `cooling_rate` | SA temperature decay | 0.995 per step | Standard SA tuning |
| `pass_ratio_threshold` | Minimum train pass_ratio to unlock next task | 0.9 | Experiment |
| `fitness_threshold` | Minimum overall fitness to unlock next task (pass_ratio - edit_cost) | 0.8 | Experiment |
| `mutation_stagnation_window` | Steps before increasing mutation size | 20 | Experiment |

**Timer constraint**: `N_stagnation` MUST be significantly larger than `mutation_stagnation_window` (recommended: ≥ 5x). The adaptive mutation ramp-up needs sufficient runway to attempt larger structural changes before the agent is killed for stagnation. With `mutation_stagnation_window=20` and `N_stagnation=100`, the agent gets ~80 steps of escalating mutation after the first ramp-up — enough time for algorithmic-level changes to take effect.

### Parameter tuning protocol

These 10+ parameters interact non-linearly. Tuning them all at once is intractable. Follow this phased protocol:

**Phase A — Fix structural parameters first** (single task: `two_sum_sorted`):

| Priority | Parameters | Sweep range | Success metric |
|----------|-----------|-------------|----------------|
| 1st | `N_stagnation`, `mutation_stagnation_window` | 50/10, 100/20, 200/40 | Agent survives >50 steps |
| 2nd | `base_survival_cost`, `improvement_reward` | cost: 0.005–0.05; reward: 0.5x–2.0x Δfitness | Median lifespan ~100 steps |
| 3rd | `initial_temperature`, `cooling_rate` | temp: 0.5–2.0; rate: 0.99–0.999 | Acceptance rate settles between 10–30% by step 100 |

**Phase B — Tune fitness weights** (all tasks):

| Priority | Parameters | Sweep range | Success metric |
|----------|-----------|-------------|----------------|
| 4th | `w1`, `w2`, `w2_floor` | w1: 0.8–1.0; w2: 0.05–0.2; floor: 0.01–0.05 | Fitness improves monotonically over first 50 steps |
| 5th | `decay_factor` | 0.995–0.9999 | No code bloat (AST node count stays within 2x of seed) |

**Phase C — Validate curriculum thresholds**:

| Priority | Parameters | Sweep range | Success metric |
|----------|-----------|-------------|----------------|
| 6th | `pass_ratio_threshold`, `fitness_threshold` | pass: 0.7–0.9; fitness: 0.6–0.8 | Agent advances to task 2 within 200 steps |

Each sweep: 5 runs per configuration, same seed set. Compare median lifespan and final fitness. Fix the winner before moving to the next priority.
