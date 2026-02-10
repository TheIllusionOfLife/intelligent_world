# Minimal Artificial Life (ALife) Project Overview

> **Document role**: Initial proposition — the founding vision and conceptual design for the project.
>
> **Related documents**:
> - [`unified_review.md`](./unified_review.md) — Research review that identified gaps, contradictions, and missing definitions in this document
> - [`technical_design_spec.md`](./technical_design_spec.md) — Concrete design decisions made after considering this document and the review

---

## 1. Project Vision

This project aims to build a **minimal artificial life system** in which AI agents can:

- Act without human-in-the-loop approval
- Evaluate themselves using a fully computable objective function
- Improve through iterative action–evaluation loops
- Survive or die based on objective performance

The goal is not to build intelligence directly, but to:

> Create an environment where evolution-like dynamics can emerge.

This project treats AI not as a tool, but as a **process that persists, adapts, and competes** under constraints.

---

## 2. Core Philosophical Shift

### Old paradigm

```
Ideas are cheap, execution is everything.
```

Value was in:
- Implementation skill
- Engineering execution

### New paradigm (AI execution era)

```
Ideas are cheap, execution is cheaper.
```

Execution is now commoditized by AI agents.

### New sources of value

1. Problem selection
2. Judgment and taste
3. Iteration speed
4. Distribution

In this project, that insight is pushed further:

> Humans are removed from the execution loop entirely.

The human role shifts from:

```
Operator → Rule designer
```

---

## 3. Why Artificial Life?

Modern AI agents often suffer from a major bottleneck:

- Human approvals
- Human interpretation
- Human judgment

This limits scaling.

To remove this bottleneck, we need:

> A fully computable evaluation function that replaces human judgment.

When:
- Evaluation is objective
- Evaluation is automatic
- Evaluation requires no human perception

Then:

> AI can act autonomously.

This leads naturally to the concept of **artificial life**.

---

## 4. Engineering Definition of Artificial Life

A system qualifies as artificial life if it satisfies four conditions:

1. It has a self-referential objective function
2. It observes an environment
3. It changes behavior based on evaluation
4. It attempts to persist or improve over time

Not required:

- Consciousness
- Emotion
- Self-awareness

The **evaluation function** is the core of life in this system.

---

## 5. Role of the Evaluation Function

Biological analogy:

| Biology | ALife System |
|---------|--------------|
| Survival | Fitness score |
| Pain | Penalty |
| Pleasure | Reward |
| Evolution | Policy update |
| Death | Fitness or energy threshold |

The evaluation function acts as:

> The equivalent of natural selection.

It replaces the need for human judgment.

---

## 6. Requirements for a Valid Evaluation Function

To enable true autonomy, the evaluation function must be:

### 1. Fully computable
- No human input required
- No physical observation required

### 2. Objective
- Deterministic or statistically stable
- Same input → same score

### 3. Differential
It must measure improvement, not just absolute performance:

```
Δfitness = f(state_t+1) − f(state_t)
```

### 4. Multi-objective
Single metrics cause degenerate behavior.

Example:

```
Fitness =
  w1 * task_success
+ w2 * efficiency
+ w3 * constraint_safety
```

### 5. Failure-aware
Instead of defining success precisely, define failure:

- Timeout
- Resource exhaustion
- Constraint violation

Failure leads to death.

---

## 7. Minimal ALife Architecture

The minimal loop is:

```
Environment
   ↓
Perception
   ↓
Policy (agent logic)
   ↓
Action
   ↓
Evaluation (fitness function)
   ↓
Policy update
   ↓
(loop)
```

Humans only:
- Design evaluation function
- Observe logs

Humans never:
- Approve actions
- Judge results
- Intervene in the loop

---

## 8. Chosen Environment: Code World

We selected **Environment B: Code World**.

### Why?

Because it satisfies all constraints:

- Fully computable
- Objective
- No physical observation
- Deterministic evaluation

### Environment definition

The world consists of:

- Small programming tasks
- Unit tests
- Runtime measurements

Agent actions:

- Modify solution code

Environment feedback:

- Test pass rate
- Runtime
- Code change cost

---

## 9. Project Directory Structure

```
alife_min/
  pyproject.toml
  README.md

  src/
    alife/
      tasks/
      evaluator/
      evolution/

  tests/

  organisms/
    current/
    archive/
```

### Key directories

**tasks/**
- Definitions of coding problems

**tests/**
- Unit tests (environment rules)

**organisms/current/**
- Current agent solutions

**organisms/archive/**
- Historical generations

---

## 10. Evaluation System

Each agent is evaluated using:

1. Test pass ratio
2. Execution time
3. Edit cost (code changes)

Example formula:

```
Fitness =
  0.8 * pass_ratio
− 0.15 * time_seconds
− 0.05 * edit_cost
```

Properties:

- Fully objective
- Fully computable
- No human interpretation

---

## 11. Agent Life Cycle

Each agent has an internal energy value.

### Initial state

```
Energy = 1.0
```

### Per-step updates

- Base survival cost reduces energy
- Improvement increases energy
- Degradation reduces energy

### Death conditions

An agent dies if:

1. Energy ≤ 0
2. No improvement for N steps
3. Too many timeouts
4. Resource budget exceeded

Death removes the agent from the system.

---

## 12. Mutation Mechanism

Agents evolve via code mutations.

### Types of mutation

1. Small numeric tweaks
2. Edge-case guards
3. Line swaps
4. Rare large algorithmic changes

Principle:

> Most mutations must be small and safe.

This avoids constant collapse.

---

## 13. Agent Loop

The core loop:

```
while agent.is_alive():

    candidate = mutate(agent.code)

    write(candidate)

    fitness = evaluate(candidate)

    if fitness > previous_fitness:
        accept(candidate)
        increase_energy()
    else:
        reject_or_probabilistic_accept()
        decrease_energy()
```

This creates:

- Adaptation
- Survival pressure
- Evolutionary dynamics

---

## 14. Initial Task Set

Three initial tasks are used:

### Task 1: two_sum_sorted
- Find indices of two numbers that sum to target
- Sorted input

### Task 2: run_length_encode
- Compress string using run-length encoding

### Task 3: slugify
- Convert text into URL-safe slug

Design goals:

- Deterministic evaluation
- Edge cases
- Performance differences

---

## 15. What Counts as “Life” in This System

Signs of artificial life emergence:

- Agent begins to avoid actions that reduce fitness
- Different agents develop different strategies
- Long-term improvements appear
- Trade-offs emerge between speed and correctness

At this point:

> The system is no longer a static program, but a dynamic process.

---

## 16. Human Role in the System

Humans:

- Design evaluation functions
- Define environment rules
- Observe system behavior

Humans do NOT:

- Approve agent actions
- Judge outputs
- Control each step

This ensures true autonomy.

---

## 17. Project Goal (Short Version)

Build a system where:

- Agents write and improve code
- Evaluation is fully automatic
- Agents survive or die objectively
- Evolutionary dynamics emerge

In essence:

> We are not building intelligence.
> We are building a world where intelligence can evolve.

