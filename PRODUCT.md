# PRODUCT.md

## Purpose
`intelligent_world` explores a minimal, safety-aware ALife runtime that can iteratively improve Python task solutions.

## Target users
- Researchers experimenting with evolutionary code-generation loops.
- Engineers evaluating mutation/evaluation strategies under strict execution boundaries.
- Contributors building toward a reproducible ALife experimentation platform.

## Core user problems
- Running repeated code-mutation experiments safely.
- Measuring task progress with deterministic, inspectable evaluation.
- Comparing strategy changes without rebuilding orchestration from scratch.

## Key features
- Task-driven evaluation with train/hidden case split.
- Mutation + selection lifecycle for single-agent and population modes.
- Docker-first execution sandbox for untrusted candidate code.
- Structured JSONL run/event logs for analysis.
- CLI-driven spikes for latency, AST feasibility, and parameter sweeps.

## Business/research objectives
- Maintain a reliable baseline runtime for ALife experiments.
- Keep iteration cost low while preserving safety boundaries.
- Enable reproducible comparisons across config variants and seeds.
- Provide clear extensibility points for future tasks, mutators, and evaluators.

## Non-goals (current scope)
- Large-scale distributed orchestration.
- Production multi-tenant hosting.
- Broad benchmark suite beyond built-in exploratory tasks.

## Success signals
- CI remains green (tests/lint/format).
- Contributors can run end-to-end experiments within minutes.
- Experiment outputs are reproducible and diagnosable from logs/config.
