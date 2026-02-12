#!/usr/bin/env python3
"""Mutation viability spike: measure mutation quality across real tasks."""

import random

from alife_core.bootstrap import static_seed
from alife_core.evaluator.core import evaluate_candidate
from alife_core.models import RunConfig, TaskSpec
from alife_core.mutation.validation import validate_candidate
from alife_core.runtime import _mutate_code
from alife_core.tasks.builtin import load_builtin_tasks

_PROCESS_CONFIG = RunConfig(sandbox_backend="process", allow_unsafe_process_backend=True)


def _measure_task(
    task: TaskSpec,
    samples: int,
    chain_depth: int,
    rng: random.Random,
) -> dict[str, float]:
    if samples <= 0:
        return {
            "syntactic_validity_rate": 0.0,
            "fitness_improvement_rate": 0.0,
            "non_destructive_rate": 0.0,
            "mean_chain_survival_steps": 0.0,
        }

    base_code = static_seed(task)
    base_eval = evaluate_candidate(base_code, task=task, edit_cost=0.0, config=_PROCESS_CONFIG)

    valid_count = 0
    improved_count = 0
    non_destructive_count = 0

    for _ in range(samples):
        candidate = _mutate_code(base_code, rng, intensity=1)
        validation = validate_candidate(candidate)
        if not validation.is_valid:
            continue
        valid_count += 1
        candidate_eval = evaluate_candidate(
            candidate, task=task, edit_cost=0.0, config=_PROCESS_CONFIG
        )
        if candidate_eval.fitness > base_eval.fitness:
            improved_count += 1
        if candidate_eval.train_pass_ratio > 0:
            non_destructive_count += 1

    validity_rate = valid_count / samples
    improvement_rate = improved_count / valid_count if valid_count > 0 else 0.0
    non_destructive_rate = non_destructive_count / valid_count if valid_count > 0 else 0.0

    # Chain survival measurement
    chain_survivals: list[int] = []
    for _ in range(samples):
        code = base_code
        survived = 0
        for _depth in range(chain_depth):
            code = _mutate_code(code, rng, intensity=1)
            validation = validate_candidate(code)
            if not validation.is_valid:
                break
            chain_eval = evaluate_candidate(code, task=task, edit_cost=0.0, config=_PROCESS_CONFIG)
            if chain_eval.train_pass_ratio == 0:
                break
            survived += 1
        chain_survivals.append(survived)

    mean_survival = sum(chain_survivals) / len(chain_survivals) if chain_survivals else 0.0

    return {
        "syntactic_validity_rate": validity_rate,
        "fitness_improvement_rate": improvement_rate,
        "non_destructive_rate": non_destructive_rate,
        "mean_chain_survival_steps": mean_survival,
    }


def run_spike(
    samples: int = 200,
    chain_depth: int = 20,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    tasks = load_builtin_tasks()
    results: dict[str, dict[str, float]] = {}
    for task_name, task in tasks.items():
        results[task_name] = _measure_task(task, samples, chain_depth, rng)
    return results


if __name__ == "__main__":
    import json

    print(json.dumps(run_spike(), indent=2))
