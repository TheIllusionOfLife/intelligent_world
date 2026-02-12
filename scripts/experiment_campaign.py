#!/usr/bin/env python3
"""Systematic experimental campaign runner for the intelligent_world ALife project.

Runs a 4×3×5 matrix of configurations across tasks and seeds,
collecting metrics for statistical analysis.
"""

import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

from alife_core.models import RunConfig, SandboxBackend
from alife_core.runtime import run_experiment

DEFAULT_SEEDS: list[int] = [0, 7, 13, 42, 99]
DEFAULT_TASKS: list[str] = ["two_sum_sorted", "run_length_encode", "slugify"]

# Configuration matrix:
# A = single_agent, no semantic mutations (baseline)
# B = single_agent, with semantic mutations
# C = population, no semantic mutations
# D = population, with semantic mutations
CONFIGS: dict[str, dict[str, object]] = {
    "A": {"evolution_mode": "single_agent", "enable_semantic_mutation": False},
    "B": {"evolution_mode": "single_agent", "enable_semantic_mutation": True},
    "C": {"evolution_mode": "population", "enable_semantic_mutation": False},
    "D": {"evolution_mode": "population", "enable_semantic_mutation": True},
}


def build_run_configs(
    seeds: list[int] | None = None,
    tasks: list[str] | None = None,
    sandbox_backend: SandboxBackend = "process",
    max_steps: int = 200,
    max_generations: int = 50,
    population_size: int = 8,
) -> list[dict]:
    """Build the full matrix of run configurations.

    Returns a list of dicts with keys: config_id, task, seed, run_config.
    """
    seeds = seeds if seeds is not None else DEFAULT_SEEDS
    tasks = tasks if tasks is not None else DEFAULT_TASKS

    combos: list[dict] = []
    for config_id, config_params in CONFIGS.items():
        for task in tasks:
            for seed in seeds:
                base = RunConfig(
                    seed=seed,
                    sandbox_backend=sandbox_backend,
                    bootstrap_backend="static",
                    allow_unsafe_process_backend=(sandbox_backend == "process"),
                    max_steps=max_steps,
                    max_generations=max_generations,
                    population_size=population_size,
                )
                run_config = replace(base, **config_params)
                combos.append(
                    {
                        "config_id": config_id,
                        "task": task,
                        "seed": seed,
                        "run_config": run_config,
                    }
                )
    return combos


def extract_run_metrics(log_path: Path, mode: str) -> dict:
    """Parse a JSONL log file and extract key metrics for the campaign summary."""
    best_fitness = 0.0
    best_train_pass_ratio = 0.0
    best_hidden_pass_ratio = 0.0
    final_energy = 0.0
    total_steps = 0
    total_generations = 0
    convergence_reason: str | None = None
    diversity_trajectory: list[dict[str, float]] = []

    try:
        text = log_path.read_text(encoding="utf-8").strip()
    except (OSError, FileNotFoundError):
        text = ""

    if not text:
        result: dict = {
            "best_fitness": 0.0,
            "best_train_pass_ratio": 0.0,
            "best_hidden_pass_ratio": 0.0,
        }
        if mode == "single_agent":
            result["total_steps"] = 0
            result["final_energy"] = 0.0
        else:
            result["total_generations"] = 0
            result["convergence_reason"] = None
            result["diversity_trajectory"] = []
        return result

    for line in text.split("\n"):
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue

        event_type = event.get("event_type", "")
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue

        if event_type == "step.evaluated":
            total_steps += 1
            fitness = payload.get("fitness", 0.0)
            if fitness > best_fitness:
                best_fitness = fitness
                best_train_pass_ratio = payload.get("train_pass_ratio", 0.0)
                best_hidden_pass_ratio = payload.get("hidden_pass_ratio", 0.0)
            final_energy = payload.get("energy", 0.0)

        elif event_type == "generation.metrics":
            total_generations += 1
            fitness = payload.get("best_fitness", 0.0)
            if fitness > best_fitness:
                best_fitness = fitness
                best_train_pass_ratio = payload.get("best_train_pass_ratio", 0.0)
                best_hidden_pass_ratio = payload.get("best_hidden_pass_ratio", 0.0)
            diversity_trajectory.append(
                {
                    "shannon_entropy": payload.get("shannon_entropy", 0.0),
                    "structural_diversity_ratio": payload.get("structural_diversity_ratio", 0.0),
                }
            )

        elif event_type == "evolution.converged":
            convergence_reason = payload.get("reason")

    result = {
        "best_fitness": best_fitness,
        "best_train_pass_ratio": best_train_pass_ratio,
        "best_hidden_pass_ratio": best_hidden_pass_ratio,
    }

    if mode == "single_agent":
        result["total_steps"] = total_steps
        result["final_energy"] = final_energy
    else:
        result["total_generations"] = total_generations
        result["convergence_reason"] = convergence_reason
        result["diversity_trajectory"] = diversity_trajectory

    return result


def run_campaign(
    output_dir: Path,
    seeds: list[int] | None = None,
    tasks: list[str] | None = None,
    sandbox_backend: SandboxBackend = "process",
    max_steps: int = 200,
    max_generations: int = 50,
    population_size: int = 8,
) -> list[dict]:
    """Run the full experimental campaign and write summary.json."""
    if sandbox_backend == "process":
        warnings.warn(
            "Running campaign with process backend bypasses container isolation",
            stacklevel=2,
        )

    combos = build_run_configs(
        seeds=seeds,
        tasks=tasks,
        sandbox_backend=sandbox_backend,
        max_steps=max_steps,
        max_generations=max_generations,
        population_size=population_size,
    )

    output_dir = Path(output_dir)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    total = len(combos)

    for i, combo in enumerate(combos, 1):
        config_id = combo["config_id"]
        task = combo["task"]
        seed = combo["seed"]
        run_config: RunConfig = combo["run_config"]
        mode = run_config.evolution_mode

        print(f"[{i}/{total}] Config {config_id} | {task} | seed={seed}")
        sys.stdout.flush()

        entry: dict = {
            "config_id": config_id,
            "task": task,
            "seed": seed,
            "error": None,
        }

        try:
            summary = run_experiment(
                task_name=task,
                config=run_config,
                output_root=output_dir,
            )
            metrics = extract_run_metrics(summary.log_path, mode=mode)
            solved = (
                metrics["best_train_pass_ratio"] >= run_config.pass_ratio_threshold
                and metrics["best_fitness"] >= run_config.fitness_threshold
            )
            entry.update(
                {
                    "solved": solved,
                    "run_id": summary.run_id,
                    "log_path": str(summary.log_path),
                    **metrics,
                }
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            entry.update(
                {
                    "solved": False,
                    "best_fitness": 0.0,
                    "best_train_pass_ratio": 0.0,
                    "best_hidden_pass_ratio": 0.0,
                    "error": str(exc),
                }
            )

        results.append(entry)

        # Write incrementally so progress is saved
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run experiment campaign")
    parser.add_argument("--output-dir", default="campaign_results", help="Output directory")
    parser.add_argument(
        "--unsafe-process-backend",
        action="store_true",
        help="Use process sandbox (faster, no Docker needed)",
    )
    args = parser.parse_args()

    backend: SandboxBackend = "process" if args.unsafe_process_backend else "docker"
    run_campaign(output_dir=Path(args.output_dir), sandbox_backend=backend)
