#!/usr/bin/env python3
import json
import statistics
import warnings
from pathlib import Path

from alife_core.models import RunConfig, SandboxBackend
from alife_core.runtime import run_experiment


def run_parameter_sweep(
    task_name: str = "two_sum_sorted",
    output_root: Path | str = Path("."),
    output_path: Path | str | None = None,
    seeds: list[int] | None = None,
    grid: list[dict[str, int]] | None = None,
    max_steps: int = 50,
    sandbox_backend: SandboxBackend = "docker",
    allow_unsafe_process_backend: bool = False,
) -> list[dict[str, float | int]]:
    sweep_root = Path(output_root)
    if sandbox_backend == "process":
        if not allow_unsafe_process_backend:
            raise ValueError(
                "process backend is unsafe for sweep runs; pass allow_unsafe_process_backend=True"
            )
        warnings.warn(
            "Running parameter sweep with process backend bypasses container isolation",
            stacklevel=2,
        )
    seeds = [0, 7, 13] if seeds is None else seeds
    grid = (
        [
            {"n_stagnation": 50, "mutation_stagnation_window": 10},
            {"n_stagnation": 100, "mutation_stagnation_window": 20},
            {"n_stagnation": 200, "mutation_stagnation_window": 40},
        ]
        if grid is None
        else grid
    )

    rows: list[dict[str, float | int]] = []
    for params in grid:
        completed_counts: list[int] = []
        for seed in seeds:
            config = RunConfig(
                seed=seed,
                sandbox_backend=sandbox_backend,
                bootstrap_backend="static",
                run_curriculum=True,
                max_steps=max_steps,
                n_stagnation=params["n_stagnation"],
                mutation_stagnation_window=params["mutation_stagnation_window"],
            )
            summary = run_experiment(
                task_name=task_name,
                config=config,
                output_root=sweep_root / "sweep_runs",
            )
            completed_counts.append(len(summary.completed_tasks))

        rows.append(
            {
                "n_stagnation": params["n_stagnation"],
                "mutation_stagnation_window": params["mutation_stagnation_window"],
                "median_completed_tasks": float(statistics.median(completed_counts)),
                "mean_completed_tasks": float(statistics.fmean(completed_counts)),
            }
        )

    if output_path is not None:
        Path(output_path).write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    return rows


if __name__ == "__main__":
    payload = run_parameter_sweep(output_path="sweep_summary.json")
    print(json.dumps(payload, sort_keys=True))
