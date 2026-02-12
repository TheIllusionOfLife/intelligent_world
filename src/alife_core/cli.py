import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

from alife_core.runtime import load_run_config, run_experiment

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(path: str) -> ModuleType:
    script_path = _REPO_ROOT / path
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_docker_latency_spike(iterations: int = 100) -> dict[str, float | str]:
    module = _load_script_module("scripts/docker_latency_spike.py")
    return module.benchmark(iterations=iterations)


def run_ast_feasibility_spike(samples: int = 100, seed: int = 0) -> dict[str, float]:
    module = _load_script_module("scripts/ast_feasibility_spike.py")
    return module.run_spike(samples=samples, seed=seed)


def run_schedule_curve_spike(steps: int = 500) -> dict[str, object]:
    module = _load_script_module("scripts/schedule_curve_spike.py")
    return module.generate_schedule_data(steps=steps)


def run_parameter_sweep_spike(
    output_path: str | None = None,
    allow_unsafe_process_backend: bool = False,
) -> list[dict[str, float | int]]:
    module = _load_script_module("scripts/parameter_sweep.py")
    return module.run_parameter_sweep(
        output_path=output_path,
        allow_unsafe_process_backend=allow_unsafe_process_backend,
    )


def run_mutation_viability_spike(
    samples: int = 200,
    chain_depth: int = 20,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    module = _load_script_module("scripts/mutation_viability_spike.py")
    return module.run_spike(samples=samples, chain_depth=chain_depth, seed=seed)


def run_metrics_report_spike(log_path: str) -> dict[str, object]:
    module = _load_script_module("scripts/metrics_report.py")
    return module.summarize_metrics(log_path)


def run_experiment_campaign_spike(
    output_dir: str = "campaign_results",
    allow_unsafe_process_backend: bool = False,
) -> list[dict]:
    module = _load_script_module("scripts/experiment_campaign.py")
    backend = "process" if allow_unsafe_process_backend else "docker"
    return module.run_campaign(
        output_dir=Path(output_dir),
        sandbox_backend=backend,
    )


def run_analyze_campaign_spike(
    results_dir: str = "campaign_results",
    output: str = "docs/experiment_campaign_report.md",
) -> dict:
    module = _load_script_module("scripts/analyze_campaign.py")
    return module.analyze_campaign(
        results_dir=Path(results_dir),
        output_path=Path(output),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alife")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--task", required=True)
    run_parser.add_argument("--config", default="configs/default.yaml")
    run_parser.add_argument("--seed", type=int, default=None)
    run_parser.add_argument("--bootstrap-backend", choices=["static", "ollama"], default=None)
    run_parser.add_argument("--ollama-model", default=None)
    run_parser.add_argument("--unsafe-process-backend", action="store_true")
    run_parser.add_argument("--population", action="store_true")
    run_parser.add_argument("--population-size", type=int, default=None)
    run_parser.add_argument("--elite-count", type=int, default=None)
    run_parser.add_argument("--tournament-k", type=int, default=None)
    run_parser.add_argument("--crossover-rate", type=float, default=None)
    run_parser.add_argument("--mutation-rate", type=float, default=None)
    run_parser.add_argument("--max-generations", type=int, default=None)
    run_parser.add_argument("--population-workers", type=int, default=None)
    curriculum_group = run_parser.add_mutually_exclusive_group()
    curriculum_group.add_argument("--curriculum", dest="curriculum", action="store_true")
    curriculum_group.add_argument("--no-curriculum", dest="curriculum", action="store_false")
    run_parser.set_defaults(curriculum=None)

    spike_parser = subparsers.add_parser("spike")
    spike_subparsers = spike_parser.add_subparsers(dest="spike_command", required=True)
    spike_subparsers.add_parser("docker-latency")
    spike_subparsers.add_parser("ast-feasibility")
    spike_subparsers.add_parser("schedule-curve")
    sweep_parser = spike_subparsers.add_parser("parameter-sweep")
    sweep_parser.add_argument("--sweep-output", default=None)
    sweep_parser.add_argument("--unsafe-process-backend", action="store_true")
    viability_parser = spike_subparsers.add_parser("mutation-viability")
    viability_parser.add_argument("--samples", type=int, default=200)
    viability_parser.add_argument("--chain-depth", type=int, default=20)
    viability_parser.add_argument("--seed", type=int, default=0)
    metrics_parser = spike_subparsers.add_parser("metrics-report")
    metrics_parser.add_argument("--log-path", required=True)
    campaign_parser = spike_subparsers.add_parser("experiment-campaign")
    campaign_parser.add_argument("--output-dir", default="campaign_results")
    campaign_parser.add_argument("--unsafe-process-backend", action="store_true")
    analyze_parser = spike_subparsers.add_parser("analyze-campaign")
    analyze_parser.add_argument("--results-dir", default="campaign_results")
    analyze_parser.add_argument("--output", default="docs/experiment_campaign_report.md")

    return parser


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "run":
        config = load_run_config(
            Path(args.config),
            seed_override=args.seed,
            bootstrap_backend_override=args.bootstrap_backend,
            ollama_model_override=args.ollama_model,
            run_curriculum_override=args.curriculum,
            allow_unsafe_process_backend_override=args.unsafe_process_backend,
            evolution_mode_override="population" if args.population else None,
            population_size_override=args.population_size,
            elite_count_override=args.elite_count,
            tournament_k_override=args.tournament_k,
            crossover_rate_override=args.crossover_rate,
            mutation_rate_override=args.mutation_rate,
            max_generations_override=args.max_generations,
            population_workers_override=args.population_workers,
        )
        summary = run_experiment(task_name=args.task, config=config, output_root=Path("."))
        print(
            json.dumps(
                {
                    "run_id": summary.run_id,
                    "log_path": str(summary.log_path),
                    "completed_tasks": summary.completed_tasks,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.spike_command == "docker-latency":
        print(json.dumps(run_docker_latency_spike(), sort_keys=True))
        return 0

    if args.spike_command == "ast-feasibility":
        print(json.dumps(run_ast_feasibility_spike(), sort_keys=True))
        return 0

    if args.spike_command == "schedule-curve":
        print(json.dumps(run_schedule_curve_spike(), sort_keys=True))
        return 0

    if args.spike_command == "parameter-sweep":
        print(
            json.dumps(
                run_parameter_sweep_spike(
                    output_path=args.sweep_output,
                    allow_unsafe_process_backend=args.unsafe_process_backend,
                ),
                sort_keys=True,
            )
        )
        return 0

    if args.spike_command == "mutation-viability":
        print(
            json.dumps(
                run_mutation_viability_spike(
                    samples=args.samples,
                    chain_depth=args.chain_depth,
                    seed=args.seed,
                ),
                sort_keys=True,
            )
        )
        return 0

    if args.spike_command == "metrics-report":
        print(json.dumps(run_metrics_report_spike(log_path=args.log_path), sort_keys=True))
        return 0

    if args.spike_command == "experiment-campaign":
        results = run_experiment_campaign_spike(
            output_dir=args.output_dir,
            allow_unsafe_process_backend=args.unsafe_process_backend,
        )
        print(json.dumps({"total_runs": len(results)}, sort_keys=True))
        return 0

    if args.spike_command == "analyze-campaign":
        analysis = run_analyze_campaign_spike(
            results_dir=args.results_dir,
            output=args.output,
        )
        print(json.dumps({"success_rates": analysis["success_rates"]}, sort_keys=True))
        return 0

    return 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return _dispatch(args)


if __name__ == "__main__":
    raise SystemExit(main())
