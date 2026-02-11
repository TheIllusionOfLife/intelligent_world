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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alife")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--task", required=True)
    run_parser.add_argument("--config", default="configs/default.yaml")
    run_parser.add_argument("--seed", type=int, default=None)

    spike_parser = subparsers.add_parser("spike")
    spike_subparsers = spike_parser.add_subparsers(dest="spike_command", required=True)
    spike_subparsers.add_parser("docker-latency")
    spike_subparsers.add_parser("ast-feasibility")
    spike_subparsers.add_parser("schedule-curve")

    return parser


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "run":
        config = load_run_config(Path(args.config), seed_override=args.seed)
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

    return 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return _dispatch(args)


if __name__ == "__main__":
    raise SystemExit(main())
