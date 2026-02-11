from pathlib import Path
from types import SimpleNamespace

from alife_core import cli
from alife_core.models import RunConfig


def test_cli_supports_run_and_spike_commands() -> None:
    parser = cli.build_parser()

    run_args = parser.parse_args(
        [
            "run",
            "--task",
            "two_sum_sorted",
            "--seed",
            "7",
            "--bootstrap-backend",
            "static",
            "--ollama-model",
            "gpt-oss:20b",
            "--curriculum",
        ]
    )
    assert run_args.command == "run"
    assert run_args.task == "two_sum_sorted"
    assert run_args.seed == 7
    assert run_args.bootstrap_backend == "static"
    assert run_args.ollama_model == "gpt-oss:20b"
    assert run_args.curriculum is True

    run_args_no_curriculum = parser.parse_args(
        [
            "run",
            "--task",
            "two_sum_sorted",
            "--no-curriculum",
        ]
    )
    assert run_args_no_curriculum.curriculum is False

    spike_args = parser.parse_args(["spike", "docker-latency"])
    assert spike_args.command == "spike"
    assert spike_args.spike_command == "docker-latency"

    sweep_args = parser.parse_args(["spike", "parameter-sweep"])
    assert sweep_args.command == "spike"
    assert sweep_args.spike_command == "parameter-sweep"
    assert sweep_args.sweep_output is None

    sweep_args_with_output = parser.parse_args(
        ["spike", "parameter-sweep", "--sweep-output", "tmp/sweep.json"]
    )
    assert sweep_args_with_output.sweep_output == "tmp/sweep.json"


def test_dispatch_run_invokes_runtime(monkeypatch, tmp_path: Path) -> None:
    called = {}

    monkeypatch.setattr(
        cli,
        "load_run_config",
        lambda _path, **kwargs: RunConfig(seed=kwargs.get("seed_override") or 0),
    )

    def fake_run_experiment(task_name, config, output_root):
        called["task_name"] = task_name
        called["seed"] = config.seed
        called["output_root"] = output_root

        return SimpleNamespace(
            run_id="run-1",
            log_path=tmp_path / "logs" / "run-1.jsonl",
            completed_tasks=[task_name],
        )

    monkeypatch.setattr(cli, "run_experiment", fake_run_experiment)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run",
            "--task",
            "two_sum_sorted",
            "--seed",
            "9",
            "--bootstrap-backend",
            "static",
            "--ollama-model",
            "gpt-oss:20b",
        ]
    )
    exit_code = cli._dispatch(args)

    assert exit_code == 0
    assert called["task_name"] == "two_sum_sorted"
    assert called["seed"] == 9


def test_dispatch_spike_calls_ast_feasibility(monkeypatch) -> None:
    called = {"ast": False}

    def fake_ast(samples=100, seed=0):
        _ = samples
        _ = seed
        called["ast"] = True
        return {"syntactic_validity_rate": 1.0, "semantic_difference_proxy_rate": 0.5}

    monkeypatch.setattr(cli, "run_ast_feasibility_spike", fake_ast)

    parser = cli.build_parser()
    args = parser.parse_args(["spike", "ast-feasibility"])
    exit_code = cli._dispatch(args)

    assert exit_code == 0
    assert called["ast"] is True


def test_spike_loader_is_stable_across_working_directories(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    payload = cli.run_schedule_curve_spike(steps=2)

    assert len(payload["temperature"]) == 3


def test_dispatch_spike_calls_parameter_sweep(monkeypatch) -> None:
    called = {"sweep": False}

    def fake_sweep(output_path=None, allow_unsafe_process_backend=False):
        _ = output_path
        _ = allow_unsafe_process_backend
        called["sweep"] = True
        return [{"n_stagnation": 1}]

    monkeypatch.setattr(cli, "run_parameter_sweep_spike", fake_sweep)

    parser = cli.build_parser()
    args = parser.parse_args(["spike", "parameter-sweep"])
    exit_code = cli._dispatch(args)

    assert exit_code == 0
    assert called["sweep"] is True


def test_dispatch_run_can_disable_curriculum(monkeypatch, tmp_path: Path) -> None:
    called = {}

    def fake_load(_path, **kwargs):
        called["run_curriculum_override"] = kwargs["run_curriculum_override"]
        return RunConfig(seed=0)

    monkeypatch.setattr(cli, "load_run_config", fake_load)
    monkeypatch.setattr(
        cli,
        "run_experiment",
        lambda task_name, config, output_root: SimpleNamespace(
            run_id="run-1",
            log_path=tmp_path / "logs" / "run-1.jsonl",
            completed_tasks=[task_name],
        ),
    )

    parser = cli.build_parser()
    args = parser.parse_args(["run", "--task", "two_sum_sorted", "--no-curriculum"])
    exit_code = cli._dispatch(args)

    assert exit_code == 0
    assert called["run_curriculum_override"] is False
