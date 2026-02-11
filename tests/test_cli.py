from pathlib import Path

from alife_core import cli


def test_cli_supports_run_and_spike_commands() -> None:
    parser = cli.build_parser()

    run_args = parser.parse_args(["run", "--task", "two_sum_sorted", "--seed", "7"])
    assert run_args.command == "run"
    assert run_args.task == "two_sum_sorted"
    assert run_args.seed == 7

    spike_args = parser.parse_args(["spike", "docker-latency"])
    assert spike_args.command == "spike"
    assert spike_args.spike_command == "docker-latency"


def test_dispatch_run_invokes_runtime(monkeypatch, tmp_path: Path) -> None:
    called = {}

    def fake_run_experiment(task_name, config, output_root):
        called["task_name"] = task_name
        called["seed"] = config.seed
        called["output_root"] = output_root

        class Summary:
            run_id = "run-1"
            log_path = tmp_path / "logs" / "run-1.jsonl"
            completed_tasks = [task_name]

        return Summary()

    monkeypatch.setattr(cli, "run_experiment", fake_run_experiment)

    parser = cli.build_parser()
    args = parser.parse_args(["run", "--task", "two_sum_sorted", "--seed", "9"])
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
