import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(path: str):
    script_path = REPO_ROOT / path
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_docker_latency_handles_zero_iterations() -> None:
    module = _load_script("scripts/docker_latency_spike.py")

    result = module.benchmark(iterations=0)

    assert result["mean_ms"] == 0.0
    assert result["p95_ms"] == 0.0
    assert result["recommended_strategy"] == "unknown"


def test_docker_latency_reports_unavailable_daemon(monkeypatch) -> None:
    module = _load_script("scripts/docker_latency_spike.py")

    class Failed:
        returncode = 1

    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: Failed())
    result = module.benchmark(iterations=1)

    assert result["recommended_strategy"] == "docker_unavailable"
    assert result["docker_available"] is False


def test_ast_feasibility_handles_zero_samples() -> None:
    module = _load_script("scripts/ast_feasibility_spike.py")

    result = module.run_spike(samples=0)

    assert result["syntactic_validity_rate"] == 0.0
    assert result["semantic_difference_proxy_rate"] == 0.0
    assert result["fitness_improvement_rate"] == 0.0


def test_ast_feasibility_reports_non_zero_improvement_rate_for_default_seed() -> None:
    module = _load_script("scripts/ast_feasibility_spike.py")

    result = module.run_spike(samples=100, seed=0)

    assert result["fitness_improvement_rate"] > 0.0


def test_schedule_curve_returns_series() -> None:
    module = _load_script("scripts/schedule_curve_spike.py")

    result = module.generate_schedule_data(steps=5, initial_temperature=1.0, cooling_rate=0.99)

    assert len(result["temperature"]) == 6
    assert len(result["w2_effective"]) == 6
    assert result["w2_leads_temperature"] is False


def test_schedule_curve_handles_negative_steps() -> None:
    module = _load_script("scripts/schedule_curve_spike.py")

    result = module.generate_schedule_data(steps=-1)

    assert result["temperature"] == []
    assert result["w2_effective"] == []
    assert result["w2_leads_temperature"] is False


def test_parameter_sweep_returns_rows(tmp_path: Path) -> None:
    module = _load_script("scripts/parameter_sweep.py")
    module.run_experiment = lambda task_name, config, output_root: type(
        "Summary", (), {"completed_tasks": [task_name]}
    )()

    rows = module.run_parameter_sweep(
        output_root=tmp_path,
        seeds=[0],
        grid=[
            {
                "n_stagnation": 1,
                "mutation_stagnation_window": 1,
            }
        ],
        max_steps=1,
    )

    assert len(rows) == 1
    assert rows[0]["n_stagnation"] == 1
    assert "median_completed_tasks" in rows[0]


def test_parameter_sweep_uses_docker_by_default(tmp_path: Path) -> None:
    module = _load_script("scripts/parameter_sweep.py")
    called = {"backends": []}

    def fake_run_experiment(task_name, config, output_root):
        _ = task_name
        _ = output_root
        called["backends"].append(config.sandbox_backend)

        class Summary:
            completed_tasks = ["two_sum_sorted"]

        return Summary()

    module.run_experiment = fake_run_experiment

    module.run_parameter_sweep(
        output_root=tmp_path,
        seeds=[0],
        grid=[{"n_stagnation": 1, "mutation_stagnation_window": 1}],
        max_steps=1,
    )

    assert called["backends"] == ["docker"]


def test_parameter_sweep_rejects_process_backend_without_unsafe_opt_in(tmp_path: Path) -> None:
    module = _load_script("scripts/parameter_sweep.py")

    try:
        module.run_parameter_sweep(
            output_root=tmp_path,
            seeds=[0],
            grid=[{"n_stagnation": 1, "mutation_stagnation_window": 1}],
            max_steps=1,
            sandbox_backend="process",
        )
    except ValueError as exc:
        assert "unsafe" in str(exc)
    else:
        raise AssertionError("expected ValueError for process backend without opt-in")


def test_parameter_sweep_respects_task_name_argument(tmp_path: Path) -> None:
    module = _load_script("scripts/parameter_sweep.py")
    called = {"task_names": []}

    def fake_run_experiment(task_name, config, output_root):
        _ = config
        _ = output_root
        called["task_names"].append(task_name)

        class Summary:
            completed_tasks = [task_name]

        return Summary()

    module.run_experiment = fake_run_experiment

    module.run_parameter_sweep(
        task_name="slugify",
        output_root=tmp_path,
        seeds=[0],
        grid=[{"n_stagnation": 1, "mutation_stagnation_window": 1}],
        max_steps=1,
    )

    assert called["task_names"] == ["slugify"]


def test_parameter_sweep_writes_output_within_output_root(tmp_path: Path) -> None:
    module = _load_script("scripts/parameter_sweep.py")
    output_path = "artifacts/sweep.json"
    module.run_experiment = lambda task_name, config, output_root: type(
        "Summary", (), {"completed_tasks": [task_name]}
    )()

    module.run_parameter_sweep(
        output_root=tmp_path,
        output_path=output_path,
        seeds=[0],
        grid=[{"n_stagnation": 1, "mutation_stagnation_window": 1}],
        max_steps=1,
    )

    assert (tmp_path / output_path).exists()


def test_parameter_sweep_rejects_output_path_outside_output_root(tmp_path: Path) -> None:
    module = _load_script("scripts/parameter_sweep.py")

    try:
        module.run_parameter_sweep(
            output_root=tmp_path,
            output_path="../escape.json",
            seeds=[0],
            grid=[{"n_stagnation": 1, "mutation_stagnation_window": 1}],
            max_steps=1,
        )
    except ValueError as exc:
        assert "output_root" in str(exc)
    else:
        raise AssertionError("expected ValueError for output_path outside output_root")


def test_parameter_sweep_fails_fast_when_docker_unavailable(tmp_path: Path) -> None:
    module = _load_script("scripts/parameter_sweep.py")

    def fail_run_experiment(*args, **kwargs):
        raise RuntimeError("Docker daemon unavailable for docker sandbox backend")

    module.run_experiment = fail_run_experiment

    try:
        module.run_parameter_sweep(
            output_root=tmp_path,
            seeds=[0],
            grid=[{"n_stagnation": 1, "mutation_stagnation_window": 1}],
            max_steps=1,
        )
    except RuntimeError as exc:
        assert "sweep aborted" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError when docker daemon is unavailable")


def test_metrics_report_handles_empty_metrics_log(tmp_path: Path) -> None:
    module = _load_script("scripts/metrics_report.py")
    log_path = tmp_path / "run.jsonl"
    log_path.write_text(
        '{"schema_version":2,"event_type":"run.started","run_id":"r1","mode":"population","task":null,"step":0,"timestamp":"2026-01-01T00:00:00+00:00","payload":{}}\n',
        encoding="utf-8",
    )

    payload = module.summarize_metrics(log_path)

    assert payload["schema_version"] == 2
    assert payload["generations"] == 0
    assert payload["last_metrics"] == {}
    assert payload["bad_lines"] == 0


def test_metrics_report_returns_latest_generation_metrics(tmp_path: Path) -> None:
    module = _load_script("scripts/metrics_report.py")
    log_path = tmp_path / "run.jsonl"
    log_path.write_text(
        "\n".join(
            [
                '{"schema_version":2,"event_type":"run.started","run_id":"r1","mode":"population","task":null,"step":0,"timestamp":"2026-01-01T00:00:00+00:00","payload":{}}',
                '{"schema_version":2,"event_type":"generation.metrics","run_id":"r1","mode":"population","task":"two_sum_sorted","step":0,"timestamp":"2026-01-01T00:00:01+00:00","payload":{"generation":0,"shannon_entropy":0.2}}',
                '{"schema_version":2,"event_type":"generation.metrics","run_id":"r1","mode":"population","task":"two_sum_sorted","step":1,"timestamp":"2026-01-01T00:00:02+00:00","payload":{"generation":1,"shannon_entropy":0.4}}',
            ]
        ),
        encoding="utf-8",
    )

    payload = module.summarize_metrics(log_path)

    assert payload["generations"] == 2
    assert payload["last_metrics"]["generation"] == 1
    assert payload["last_metrics"]["shannon_entropy"] == 0.4
    assert payload["bad_lines"] == 0


def test_metrics_report_does_not_require_read_text(monkeypatch, tmp_path: Path) -> None:
    module = _load_script("scripts/metrics_report.py")
    log_path = tmp_path / "run.jsonl"
    log_path.write_text(
        '{"schema_version":2,"event_type":"generation.metrics","run_id":"r1","mode":"population","task":"two_sum_sorted","step":0,"timestamp":"2026-01-01T00:00:01+00:00","payload":{"generation":0}}\n',
        encoding="utf-8",
    )

    def fail_read_text(*args, **kwargs):  # noqa: ANN002,ANN003
        _ = args
        _ = kwargs
        raise AssertionError("read_text should not be called")

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    payload = module.summarize_metrics(log_path)

    assert payload["generations"] == 1


def test_metrics_report_skips_malformed_lines(tmp_path: Path) -> None:
    module = _load_script("scripts/metrics_report.py")
    log_path = tmp_path / "run.jsonl"
    log_path.write_text(
        "\n".join(
            [
                '{"schema_version":2,"event_type":"generation.metrics","run_id":"r1","mode":"population","task":"two_sum_sorted","step":0,"timestamp":"2026-01-01T00:00:01+00:00","payload":{"generation":0}}',
                "{not-json}",
                "[]",
                '{"schema_version":2,"event_type":"generation.metrics","run_id":"r1","mode":"population","task":"two_sum_sorted","step":1,"timestamp":"2026-01-01T00:00:02+00:00","payload":{"generation":1}}',
            ]
        ),
        encoding="utf-8",
    )

    payload = module.summarize_metrics(log_path)

    assert payload["generations"] == 2
    assert payload["last_metrics"]["generation"] == 1
    assert payload["bad_lines"] == 2
