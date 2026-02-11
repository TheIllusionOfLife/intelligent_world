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
