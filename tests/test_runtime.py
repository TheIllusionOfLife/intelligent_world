import json
from pathlib import Path

from alife_core.models import RunConfig
from alife_core.runtime import load_run_config, run_experiment


def test_load_run_config_reads_yaml_and_applies_seed_override(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 1",
                "w1_pass_ratio: 0.8",
                "w2_ast_edit_cost: 0.2",
                "base_survival_cost: 0.01",
                "pass_ratio_threshold: 0.7",
                "fitness_threshold: 0.6",
                "exec_timeout_seconds: 1.0",
                "sandbox_backend: process",
                "docker_image: python:3.12-slim",
                "initial_energy: 1.2",
                "max_steps: 10",
                "n_stagnation: 5",
                "improvement_multiplier: 1.5",
                "degradation_multiplier: 1.1",
                "initial_temperature: 1.0",
                "cooling_rate: 9.9e-1  # scientific notation",
                "w2_floor: 0.02",
                "decay_factor: 0.999",
                "mutation_stagnation_window: 3",
                "goodhart_gap_threshold: 0.2",
                "bootstrap_backend: static",
                "ollama_model: gpt-oss:20b",
                "bootstrap_timeout_seconds: 3.0",
                "bootstrap_fallback_to_static: true",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_run_config(
        config_path,
        seed_override=7,
        bootstrap_backend_override="ollama",
        ollama_model_override="gpt-oss:20b",
    )

    assert loaded.seed == 7
    assert loaded.sandbox_backend == "process"
    assert loaded.initial_energy == 1.2
    assert loaded.max_steps == 10
    assert loaded.cooling_rate == 0.99
    assert loaded.bootstrap_backend == "ollama"
    assert loaded.ollama_model == "gpt-oss:20b"


def test_run_experiment_writes_reproducible_logs_and_artifacts(tmp_path: Path) -> None:
    config = RunConfig(
        seed=7,
        sandbox_backend="process",
        max_steps=8,
        n_stagnation=4,
        pass_ratio_threshold=0.9,
        fitness_threshold=0.3,
    )

    summary = run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)

    assert summary.run_id
    assert summary.log_path.exists()
    assert (tmp_path / "organisms" / "current" / "two_sum_sorted.py").exists()

    lines = summary.log_path.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines]
    assert payloads[0]["type"] == "run_start"
    assert payloads[0]["random_seed"] == 7
    assert "python_version" in payloads[0]
    assert "cpu_architecture" in payloads[0]
    assert "framework_git_sha" in payloads[0]
    assert "parameters" in payloads[0]

    assert any(item["type"] == "step" for item in payloads)
    assert summary.completed_tasks == ["two_sum_sorted"]


def test_run_experiment_generates_unique_run_ids(tmp_path: Path) -> None:
    config = RunConfig(
        seed=7,
        sandbox_backend="process",
        max_steps=1,
        n_stagnation=1,
        bootstrap_backend="static",
    )

    first = run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)
    second = run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)

    assert first.run_id != second.run_id


def test_run_experiment_falls_back_to_static_seed_when_bootstrap_fails(
    monkeypatch, tmp_path: Path
) -> None:
    from alife_core import runtime
    from alife_core.bootstrap import BootstrapError

    config = RunConfig(
        seed=11,
        sandbox_backend="process",
        max_steps=2,
        n_stagnation=1,
        bootstrap_backend="ollama",
        ollama_model="gpt-oss:20b",
        bootstrap_fallback_to_static=True,
    )

    def fail_generate_seed(*_args, **_kwargs) -> str:
        raise BootstrapError("ollama unavailable")

    monkeypatch.setattr(runtime, "generate_seed", fail_generate_seed)

    summary = run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)

    payloads = [
        json.loads(line) for line in summary.log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert payloads[0]["type"] == "run_start"
    assert payloads[0]["parameters"]["bootstrap_backend"] == "ollama"
    assert payloads[1]["type"] == "step"
    assert payloads[1]["bootstrap_fallback_used"] is True
    assert payloads[1]["bootstrap_model"] == ""


def test_run_experiment_does_not_swallow_unexpected_bootstrap_errors(
    monkeypatch, tmp_path: Path
) -> None:
    from alife_core import runtime

    config = RunConfig(
        sandbox_backend="process",
        bootstrap_backend="ollama",
        bootstrap_fallback_to_static=True,
    )

    def fail_generate_seed(*_args, **_kwargs) -> str:
        raise ValueError("unexpected")

    monkeypatch.setattr(runtime, "generate_seed", fail_generate_seed)

    try:
        run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)
    except ValueError as exc:
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("expected ValueError to propagate")


def test_run_experiment_curriculum_mode_advances_tasks(tmp_path: Path) -> None:
    config = RunConfig(
        seed=5,
        sandbox_backend="process",
        max_steps=4,
        n_stagnation=2,
        bootstrap_backend="static",
        run_curriculum=True,
        pass_ratio_threshold=0.1,
        fitness_threshold=0.05,
    )

    summary = run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)

    assert "two_sum_sorted" in summary.completed_tasks
    assert len(summary.completed_tasks) >= 2


def test_curriculum_mode_carries_energy_across_tasks(tmp_path: Path) -> None:
    config = RunConfig(
        seed=5,
        sandbox_backend="process",
        bootstrap_backend="static",
        run_curriculum=True,
        pass_ratio_threshold=0.1,
        fitness_threshold=0.05,
        max_steps=2,
        n_stagnation=1,
        base_survival_cost=0.05,
    )

    summary = run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)
    payloads = [
        json.loads(line) for line in summary.log_path.read_text(encoding="utf-8").splitlines()
    ]
    step0_events = [
        item for item in payloads if item.get("type") == "step" and item.get("step") == 0
    ]

    assert len(step0_events) >= 2
    assert step0_events[1]["energy"] < config.initial_energy


def test_step_logs_include_mutation_viability_metrics(tmp_path: Path) -> None:
    config = RunConfig(
        seed=2,
        sandbox_backend="process",
        max_steps=3,
        n_stagnation=2,
        bootstrap_backend="static",
    )

    summary = run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)
    payloads = [
        json.loads(line) for line in summary.log_path.read_text(encoding="utf-8").splitlines()
    ]
    step_events = [payload for payload in payloads if payload["type"] == "step"]

    assert "rolling_improvement_rate" in step_events[-1]
    assert "rolling_validity_rate" in step_events[-1]
    assert "mutation_fallback_active" in step_events[-1]


def test_mutation_fallback_changes_mutation_mode_when_threshold_breached(tmp_path: Path) -> None:
    config = RunConfig(
        seed=1,
        sandbox_backend="process",
        bootstrap_backend="static",
        max_steps=5,
        n_stagnation=4,
        pass_ratio_threshold=1.1,
        fitness_threshold=1.1,
        viability_window=1,
        viability_min_improvement_rate=1.0,
    )

    summary = run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)
    payloads = [
        json.loads(line) for line in summary.log_path.read_text(encoding="utf-8").splitlines()
    ]
    step_events = [item for item in payloads if item.get("type") == "step"]
    assert any(item.get("mutation_fallback_active") for item in step_events[1:])


def test_mutate_code_can_adjust_comparison_operator() -> None:
    from alife_core import runtime

    class FakeRng:
        def random(self) -> float:
            return 0.0

        def choice(self, seq):  # noqa: ANN001
            return seq[0]

    source = "def solve(x, y):\n    if x < y:\n        return x\n    return y\n"

    mutated = runtime._mutate_code(source, FakeRng(), intensity=1)

    assert "if x < y" not in mutated


def test_mutate_statement_swap_can_operate_on_nested_body() -> None:
    import ast

    from alife_core import runtime

    class FakeRng:
        def random(self) -> float:
            return 0.0

        def choice(self, seq):  # noqa: ANN001
            return seq[0]

    tree = ast.parse("def solve(x):\n    if x > 0:\n        y = 1\n        z = 2\n")

    changed = runtime._mutate_statement_swap(tree, FakeRng())
    rendered = ast.unparse(tree)

    assert changed is True
    assert "z = 2" in rendered and "y = 1" in rendered
    assert rendered.index("z = 2") < rendered.index("y = 1")


def test_mutate_constant_supports_float_literals() -> None:
    import ast

    from alife_core import runtime

    class FakeRng:
        def random(self) -> float:
            return 0.0

        def choice(self, seq):  # noqa: ANN001
            return seq[0]

    tree = ast.parse("def solve(x):\n    return x < 0.5\n")
    changed = runtime._mutate_constant(tree, FakeRng())
    rendered = ast.unparse(tree)

    assert changed is True
    assert "0.5" not in rendered
