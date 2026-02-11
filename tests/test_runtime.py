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
                "cooling_rate: 0.99",
                "w2_floor: 0.02",
                "decay_factor: 0.999",
                "mutation_stagnation_window: 3",
                "goodhart_gap_threshold: 0.2",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_run_config(config_path, seed_override=7)

    assert loaded.seed == 7
    assert loaded.sandbox_backend == "process"
    assert loaded.initial_energy == 1.2
    assert loaded.max_steps == 10


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
