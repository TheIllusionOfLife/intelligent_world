import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.experiment_campaign import (
    DEFAULT_SEEDS,
    DEFAULT_TASKS,
    build_run_configs,
    extract_run_metrics,
    run_campaign,
)


class TestBuildRunConfigs:
    def test_generates_correct_number_of_combinations(self) -> None:
        configs = build_run_configs()
        assert len(configs) == 4 * 3 * 5  # 4 configs × 3 tasks × 5 seeds = 60

    def test_all_seeds_present(self) -> None:
        configs = build_run_configs()
        seeds = {c["seed"] for c in configs}
        assert seeds == set(DEFAULT_SEEDS)

    def test_all_tasks_present(self) -> None:
        configs = build_run_configs()
        tasks = {c["task"] for c in configs}
        assert tasks == set(DEFAULT_TASKS)

    def test_all_config_ids_present(self) -> None:
        configs = build_run_configs()
        config_ids = {c["config_id"] for c in configs}
        assert config_ids == {"A", "B", "C", "D"}

    def test_config_a_is_single_agent_no_semantic(self) -> None:
        configs = build_run_configs()
        a_configs = [c for c in configs if c["config_id"] == "A"]
        for c in a_configs:
            assert c["run_config"].evolution_mode == "single_agent"
            assert c["run_config"].enable_semantic_mutation is False

    def test_config_b_is_single_agent_with_semantic(self) -> None:
        configs = build_run_configs()
        b_configs = [c for c in configs if c["config_id"] == "B"]
        for c in b_configs:
            assert c["run_config"].evolution_mode == "single_agent"
            assert c["run_config"].enable_semantic_mutation is True

    def test_config_c_is_population_no_semantic(self) -> None:
        configs = build_run_configs()
        c_configs = [c for c in configs if c["config_id"] == "C"]
        for c in c_configs:
            assert c["run_config"].evolution_mode == "population"
            assert c["run_config"].enable_semantic_mutation is False

    def test_config_d_is_population_with_semantic(self) -> None:
        configs = build_run_configs()
        d_configs = [c for c in configs if c["config_id"] == "D"]
        for c in d_configs:
            assert c["run_config"].evolution_mode == "population"
            assert c["run_config"].enable_semantic_mutation is True

    def test_all_use_static_bootstrap(self) -> None:
        configs = build_run_configs()
        for c in configs:
            assert c["run_config"].bootstrap_backend == "static"

    def test_all_use_process_sandbox(self) -> None:
        configs = build_run_configs()
        for c in configs:
            assert c["run_config"].sandbox_backend == "process"
            assert c["run_config"].allow_unsafe_process_backend is True

    def test_custom_seeds_and_tasks(self) -> None:
        configs = build_run_configs(seeds=[1, 2], tasks=["slugify"])
        assert len(configs) == 4 * 1 * 2  # 4 configs × 1 task × 2 seeds

    def test_population_configs_use_specified_params(self) -> None:
        configs = build_run_configs(
            max_generations=25, population_size=4, seeds=[0], tasks=["slugify"]
        )
        pop_configs = [c for c in configs if c["config_id"] in ("C", "D")]
        for c in pop_configs:
            assert c["run_config"].max_generations == 25
            assert c["run_config"].population_size == 4

    def test_single_agent_configs_use_specified_max_steps(self) -> None:
        configs = build_run_configs(max_steps=100, seeds=[0], tasks=["slugify"])
        sa_configs = [c for c in configs if c["config_id"] in ("A", "B")]
        for c in sa_configs:
            assert c["run_config"].max_steps == 100


class TestExtractRunMetrics:
    def test_extracts_from_single_agent_log(self, tmp_path: Path) -> None:
        log_path = tmp_path / "run.jsonl"
        events = [
            {
                "event_type": "step.evaluated",
                "payload": {
                    "fitness": 0.5,
                    "train_pass_ratio": 0.6,
                    "hidden_pass_ratio": 0.4,
                    "energy": 0.9,
                },
            },
            {
                "event_type": "step.evaluated",
                "payload": {
                    "fitness": 0.8,
                    "train_pass_ratio": 0.95,
                    "hidden_pass_ratio": 0.9,
                    "energy": 0.7,
                },
            },
            {
                "event_type": "run.completed",
                "payload": {"completed_tasks": ["two_sum_sorted"]},
            },
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

        metrics = extract_run_metrics(log_path, mode="single_agent")
        assert metrics["best_fitness"] == 0.8
        assert metrics["best_train_pass_ratio"] == 0.95
        assert metrics["best_hidden_pass_ratio"] == 0.9
        assert metrics["final_energy"] == 0.7
        assert metrics["total_steps"] == 2

    def test_extracts_from_population_log(self, tmp_path: Path) -> None:
        log_path = tmp_path / "run.jsonl"
        events = [
            {
                "event_type": "generation.metrics",
                "payload": {
                    "generation": 0,
                    "best_fitness": 0.3,
                    "best_train_pass_ratio": 0.4,
                    "best_hidden_pass_ratio": 0.2,
                    "shannon_entropy": 0.9,
                    "structural_diversity_ratio": 0.8,
                },
            },
            {
                "event_type": "generation.metrics",
                "payload": {
                    "generation": 1,
                    "best_fitness": 0.75,
                    "best_train_pass_ratio": 0.92,
                    "best_hidden_pass_ratio": 0.85,
                    "shannon_entropy": 0.7,
                    "structural_diversity_ratio": 0.6,
                },
            },
            {
                "event_type": "run.completed",
                "payload": {"completed_tasks": ["two_sum_sorted"]},
            },
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

        metrics = extract_run_metrics(log_path, mode="population")
        assert metrics["best_fitness"] == 0.75
        assert metrics["best_train_pass_ratio"] == 0.92
        assert metrics["best_hidden_pass_ratio"] == 0.85
        assert metrics["total_generations"] == 2
        assert len(metrics["diversity_trajectory"]) == 2

    def test_handles_convergence_event(self, tmp_path: Path) -> None:
        log_path = tmp_path / "run.jsonl"
        events = [
            {
                "event_type": "generation.metrics",
                "payload": {
                    "generation": 0,
                    "best_fitness": 0.3,
                    "best_train_pass_ratio": 0.4,
                    "best_hidden_pass_ratio": 0.2,
                    "shannon_entropy": 0.1,
                    "structural_diversity_ratio": 0.1,
                },
            },
            {
                "event_type": "evolution.converged",
                "payload": {"reason": "low_diversity"},
            },
            {
                "event_type": "run.completed",
                "payload": {"completed_tasks": []},
            },
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

        metrics = extract_run_metrics(log_path, mode="population")
        assert metrics["convergence_reason"] == "low_diversity"

    def test_empty_log_returns_defaults(self, tmp_path: Path) -> None:
        log_path = tmp_path / "run.jsonl"
        log_path.write_text("", encoding="utf-8")

        metrics = extract_run_metrics(log_path, mode="single_agent")
        assert metrics["best_fitness"] == 0.0
        assert metrics["total_steps"] == 0


class TestRunCampaign:
    def test_writes_summary_json(self, tmp_path: Path) -> None:
        call_count = {"n": 0}

        def fake_run_experiment(task_name, config, output_root):
            call_count["n"] += 1
            log_path = output_root / "logs" / f"run-{call_count['n']}.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            events = [
                {
                    "event_type": "step.evaluated",
                    "payload": {
                        "fitness": 0.5,
                        "train_pass_ratio": 0.6,
                        "hidden_pass_ratio": 0.4,
                        "energy": 0.8,
                    },
                },
                {
                    "event_type": "run.completed",
                    "payload": {"completed_tasks": []},
                },
            ]
            log_path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
            return SimpleNamespace(
                run_id=f"run-{call_count['n']}",
                log_path=log_path,
                completed_tasks=[],
            )

        with patch("scripts.experiment_campaign.run_experiment", fake_run_experiment):
            results = run_campaign(
                output_dir=tmp_path,
                seeds=[0],
                tasks=["slugify"],
            )

        summary_path = tmp_path / "summary.json"
        assert summary_path.exists()
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        assert len(data) == 4  # 4 configs × 1 task × 1 seed
        assert len(results) == 4

        for entry in data:
            assert "config_id" in entry
            assert "task" in entry
            assert "seed" in entry
            assert "solved" in entry
            assert "best_fitness" in entry

    def test_continues_on_failure(self, tmp_path: Path) -> None:
        call_count = {"n": 0}

        def failing_run_experiment(task_name, config, output_root):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("Simulated failure")
            log_path = output_root / "logs" / f"run-{call_count['n']}.jsonl"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            events = [
                {
                    "event_type": "step.evaluated",
                    "payload": {
                        "fitness": 0.5,
                        "train_pass_ratio": 0.6,
                        "hidden_pass_ratio": 0.4,
                        "energy": 0.8,
                    },
                },
                {
                    "event_type": "run.completed",
                    "payload": {"completed_tasks": []},
                },
            ]
            log_path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
            return SimpleNamespace(
                run_id=f"run-{call_count['n']}",
                log_path=log_path,
                completed_tasks=[],
            )

        output_dir = tmp_path
        with patch("scripts.experiment_campaign.run_experiment", failing_run_experiment):
            results = run_campaign(
                output_dir=output_dir,
                seeds=[0],
                tasks=["slugify"],
            )

        # Should have 4 results: 1 failed + 3 succeeded
        assert len(results) == 4
        failed = [r for r in results if r.get("error")]
        succeeded = [r for r in results if not r.get("error")]
        assert len(failed) == 1
        assert len(succeeded) == 3
