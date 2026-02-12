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
        allow_unsafe_process_backend_override=True,
    )

    assert loaded.seed == 7
    assert loaded.sandbox_backend == "process"
    assert loaded.initial_energy == 1.2
    assert loaded.max_steps == 10
    assert loaded.cooling_rate == 0.99
    assert loaded.bootstrap_backend == "ollama"
    assert loaded.ollama_model == "gpt-oss:20b"
    assert loaded.allow_unsafe_process_backend is True


def test_run_experiment_writes_reproducible_logs_and_artifacts(tmp_path: Path) -> None:
    config = RunConfig(
        seed=7,
        sandbox_backend="process",
        bootstrap_backend="static",
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
    assert payloads[0]["event_type"] == "run.started"
    assert payloads[0]["payload"]["random_seed"] == 7
    assert "python_version" in payloads[0]["payload"]
    assert "cpu_architecture" in payloads[0]["payload"]
    assert "framework_git_sha" in payloads[0]["payload"]
    assert "parameters" in payloads[0]["payload"]

    assert any(item["event_type"] == "step.evaluated" for item in payloads)
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
        allow_unsafe_process_backend=True,
    )

    def fail_generate_seed(*_args, **_kwargs) -> str:
        raise BootstrapError("ollama unavailable")

    monkeypatch.setattr(runtime, "generate_seed", fail_generate_seed)

    summary = run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)

    payloads = [
        json.loads(line) for line in summary.log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert payloads[0]["event_type"] == "run.started"
    assert payloads[0]["payload"]["parameters"]["bootstrap_backend"] == "ollama"
    assert payloads[1]["event_type"] == "step.evaluated"
    assert payloads[1]["payload"]["bootstrap_fallback_used"] is True
    assert payloads[1]["payload"]["bootstrap_model"] == ""


def test_run_experiment_does_not_swallow_unexpected_bootstrap_errors(
    monkeypatch, tmp_path: Path
) -> None:
    from alife_core import runtime

    config = RunConfig(
        sandbox_backend="process",
        bootstrap_backend="ollama",
        bootstrap_fallback_to_static=True,
        allow_unsafe_process_backend=True,
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


def test_run_experiment_rejects_unsafe_ollama_process_without_opt_in(tmp_path: Path) -> None:
    config = RunConfig(
        sandbox_backend="process",
        bootstrap_backend="ollama",
        allow_unsafe_process_backend=False,
    )

    try:
        run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)
    except ValueError as exc:
        assert "unsafe" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsafe process backend combination")


def test_run_experiment_allows_unsafe_ollama_process_with_explicit_opt_in(
    monkeypatch, tmp_path: Path
) -> None:
    from alife_core import runtime

    config = RunConfig(
        sandbox_backend="process",
        bootstrap_backend="ollama",
        allow_unsafe_process_backend=True,
        max_steps=1,
        n_stagnation=1,
    )
    monkeypatch.setattr(runtime, "generate_seed", lambda *_args, **_kwargs: static_seed_payload())

    summary = run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)
    assert summary.run_id


def static_seed_payload() -> str:
    return (
        "def two_sum_sorted(numbers, target):\n"
        "    for i in range(len(numbers)):\n"
        "        for j in range(i + 1, len(numbers)):\n"
        "            if numbers[i] + numbers[j] == target:\n"
        "                return (i + 1, j + 1)\n"
        "    return (1, 1)\n"
    )


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
        item
        for item in payloads
        if item.get("event_type") == "step.evaluated" and item.get("step") == 0
    ]

    assert len(step0_events) >= 2
    assert step0_events[1]["payload"]["energy"] < config.initial_energy


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
    step_events = [payload for payload in payloads if payload["event_type"] == "step.evaluated"]

    assert "rolling_improvement_rate" in step_events[-1]["payload"]
    assert "rolling_validity_rate" in step_events[-1]["payload"]
    assert "mutation_fallback_active" in step_events[-1]["payload"]


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
    step_events = [item for item in payloads if item.get("event_type") == "step.evaluated"]
    assert any(item["payload"].get("mutation_fallback_active") for item in step_events[1:])


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


def test_run_experiment_population_mode_writes_generation_events(tmp_path: Path) -> None:
    config = RunConfig(
        seed=3,
        sandbox_backend="process",
        bootstrap_backend="static",
        evolution_mode="population",
        population_size=4,
        elite_count=1,
        tournament_k=2,
        crossover_rate=0.5,
        mutation_rate=0.9,
        max_generations=3,
    )

    summary = run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)
    payloads = [
        json.loads(line) for line in summary.log_path.read_text(encoding="utf-8").splitlines()
    ]

    generation_end = [item for item in payloads if item.get("event_type") == "generation.ended"]
    diversity_snapshot = [
        item for item in payloads if item.get("event_type") == "generation.metrics"
    ]
    assert generation_end
    assert diversity_snapshot


def test_run_experiment_population_mode_completes_task_when_thresholds_are_easy(
    tmp_path: Path,
) -> None:
    config = RunConfig(
        seed=2,
        sandbox_backend="process",
        bootstrap_backend="static",
        evolution_mode="population",
        population_size=4,
        elite_count=1,
        tournament_k=2,
        max_generations=2,
        pass_ratio_threshold=0.0,
        fitness_threshold=0.0,
    )

    summary = run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)

    assert "two_sum_sorted" in summary.completed_tasks


def test_run_experiment_fails_fast_when_docker_unavailable(monkeypatch, tmp_path: Path) -> None:
    from alife_core import runtime

    config = RunConfig(
        sandbox_backend="docker",
        bootstrap_backend="static",
    )

    class Failed:
        returncode = 1
        stdout = ""
        stderr = "daemon unavailable"

    monkeypatch.setattr(runtime.subprocess, "run", lambda *args, **kwargs: Failed())

    try:
        run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)
    except RuntimeError as exc:
        assert "docker daemon" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError when docker daemon is unavailable")


def test_ast_shape_fingerprint_ignores_names_and_constants() -> None:
    from alife_core.metrics.evolution import ast_shape_fingerprint

    left = "def solve(x):\n    value = x + 1\n    return value\n"
    right = "def solve(y):\n    other = y + 2\n    return other\n"

    assert ast_shape_fingerprint(left) == ast_shape_fingerprint(right)


def test_crossover_code_returns_parseable_candidate_for_structured_parents() -> None:
    import ast
    import random

    from alife_core import runtime

    parent_a = (
        "def solve(x):\n"
        "    if x > 0:\n"
        "        y = x + 1\n"
        "    else:\n"
        "        y = x - 1\n"
        "    return y\n"
    )
    parent_b = "def solve(x):\n    if x % 2 == 0:\n        return x // 2\n    return x * 3\n"

    candidate = runtime._crossover_code(parent_a, parent_b, random.Random(1))
    ast.parse(candidate)


def test_evaluate_population_uses_parallel_workers(monkeypatch) -> None:
    import threading
    import time

    from alife_core import runtime
    from alife_core.models import EvaluationResult
    from alife_core.tasks.builtin import load_builtin_tasks

    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_evaluate(code, task, edit_cost, config):
        nonlocal active, max_active
        _ = code
        _ = task
        _ = edit_cost
        _ = config
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return EvaluationResult(
            train_pass_ratio=0.0,
            hidden_pass_ratio=0.0,
            ast_edit_cost=0.0,
            fitness=0.0,
            train_failures=1,
            hidden_failures=1,
        )

    monkeypatch.setattr(runtime, "evaluate_candidate", fake_evaluate)
    task = load_builtin_tasks()["two_sum_sorted"]
    config = RunConfig(sandbox_backend="process", population_workers=4)
    runtime._evaluate_population(
        ["def two_sum_sorted(numbers, target):\n    return (1, 2)\n"] * 4,
        task,
        config,
    )

    assert max_active > 1


def test_population_mode_logs_step_and_energy_keys(tmp_path: Path) -> None:
    config = RunConfig(
        seed=4,
        sandbox_backend="process",
        bootstrap_backend="static",
        evolution_mode="population",
        population_size=4,
        elite_count=1,
        max_generations=2,
    )
    summary = run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)
    payloads = [
        json.loads(line) for line in summary.log_path.read_text(encoding="utf-8").splitlines()
    ]
    generation_end = next(item for item in payloads if item.get("event_type") == "generation.ended")

    assert "step" in generation_end
    assert generation_end["mode"] == "population"
    assert "timestamp" in generation_end


def test_population_mode_checks_final_generation_candidate(monkeypatch, tmp_path: Path) -> None:
    from alife_core import runtime
    from alife_core.models import OrganismState
    from alife_core.tasks.builtin import load_builtin_tasks

    task = load_builtin_tasks()["two_sum_sorted"]
    init_codes = [static_seed_payload()] * 4
    calls = {"count": 0}

    monkeypatch.setattr(
        runtime,
        "_ensure_docker_daemon_available",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runtime,
        "_initialize_population_for_task",
        lambda *_args, **_kwargs: init_codes,
    )

    def fake_evaluate_population(codes, task_obj, config):
        _ = codes
        _ = task_obj
        _ = config
        calls["count"] += 1
        if calls["count"] == 1:
            return [
                OrganismState(
                    code=static_seed_payload(),
                    fitness=0.0,
                    train_pass_ratio=0.0,
                    hidden_pass_ratio=0.0,
                )
            ] * 4
        return [
            OrganismState(
                code=static_seed_payload(),
                fitness=1.0,
                train_pass_ratio=1.0,
                hidden_pass_ratio=1.0,
            )
        ] * 4

    monkeypatch.setattr(runtime, "_evaluate_population", fake_evaluate_population)
    monkeypatch.setattr(runtime, "load_builtin_tasks", lambda: {"two_sum_sorted": task})
    monkeypatch.setattr(
        runtime,
        "_mutate_code",
        lambda code, rng, intensity=1, prefer_structural=False: code,
    )

    config = RunConfig(
        seed=0,
        sandbox_backend="process",
        bootstrap_backend="static",
        evolution_mode="population",
        population_size=4,
        elite_count=1,
        tournament_k=2,
        max_generations=1,
        pass_ratio_threshold=0.9,
        fitness_threshold=0.9,
    )

    summary = run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)
    assert "two_sum_sorted" in summary.completed_tasks


def test_load_run_config_rejects_invalid_population_rates(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "crossover_rate: 1.1",
                "mutation_rate: -0.1",
                "population_workers: 0",
            ]
        ),
        encoding="utf-8",
    )

    try:
        load_run_config(config_path)
    except ValueError as exc:
        assert "crossover_rate" in str(exc) or "mutation_rate" in str(exc) or "workers" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid population config")


def test_run_experiment_rejects_invalid_population_workers(tmp_path: Path) -> None:
    config = RunConfig(
        sandbox_backend="process",
        bootstrap_backend="static",
        population_workers=0,
    )
    try:
        run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)
    except ValueError as exc:
        assert "population_workers" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid population_workers")


def test_load_run_config_rejects_invalid_elite_count_for_population_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "evolution_mode: population",
                "population_size: 4",
                "elite_count: 4",
            ]
        ),
        encoding="utf-8",
    )

    try:
        load_run_config(config_path)
    except ValueError as exc:
        assert "elite_count" in str(exc)
    else:
        raise AssertionError("expected ValueError for invalid elite_count")


def test_population_mode_emits_schema_v2_generation_metrics(tmp_path: Path) -> None:
    config = RunConfig(
        seed=6,
        sandbox_backend="process",
        bootstrap_backend="static",
        evolution_mode="population",
        population_size=4,
        elite_count=1,
        max_generations=2,
    )
    summary = run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)
    payloads = [
        json.loads(line) for line in summary.log_path.read_text(encoding="utf-8").splitlines()
    ]

    run_started = payloads[0]
    assert run_started["schema_version"] == 2
    assert run_started["event_type"] == "run.started"
    assert "payload" in run_started
    generation_metrics = [
        item for item in payloads if item.get("event_type") == "generation.metrics"
    ]
    assert generation_metrics
    assert "shannon_entropy" in generation_metrics[0]["payload"]


def test_step_event_includes_execution_status(tmp_path: Path) -> None:
    config = RunConfig(
        seed=7,
        sandbox_backend="process",
        bootstrap_backend="static",
        max_steps=2,
        n_stagnation=1,
    )

    summary = run_experiment(task_name="two_sum_sorted", config=config, output_root=tmp_path)
    payloads = [
        json.loads(line) for line in summary.log_path.read_text(encoding="utf-8").splitlines()
    ]
    step_events = [item for item in payloads if item["event_type"] == "step.evaluated"]

    assert step_events
    for event in step_events:
        assert "execution_status" in event["payload"]
        assert "hard_failure" in event["payload"]


def test_mutate_constant_distributes_across_all_constants() -> None:
    import ast

    from alife_core import runtime

    source = "def solve(x):\n    a = 1\n    b = 2\n    c = 3\n    return a + b + c\n"
    hit_counts = {1: 0, 2: 0, 3: 0}

    for seed in range(100):
        rng = __import__("random").Random(seed)
        tree = ast.parse(source)
        changed = runtime._mutate_constant(tree, rng)
        if changed:
            rendered = ast.unparse(tree)
            for original_val in (1, 2, 3):
                original_str = f"a = {original_val}" if original_val == 1 else (
                    f"b = {original_val}" if original_val == 2 else f"c = {original_val}"
                )
                if original_str not in rendered:
                    hit_counts[original_val] += 1

    for val, count in hit_counts.items():
        assert count >= 10, f"constant {val} was mutated only {count}/100 times"


def test_mutate_binop_distributes_across_all_add_ops() -> None:
    import ast

    from alife_core import runtime

    source = "def solve(x, y, z):\n    return (x + y) + z\n"
    # Track which Add node got mutated by checking structure
    first_changed = 0
    second_changed = 0

    for seed in range(100):
        rng = __import__("random").Random(seed)
        tree = ast.parse(source)
        changed = runtime._mutate_binop(tree, rng)
        if changed:
            rendered = ast.unparse(tree)
            # If outer Add changed, inner x + y remains
            # If inner Add changed, outer + z remains but inner uses different op
            inner_tree = ast.parse(rendered)
            func_body = inner_tree.body[0].body[0].value  # type: ignore[attr-defined]
            if isinstance(func_body, ast.BinOp):
                if not isinstance(func_body.op, ast.Add):
                    second_changed += 1
                elif isinstance(func_body.left, ast.BinOp) and not isinstance(
                    func_body.left.op, ast.Add
                ):
                    first_changed += 1

    assert first_changed >= 10, f"inner Add mutated only {first_changed}/100 times"
    assert second_changed >= 10, f"outer Add mutated only {second_changed}/100 times"


def test_mutate_compare_distributes_across_comparisons() -> None:
    import ast

    from alife_core import runtime

    source = (
        "def solve(x, y):\n"
        "    if x < y:\n"
        "        return x\n"
        "    if x > y:\n"
        "        return y\n"
        "    return 0\n"
    )
    first_changed = 0
    second_changed = 0

    for seed in range(100):
        rng = __import__("random").Random(seed)
        tree = ast.parse(source)
        changed = runtime._mutate_compare(tree, rng)
        if changed:
            rendered = ast.unparse(tree)
            if "x < y" not in rendered and "x > y" in rendered:
                first_changed += 1
            elif "x > y" not in rendered and "x < y" in rendered:
                second_changed += 1

    assert first_changed >= 10, f"first Compare mutated only {first_changed}/100 times"
    assert second_changed >= 10, f"second Compare mutated only {second_changed}/100 times"


def test_mutate_constant_returns_false_when_no_constants() -> None:
    import ast

    from alife_core import runtime

    tree = ast.parse("def solve(x):\n    return x\n")
    rng = __import__("random").Random(0)
    assert runtime._mutate_constant(tree, rng) is False


def test_evaluate_population_skips_pre_evaluated_organisms(monkeypatch) -> None:
    from alife_core import runtime
    from alife_core.models import EvaluationResult, OrganismState
    from alife_core.tasks.builtin import load_builtin_tasks

    calls = {"count": 0}

    def fake_evaluate(code, task, edit_cost, config):
        _ = code
        _ = task
        _ = edit_cost
        _ = config
        calls["count"] += 1
        return EvaluationResult(
            train_pass_ratio=1.0,
            hidden_pass_ratio=1.0,
            ast_edit_cost=0.0,
            fitness=1.0,
            train_failures=0,
            hidden_failures=0,
        )

    monkeypatch.setattr(runtime, "evaluate_candidate", fake_evaluate)
    task = load_builtin_tasks()["two_sum_sorted"]
    config = RunConfig(sandbox_backend="process", population_workers=1)
    organisms = [
        OrganismState(
            organism_id="elite-1",
            parent_ids=(),
            birth_generation=0,
            code=static_seed_payload(),
            fitness=0.7,
            train_pass_ratio=0.8,
            hidden_pass_ratio=0.8,
            ast_nodes=10,
            ast_depth=3,
            shape_fingerprint="x",
            lineage_depth=2,
            evaluated=True,
        ),
        OrganismState(
            organism_id="child-1",
            parent_ids=("elite-1",),
            birth_generation=1,
            code=static_seed_payload(),
            fitness=0.0,
            train_pass_ratio=0.0,
            hidden_pass_ratio=0.0,
            lineage_depth=3,
            evaluated=False,
        ),
    ]

    evaluated = runtime._evaluate_population(organisms, task, config)

    assert calls["count"] == 1
    assert evaluated[0].evaluated is True
    assert evaluated[0].fitness == 0.7
    assert evaluated[1].evaluated is True
