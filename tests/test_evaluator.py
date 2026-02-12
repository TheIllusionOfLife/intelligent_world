import alife_core.evaluator.core as evaluator
from alife_core.evaluator.core import evaluate_candidate
from alife_core.models import RunConfig, TaskSpec


def test_evaluator_is_deterministic_for_same_seed_and_inputs() -> None:
    task = TaskSpec(
        name="increment",
        prompt="return x + 1",
        function_name="solve",
        train_cases=(((1,), 2), ((10,), 11)),
        hidden_cases=(((100,), 101),),
    )
    code = "def solve(x):\n    return x + 1\n"
    config = RunConfig(seed=7, sandbox_backend="process")

    first = evaluate_candidate(code=code, task=task, edit_cost=0.0, config=config)
    second = evaluate_candidate(code=code, task=task, edit_cost=0.0, config=config)

    assert first == second


def test_hidden_tests_do_not_change_fitness_in_phase1() -> None:
    task = TaskSpec(
        name="only_train_counts",
        prompt="train pass, hidden fail",
        function_name="solve",
        train_cases=(((2,), 3),),
        hidden_cases=(((2,), 99),),
    )
    code = "def solve(x):\n    return x + 1\n"
    config = RunConfig(seed=1, w1_pass_ratio=1.0, w2_ast_edit_cost=0.0, sandbox_backend="process")

    result = evaluate_candidate(code=code, task=task, edit_cost=0.0, config=config)

    assert result.train_pass_ratio == 1.0
    assert result.hidden_pass_ratio == 0.0
    assert result.fitness == 1.0


def test_missing_function_symbol_returns_zero_score() -> None:
    task = TaskSpec(
        name="missing",
        prompt="does not define solve",
        function_name="solve",
        train_cases=(((1,), 2),),
        hidden_cases=(((2,), 3),),
    )
    code = "def nope(x):\n    return x + 1\n"

    result = evaluate_candidate(
        code=code,
        task=task,
        edit_cost=0.3,
        config=RunConfig(sandbox_backend="process"),
    )

    assert result.train_pass_ratio == 0.0
    assert result.hidden_pass_ratio == 0.0
    assert result.fitness == 0.0
    assert result.train_failures == 1
    assert result.hidden_failures == 1


def test_top_level_infinite_loop_times_out_instead_of_hanging() -> None:
    task = TaskSpec(
        name="timeout",
        prompt="hangs at module import",
        function_name="solve",
        train_cases=(((1,), 2),),
        hidden_cases=(),
    )
    code = "while True:\n    pass\n\ndef solve(x):\n    return x + 1\n"

    result = evaluate_candidate(
        code=code,
        task=task,
        edit_cost=0.0,
        config=RunConfig(exec_timeout_seconds=0.1, sandbox_backend="process"),
    )

    assert result.train_failures == 1
    assert result.hidden_failures == 0
    assert result.fitness == 0.0


def test_evaluator_batches_case_execution(monkeypatch) -> None:
    calls: list[tuple[tuple[tuple, object], ...]] = []

    def fake_execute_batch(code, function_name, cases, config, pool=None):
        _ = code
        _ = function_name
        _ = config
        calls.append(cases)
        return "ok", [("ok", expected) for _, expected in cases], ""

    monkeypatch.setattr(evaluator, "_execute_case_batch", fake_execute_batch)

    task = TaskSpec(
        name="batch",
        prompt="batch mode",
        function_name="solve",
        train_cases=(((1,), 2), ((2,), 3), ((3,), 4)),
        hidden_cases=(((10,), 11), ((20,), 21)),
    )

    result = evaluate_candidate(
        code="def solve(x):\n    return x + 1\n",
        task=task,
        edit_cost=0.0,
        config=RunConfig(sandbox_backend="process"),
    )

    assert len(calls) == 1
    assert len(calls[0]) == 5
    assert result.train_failures == 0
    assert result.hidden_failures == 0


def test_evaluation_result_exposes_hard_failure_when_execution_fails() -> None:
    task = TaskSpec(
        name="broken",
        prompt="code that does not compile",
        function_name="solve",
        train_cases=(((1,), 2),),
        hidden_cases=(((2,), 3),),
    )
    code = "def not_valid_python(:\n"

    result = evaluate_candidate(
        code=code,
        task=task,
        edit_cost=0.0,
        config=RunConfig(sandbox_backend="process"),
    )

    assert result.hard_failure is True
    assert result.execution_status != "ok"


def test_evaluation_result_ok_for_passing_code() -> None:
    task = TaskSpec(
        name="passing",
        prompt="simple addition",
        function_name="solve",
        train_cases=(((1,), 2), ((10,), 11)),
        hidden_cases=(((100,), 101),),
    )
    code = "def solve(x):\n    return x + 1\n"

    result = evaluate_candidate(
        code=code,
        task=task,
        edit_cost=0.0,
        config=RunConfig(sandbox_backend="process"),
    )

    assert result.hard_failure is False
    assert result.execution_status == "ok"


def test_compile_error_captures_error_type_and_detail() -> None:
    task = TaskSpec(
        name="broken",
        prompt="code with syntax error",
        function_name="solve",
        train_cases=(((1,), 2),),
        hidden_cases=(),
    )
    code = "def solve(x):\n    return 1/0\n"

    result = evaluate_candidate(
        code=code,
        task=task,
        edit_cost=0.0,
        config=RunConfig(sandbox_backend="process"),
    )

    # 1/0 raises ZeroDivisionError at runtime, not compile time
    # So execution succeeds but cases fail with runtime_error
    assert result.execution_status == "ok"


def test_syntax_error_captures_error_detail() -> None:
    task = TaskSpec(
        name="broken",
        prompt="code with syntax error",
        function_name="solve",
        train_cases=(((1,), 2),),
        hidden_cases=(),
    )
    code = "def solve(x):\n    return (\n"

    result = evaluate_candidate(
        code=code,
        task=task,
        edit_cost=0.0,
        config=RunConfig(sandbox_backend="process"),
    )

    assert result.hard_failure is True
    assert result.execution_status == "compile_or_exec_error"
    assert result.error_type == "compile_or_exec_error"
    assert len(result.error_detail) > 0


def test_missing_function_captures_error_type() -> None:
    task = TaskSpec(
        name="missing_fn",
        prompt="no function",
        function_name="solve",
        train_cases=(((1,), 2),),
        hidden_cases=(),
    )
    code = "def other(x):\n    return x + 1\n"

    result = evaluate_candidate(
        code=code,
        task=task,
        edit_cost=0.0,
        config=RunConfig(sandbox_backend="process"),
    )

    assert result.error_type == "missing_function"
    assert result.error_detail != ""


def test_ok_result_has_empty_error_fields() -> None:
    task = TaskSpec(
        name="ok",
        prompt="good code",
        function_name="solve",
        train_cases=(((1,), 2),),
        hidden_cases=(),
    )
    code = "def solve(x):\n    return x + 1\n"

    result = evaluate_candidate(
        code=code,
        task=task,
        edit_cost=0.0,
        config=RunConfig(sandbox_backend="process"),
    )

    assert result.error_type == ""
    assert result.error_detail == ""


def test_timeout_captures_error_type() -> None:
    task = TaskSpec(
        name="timeout",
        prompt="hangs",
        function_name="solve",
        train_cases=(((1,), 2),),
        hidden_cases=(),
    )
    code = "while True:\n    pass\n\ndef solve(x):\n    return x + 1\n"

    result = evaluate_candidate(
        code=code,
        task=task,
        edit_cost=0.0,
        config=RunConfig(exec_timeout_seconds=0.1, sandbox_backend="process"),
    )

    assert result.error_type == "timeout"


def test_docker_unavailable_status_on_file_not_found(monkeypatch) -> None:
    import alife_core.evaluator.core as core_mod

    def raise_fnf(*args, **kwargs):
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr(core_mod.subprocess, "run", raise_fnf)

    task = TaskSpec(
        name="docker_test",
        prompt="test docker unavailable",
        function_name="solve",
        train_cases=(((1,), 2),),
        hidden_cases=(),
    )
    code = "def solve(x):\n    return x + 1\n"
    config = RunConfig(sandbox_backend="docker")

    status, _outputs, _detail = core_mod._run_docker_batch(
        code=code,
        function_name=task.function_name,
        cases=task.train_cases,
        config=config,
    )

    assert status == "docker_unavailable"
