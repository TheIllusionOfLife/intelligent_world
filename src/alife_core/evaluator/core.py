from collections.abc import Iterable

from alife_core.models import EvaluationResult, RunConfig, TaskSpec


def _run_cases(function, cases: Iterable[tuple[tuple, object]]) -> tuple[int, int]:
    failures = 0
    total = 0
    for args, expected in cases:
        total += 1
        try:
            actual = function(*args)
        except Exception:
            failures += 1
            continue
        if actual != expected:
            failures += 1
    return failures, total


def evaluate_candidate(
    code: str,
    task: TaskSpec,
    edit_cost: float,
    config: RunConfig,
) -> EvaluationResult:
    namespace: dict[str, object] = {}
    exec(compile(code, "<candidate>", "exec"), {}, namespace)
    function = namespace[task.function_name]

    train_failures, train_total = _run_cases(function, task.train_cases)
    hidden_failures, hidden_total = _run_cases(function, task.hidden_cases)

    train_pass_ratio = 0.0 if train_total == 0 else (train_total - train_failures) / train_total
    hidden_pass_ratio = (
        0.0 if hidden_total == 0 else (hidden_total - hidden_failures) / hidden_total
    )

    fitness = (config.w1_pass_ratio * train_pass_ratio) - (config.w2_ast_edit_cost * edit_cost)

    return EvaluationResult(
        train_pass_ratio=train_pass_ratio,
        hidden_pass_ratio=hidden_pass_ratio,
        ast_edit_cost=edit_cost,
        fitness=fitness,
        train_failures=train_failures,
        hidden_failures=hidden_failures,
    )
