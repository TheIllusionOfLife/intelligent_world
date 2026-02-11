from collections.abc import Iterable
from multiprocessing import get_context
from queue import Empty

from alife_core.models import EvaluationResult, RunConfig, TaskSpec


def _worker(code: str, function_name: str, args: tuple, queue) -> None:
    namespace: dict[str, object] = {}
    try:
        exec(compile(code, "<candidate>", "exec"), {}, namespace)
    except Exception as exc:  # noqa: BLE001,S102
        queue.put(("compile_or_exec_error", str(exc)))
        return

    function = namespace.get(function_name)
    if function is None or not callable(function):
        queue.put(("missing_function", "missing callable function"))
        return

    try:
        queue.put(("ok", function(*args)))
    except Exception as exc:  # noqa: BLE001
        queue.put(("runtime_error", str(exc)))


def _run_case_with_timeout(
    code: str,
    function_name: str,
    args: tuple,
    timeout_seconds: float,
) -> tuple[str, object | None]:
    context = get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_worker, args=(code, function_name, args, queue), daemon=True)
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return "timeout", None

    try:
        return queue.get_nowait()
    except Empty:
        return "compile_or_exec_error", None


def _run_cases(
    code: str,
    task: TaskSpec,
    cases: Iterable[tuple[tuple, object]],
    timeout_seconds: float,
) -> tuple[int, int, bool]:
    failures = 0
    total = 0

    case_list = list(cases)
    for args, expected in case_list:
        total += 1
        status, payload = _run_case_with_timeout(
            code=code,
            function_name=task.function_name,
            args=args,
            timeout_seconds=timeout_seconds,
        )

        if status in {"missing_function", "compile_or_exec_error", "timeout"}:
            return len(case_list), len(case_list), True

        if status != "ok" or payload != expected:
            failures += 1

    return failures, total, False


def evaluate_candidate(
    code: str,
    task: TaskSpec,
    edit_cost: float,
    config: RunConfig,
) -> EvaluationResult:
    train_failures, train_total, train_hard_failure = _run_cases(
        code=code,
        task=task,
        cases=task.train_cases,
        timeout_seconds=config.exec_timeout_seconds,
    )
    hidden_failures, hidden_total, hidden_hard_failure = _run_cases(
        code=code,
        task=task,
        cases=task.hidden_cases,
        timeout_seconds=config.exec_timeout_seconds,
    )

    train_pass_ratio = 0.0 if train_total == 0 else (train_total - train_failures) / train_total
    hidden_pass_ratio = (
        0.0 if hidden_total == 0 else (hidden_total - hidden_failures) / hidden_total
    )

    hard_failure = train_hard_failure or hidden_hard_failure
    if hard_failure:
        fitness = 0.0
    else:
        fitness = (config.w1_pass_ratio * train_pass_ratio) - (config.w2_ast_edit_cost * edit_cost)

    return EvaluationResult(
        train_pass_ratio=train_pass_ratio,
        hidden_pass_ratio=hidden_pass_ratio,
        ast_edit_cost=edit_cost,
        fitness=fitness,
        train_failures=train_failures,
        hidden_failures=hidden_failures,
    )
