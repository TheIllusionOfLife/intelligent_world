import base64
import pickle
import subprocess
from multiprocessing import get_context
from queue import Empty

from alife_core.evaluator.docker_pool import DockerPool, restricted_loads
from alife_core.models import Case, EvaluationResult, RunConfig, TaskSpec

_RUNNER_SCRIPT = r"""
import base64
import pickle
import sys


def main():
    raw = sys.stdin.buffer.read()
    payload = pickle.loads(base64.b64decode(raw))
    code = payload["code"]
    function_name = payload["function_name"]
    cases = payload["cases"]

    namespace = {}
    try:
        exec(compile(code, "<candidate>", "exec"), {}, namespace)
    except Exception as exc:
        sys.stdout.buffer.write(base64.b64encode(pickle.dumps(("compile_or_exec_error", str(exc)))))
        return

    function = namespace.get(function_name)
    if function is None or not callable(function):
        sys.stdout.buffer.write(base64.b64encode(pickle.dumps(("missing_function", None))))
        return

    outputs = []
    for args, _expected in cases:
        try:
            outputs.append(("ok", function(*args)))
        except Exception as exc:
            outputs.append(("runtime_error", str(exc)))

    sys.stdout.buffer.write(base64.b64encode(pickle.dumps(("ok", outputs))))


if __name__ == "__main__":
    main()
"""


def _process_worker(
    queue_obj,
    code: str,
    function_name: str,
    cases: tuple[Case, ...],
) -> None:
    namespace: dict[str, object] = {}
    try:
        exec(compile(code, "<candidate>", "exec"), {}, namespace)
    except Exception as exc:  # noqa: BLE001,S102
        queue_obj.put(("compile_or_exec_error", str(exc)))
        return

    function = namespace.get(function_name)
    if function is None or not callable(function):
        queue_obj.put(("missing_function", None))
        return

    outputs: list[tuple[str, object]] = []
    for args, _expected in cases:
        try:
            outputs.append(("ok", function(*args)))
        except Exception as exc:  # noqa: BLE001
            outputs.append(("runtime_error", str(exc)))

    queue_obj.put(("ok", outputs))


def _encode_payload(code: str, function_name: str, cases: tuple[Case, ...]) -> bytes:
    payload = {
        "code": code,
        "function_name": function_name,
        "cases": cases,
    }
    return base64.b64encode(pickle.dumps(payload))


def _run_process_batch(
    code: str,
    function_name: str,
    cases: tuple[Case, ...],
    timeout_seconds: float,
) -> tuple[str, list[tuple[str, object]] | None, str]:
    context = get_context("spawn")
    queue = context.Queue()

    process = context.Process(
        target=_process_worker,
        args=(queue, code, function_name, cases),
        daemon=True,
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return "timeout", None, "execution exceeded timeout"

    try:
        status, payload = queue.get_nowait()
        if status == "ok":
            return status, payload, ""
        detail = str(payload) if payload is not None else status
        return status, None, detail
    except Empty:
        return "compile_or_exec_error", None, "process exited without result"


def _run_docker_batch(
    code: str,
    function_name: str,
    cases: tuple[Case, ...],
    config: RunConfig,
) -> tuple[str, list[tuple[str, object]] | None, str]:
    payload = _encode_payload(code=code, function_name=function_name, cases=cases)
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "64",
        "--memory",
        "256m",
        "--cpus",
        "0.5",
        "-i",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        config.docker_image,
        "python",
        "-c",
        _RUNNER_SCRIPT,
    ]

    try:
        completed = subprocess.run(
            command,
            input=payload,
            capture_output=True,
            check=False,
            timeout=config.exec_timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return "timeout", None, "docker execution exceeded timeout"
    except FileNotFoundError:
        return "docker_unavailable", None, "docker binary not found"

    if completed.returncode != 0:
        raw = completed.stderr or b""
        stderr_text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        lower = stderr_text.lower()
        if (
            "cannot connect" in lower
            or "connection refused" in lower
            or "is the docker daemon running" in lower
        ):
            return "docker_unavailable", None, stderr_text
        return "compile_or_exec_error", None, stderr_text

    try:
        status, result_payload = restricted_loads(base64.b64decode(completed.stdout))
    except Exception:  # noqa: BLE001
        return "compile_or_exec_error", None, "failed to decode runner output"

    if status == "ok":
        return status, result_payload, ""
    detail = str(result_payload) if result_payload is not None else status
    return status, None, detail


def _execute_case_batch(
    code: str,
    function_name: str,
    cases: tuple[Case, ...],
    config: RunConfig,
    pool: DockerPool | None = None,
) -> tuple[str, list[tuple[str, object]] | None, str]:
    if config.sandbox_backend == "process":
        return _run_process_batch(
            code=code,
            function_name=function_name,
            cases=cases,
            timeout_seconds=config.exec_timeout_seconds,
        )

    if config.use_persistent_docker and pool is not None:
        return pool.execute(
            code=code,
            function_name=function_name,
            cases=cases,
        )

    return _run_docker_batch(
        code=code,
        function_name=function_name,
        cases=cases,
        config=config,
    )


def _score_cases(
    cases: tuple[Case, ...],
    status: str,
    outputs: list[tuple[str, object]] | None,
) -> tuple[int, int, bool]:
    total = len(cases)
    if status != "ok" or outputs is None:
        return total, total, True

    if len(outputs) != total:
        return total, total, True

    failures = 0
    for (_args, expected), (item_status, payload) in zip(cases, outputs, strict=True):
        if item_status != "ok" or payload != expected:
            failures += 1

    return failures, total, False


def evaluate_candidate(
    code: str,
    task: TaskSpec,
    edit_cost: float,
    config: RunConfig,
    pool: DockerPool | None = None,
) -> EvaluationResult:
    combined_cases = task.train_cases + task.hidden_cases
    status, outputs, error_detail = _execute_case_batch(
        code=code,
        function_name=task.function_name,
        cases=combined_cases,
        config=config,
        pool=pool,
    )

    train_count = len(task.train_cases)
    if status == "ok" and outputs is not None:
        train_outputs = outputs[:train_count]
        hidden_outputs = outputs[train_count:]
    else:
        train_outputs = None
        hidden_outputs = None

    train_failures, train_total, train_hard_failure = _score_cases(
        task.train_cases,
        status,
        train_outputs,
    )
    hidden_failures, hidden_total, hidden_hard_failure = _score_cases(
        task.hidden_cases,
        status,
        hidden_outputs,
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

    if hard_failure:
        execution_status = status if status != "ok" else "internal_error"
    else:
        execution_status = "ok"

    return EvaluationResult(
        train_pass_ratio=train_pass_ratio,
        hidden_pass_ratio=hidden_pass_ratio,
        ast_edit_cost=edit_cost,
        fitness=fitness,
        train_failures=train_failures,
        hidden_failures=hidden_failures,
        hard_failure=hard_failure,
        execution_status=execution_status,
        error_type=execution_status if hard_failure else "",
        error_detail=error_detail if hard_failure else "",
    )
