"""Persistent Docker container pool for faster evaluation via docker exec."""

import base64
import logging
import pickle
import subprocess
import threading
from typing import Self

from alife_core.models import Case, RunConfig

LOGGER = logging.getLogger(__name__)

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


def _encode_payload(code: str, function_name: str, cases: tuple[Case, ...]) -> bytes:
    payload = {
        "code": code,
        "function_name": function_name,
        "cases": cases,
    }
    return base64.b64encode(pickle.dumps(payload))


class DockerPool:
    """Manages long-lived Docker containers for evaluation via docker exec."""

    def __init__(self, config: RunConfig, pool_size: int = 1) -> None:
        self._config = config
        self._pool_size = max(1, pool_size)
        self._containers: list[str] = []
        self._lock = threading.Lock()
        self._round_robin = 0

    def start(self) -> None:
        for _ in range(self._pool_size):
            container_id = self._create_container()
            self._containers.append(container_id)
        LOGGER.info("DockerPool started with %d containers", len(self._containers))

    def stop(self) -> None:
        for container_id in self._containers:
            self._remove_container(container_id)
        self._containers.clear()
        LOGGER.info("DockerPool stopped")

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()

    def execute(
        self,
        code: str,
        function_name: str,
        cases: tuple[Case, ...],
    ) -> tuple[str, list[tuple[str, object]] | None]:
        with self._lock:
            idx = self._round_robin % len(self._containers)
            self._round_robin += 1
            container_id = self._containers[idx]

        status, result = self._exec_in_container(container_id, code, function_name, cases)

        if status not in ("ok", "timeout", "compile_or_exec_error", "missing_function"):
            LOGGER.warning("Container %s may be unhealthy, attempting recovery", container_id[:12])
            new_id = self._recover_container(idx, container_id)
            if new_id:
                status, result = self._exec_in_container(new_id, code, function_name, cases)

        return status, result

    def _create_container(self) -> str:
        command = [
            "docker",
            "create",
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
            self._config.docker_image,
            "python",
            "-c",
            _RUNNER_SCRIPT,
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Failed to create container: {completed.stderr.decode('utf-8', errors='replace')}"
            )
        container_id = completed.stdout.decode("utf-8").strip()
        # Start the container
        subprocess.run(
            ["docker", "start", container_id],
            capture_output=True,
            check=False,
            timeout=10,
        )
        return container_id

    def _remove_container(self, container_id: str) -> None:
        subprocess.run(
            ["docker", "rm", "-f", container_id],
            capture_output=True,
            check=False,
            timeout=10,
        )

    def _exec_in_container(
        self,
        container_id: str,
        code: str,
        function_name: str,
        cases: tuple[Case, ...],
    ) -> tuple[str, list[tuple[str, object]] | None]:
        payload = _encode_payload(code=code, function_name=function_name, cases=cases)
        command = [
            "docker",
            "exec",
            "-i",
            container_id,
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
                timeout=self._config.exec_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return "timeout", None

        if completed.returncode != 0:
            return "container_error", None

        try:
            status, result_payload = pickle.loads(base64.b64decode(completed.stdout))
        except Exception:  # noqa: BLE001
            return "compile_or_exec_error", None

        if status == "ok":
            return status, result_payload
        return status, None

    def _is_container_healthy(self, container_id: str) -> bool:
        completed = subprocess.run(
            ["docker", "inspect", "--format={{.State.Running}}", container_id],
            capture_output=True,
            check=False,
            timeout=5,
        )
        return completed.returncode == 0 and b"true" in completed.stdout.lower()

    def _recover_container(self, idx: int, old_id: str) -> str | None:
        with self._lock:
            if not self._is_container_healthy(old_id):
                self._remove_container(old_id)
                try:
                    new_id = self._create_container()
                except RuntimeError:
                    LOGGER.error("Failed to recover container at index %d", idx)
                    return None
                self._containers[idx] = new_id
                LOGGER.info(
                    "Recovered container at index %d: %s -> %s",
                    idx,
                    old_id[:12],
                    new_id[:12],
                )
                return new_id
        return old_id
