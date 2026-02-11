import json
from datetime import UTC, datetime
from pathlib import Path

from alife_core.models import RunConfig


def write_event(log_path: Path, payload: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def write_run_start(
    log_path: Path,
    run_id: str,
    config: RunConfig,
    framework_git_sha: str,
    docker_image_digest: str,
    python_version: str,
    cpu_architecture: str,
    parameters: dict[str, object],
) -> None:
    payload = {
        "type": "run_start",
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "random_seed": config.seed,
        "framework_git_sha": framework_git_sha,
        "docker_image_digest": docker_image_digest,
        "python_version": python_version,
        "cpu_architecture": cpu_architecture,
        "parameters": parameters,
    }
    write_event(log_path, payload)
