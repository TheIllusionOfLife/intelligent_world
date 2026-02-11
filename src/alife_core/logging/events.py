import json
from datetime import UTC, datetime
from pathlib import Path

from alife_core.models import RunConfig


def write_run_start(
    log_path: Path,
    run_id: str,
    config: RunConfig,
    git_commit: str,
    config_hash: str,
    docker_image_digest: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "run_start",
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "seed": config.seed,
        "git_commit": git_commit,
        "config_hash": config_hash,
        "docker_image_digest": docker_image_digest,
    }
    with log_path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(payload, sort_keys=True) + "\n")
