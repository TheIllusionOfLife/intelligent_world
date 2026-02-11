import json
from pathlib import Path

from alife_core.logging.events import write_run_start
from alife_core.models import RunConfig


def test_run_start_contains_reproducibility_fields(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    config = RunConfig(seed=123)

    write_run_start(
        log_path=log_path,
        run_id="run-1",
        config=config,
        git_commit="abc123",
        config_hash="cfg-hash",
        docker_image_digest="sha256:deadbeef",
    )

    payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["type"] == "run_start"
    assert payload["seed"] == 123
    assert payload["git_commit"] == "abc123"
    assert payload["config_hash"] == "cfg-hash"
    assert payload["docker_image_digest"] == "sha256:deadbeef"
