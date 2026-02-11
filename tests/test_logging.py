import json
from pathlib import Path

from alife_core.logging.events import write_event, write_run_start
from alife_core.models import RunConfig


def test_run_start_contains_reproducibility_fields(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"
    config = RunConfig(seed=123)

    write_run_start(
        log_path=log_path,
        run_id="run-1",
        config=config,
        framework_git_sha="abc123",
        docker_image_digest="sha256:deadbeef",
        python_version="3.12.1",
        cpu_architecture="arm64",
        parameters={"seed": 123},
    )

    payload = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["type"] == "run_start"
    assert payload["random_seed"] == 123
    assert payload["framework_git_sha"] == "abc123"
    assert payload["docker_image_digest"] == "sha256:deadbeef"
    assert payload["python_version"] == "3.12.1"
    assert payload["cpu_architecture"] == "arm64"
    assert payload["parameters"] == {"seed": 123}


def test_write_event_appends_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"

    write_event(log_path, {"type": "step", "step": 1, "fitness": 0.5})
    write_event(log_path, {"type": "step", "step": 2, "fitness": 0.6})

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 1
    assert json.loads(lines[1])["step"] == 2
