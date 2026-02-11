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
    assert payload["schema_version"] == 2
    assert payload["event_type"] == "run.started"
    assert payload["run_id"] == "run-1"
    assert payload["payload"]["random_seed"] == 123
    assert payload["payload"]["framework_git_sha"] == "abc123"
    assert payload["payload"]["docker_image_digest"] == "sha256:deadbeef"
    assert payload["payload"]["python_version"] == "3.12.1"
    assert payload["payload"]["cpu_architecture"] == "arm64"
    assert payload["payload"]["parameters"] == {"seed": 123}


def test_write_event_appends_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "run.jsonl"

    write_event(
        log_path,
        {
            "schema_version": 2,
            "event_type": "step.evaluated",
            "run_id": "run-1",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "mode": "single_agent",
            "task": "two_sum_sorted",
            "step": 1,
            "payload": {"fitness": 0.5},
        },
    )
    write_event(
        log_path,
        {
            "schema_version": 2,
            "event_type": "step.evaluated",
            "run_id": "run-1",
            "timestamp": "2026-01-01T00:00:01+00:00",
            "mode": "single_agent",
            "task": "two_sum_sorted",
            "step": 2,
            "payload": {"fitness": 0.6},
        },
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["event_type"] == "step.evaluated"
    assert first["step"] == 1
    assert second["event_type"] == "step.evaluated"
    assert second["step"] == 2
