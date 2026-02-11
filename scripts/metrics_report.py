#!/usr/bin/env python3
import json
from pathlib import Path


def summarize_metrics(log_path: Path | str) -> dict[str, object]:
    path = Path(log_path)
    generations = 0
    bad_lines = 0
    latest_metrics: dict[str, object] = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                bad_lines += 1
                continue
            if not isinstance(event, dict):
                bad_lines += 1
                continue
            if event.get("event_type") != "generation.metrics":
                continue
            generations += 1
            payload = event.get("payload", {})
            if isinstance(payload, dict):
                latest_metrics = payload

    if generations == 0:
        return {
            "schema_version": 2,
            "log_path": str(path),
            "generations": 0,
            "last_metrics": {},
            "bad_lines": bad_lines,
        }
    return {
        "schema_version": 2,
        "log_path": str(path),
        "generations": generations,
        "last_metrics": latest_metrics,
        "bad_lines": bad_lines,
    }


if __name__ == "__main__":
    payload = summarize_metrics(Path("logs/latest.jsonl"))
    print(json.dumps(payload, sort_keys=True))
