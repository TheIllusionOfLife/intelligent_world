import importlib.util
from pathlib import Path


def test_docker_latency_handles_zero_iterations() -> None:
    script_path = Path("scripts/docker_latency_spike.py")
    spec = importlib.util.spec_from_file_location("docker_latency_spike", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    result = module.benchmark(iterations=0)

    assert result["mean_ms"] == 0.0
    assert result["p95_ms"] == 0.0
