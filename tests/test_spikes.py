import importlib.util
from pathlib import Path


def _load_script(path: str):
    script_path = Path(path)
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_docker_latency_handles_zero_iterations() -> None:
    module = _load_script("scripts/docker_latency_spike.py")

    result = module.benchmark(iterations=0)

    assert result["mean_ms"] == 0.0
    assert result["p95_ms"] == 0.0


def test_ast_feasibility_handles_zero_samples() -> None:
    module = _load_script("scripts/ast_feasibility_spike.py")

    result = module.run_spike(samples=0)

    assert result["syntactic_validity_rate"] == 0.0
    assert result["semantic_difference_proxy_rate"] == 0.0
