import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(path: str):
    script_path = REPO_ROOT / path
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
    assert result["recommended_strategy"] == "unknown"


def test_ast_feasibility_handles_zero_samples() -> None:
    module = _load_script("scripts/ast_feasibility_spike.py")

    result = module.run_spike(samples=0)

    assert result["syntactic_validity_rate"] == 0.0
    assert result["semantic_difference_proxy_rate"] == 0.0
    assert result["fitness_improvement_rate"] == 0.0


def test_ast_feasibility_reports_non_zero_improvement_rate_for_default_seed() -> None:
    module = _load_script("scripts/ast_feasibility_spike.py")

    result = module.run_spike(samples=100, seed=0)

    assert result["fitness_improvement_rate"] > 0.0


def test_schedule_curve_returns_series() -> None:
    module = _load_script("scripts/schedule_curve_spike.py")

    result = module.generate_schedule_data(steps=5, initial_temperature=1.0, cooling_rate=0.99)

    assert len(result["temperature"]) == 6
    assert len(result["w2_effective"]) == 6
    assert result["w2_leads_temperature"] is False


def test_schedule_curve_handles_negative_steps() -> None:
    module = _load_script("scripts/schedule_curve_spike.py")

    result = module.generate_schedule_data(steps=-1)

    assert result["temperature"] == []
    assert result["w2_effective"] == []
    assert result["w2_leads_temperature"] is False
