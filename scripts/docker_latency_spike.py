#!/usr/bin/env python3
import statistics
import subprocess
import time


def _strategy_for_latency(mean_ms: float) -> str:
    if mean_ms < 200:
        return "docker_run_per_eval"
    if mean_ms <= 1000:
        return "container_reuse"
    return "process_fallback"


def benchmark(iterations: int = 100) -> dict[str, float | str]:
    samples: list[float] = []
    failed_runs = 0
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                ["docker", "run", "--rm", "python:3.12-slim", "python", "-c", "print('ok')"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
            )
            if completed.returncode != 0:
                failed_runs += 1
                continue
        except subprocess.TimeoutExpired:
            samples.append(60000.0)
            continue
        except FileNotFoundError:
            failed_runs += 1
            continue
        samples.append((time.perf_counter() - start) * 1000.0)

    if failed_runs > 0 and not samples:
        return {
            "mean_ms": 0.0,
            "p95_ms": 0.0,
            "recommended_strategy": "docker_unavailable",
            "docker_available": False,
        }

    if not samples:
        return {
            "mean_ms": 0.0,
            "p95_ms": 0.0,
            "recommended_strategy": "unknown",
            "docker_available": True,
        }

    mean = statistics.fmean(samples)
    p95 = sorted(samples)[int(0.95 * (len(samples) - 1))]
    return {
        "mean_ms": mean,
        "p95_ms": p95,
        "recommended_strategy": _strategy_for_latency(mean),
        "docker_available": failed_runs == 0,
    }


if __name__ == "__main__":
    print(benchmark())
