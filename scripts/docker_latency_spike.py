#!/usr/bin/env python3
import statistics
import subprocess
import time


def benchmark(iterations: int = 100) -> dict[str, float]:
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            subprocess.run(
                ["docker", "run", "--rm", "python:3.12-slim", "python", "-c", "print('ok')"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            samples.append(60000.0)
            continue
        samples.append((time.perf_counter() - start) * 1000.0)

    if not samples:
        return {"mean_ms": 0.0, "p95_ms": 0.0}

    mean = statistics.fmean(samples)
    p95 = sorted(samples)[int(0.95 * (len(samples) - 1))]
    return {"mean_ms": mean, "p95_ms": p95}


if __name__ == "__main__":
    print(benchmark())
