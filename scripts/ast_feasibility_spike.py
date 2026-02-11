#!/usr/bin/env python3
import ast
import random

BASE = "def solve(x):\n    return x + 0\n"


def _fitness(source: str) -> float:
    namespace: dict[str, object] = {}
    try:
        exec(compile(source, "<candidate>", "exec"), {}, namespace)  # noqa: S102
        function = namespace["solve"]
        if not callable(function):
            return 0.0
    except (SyntaxError, KeyError, TypeError):
        return 0.0
    except Exception:  # noqa: BLE001
        return 0.0
    passing = 0
    for value in (1, 2, 3, 10):
        if function(value) == value + 1:
            passing += 1
    return passing / 4


def mutate_constant(source: str, rng: random.Random) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            node.value = rng.randint(-5, 5)
            break
    return ast.unparse(tree) + "\n"


def run_spike(samples: int = 100, seed: int = 0) -> dict[str, float]:
    if samples <= 0:
        return {
            "syntactic_validity_rate": 0.0,
            "semantic_difference_proxy_rate": 0.0,
            "fitness_improvement_rate": 0.0,
        }

    rng = random.Random(seed)
    valid = 0
    changed = 0
    improved = 0
    baseline = _fitness(BASE)

    for _ in range(samples):
        candidate = mutate_constant(BASE, rng)
        try:
            ast.parse(candidate)
            valid += 1
        except SyntaxError:
            continue
        if candidate != BASE:
            changed += 1
        if _fitness(candidate) > baseline:
            improved += 1

    return {
        "syntactic_validity_rate": valid / samples,
        "semantic_difference_proxy_rate": changed / samples,
        "fitness_improvement_rate": improved / samples,
    }


if __name__ == "__main__":
    print(run_spike())
