#!/usr/bin/env python3
import ast
import random

BASE = "def solve(x):\n    return x + 1\n"


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
        }

    rng = random.Random(seed)
    valid = 0
    changed = 0
    for _ in range(samples):
        candidate = mutate_constant(BASE, rng)
        try:
            ast.parse(candidate)
            valid += 1
        except SyntaxError:
            continue
        if candidate != BASE:
            changed += 1
    return {
        "syntactic_validity_rate": valid / samples,
        "semantic_difference_proxy_rate": changed / samples,
    }


if __name__ == "__main__":
    print(run_spike())
