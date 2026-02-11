from alife_core.metrics.evolution import (
    EvolutionMetrics,
    compute_generation_metrics,
)
from alife_core.models import OrganismState


def _organism(
    organism_id: str,
    code: str,
    fitness: float,
    parent_ids: tuple[str, ...] = (),
    birth_generation: int = 0,
    lineage_depth: int = 0,
) -> OrganismState:
    return OrganismState(
        organism_id=organism_id,
        parent_ids=parent_ids,
        birth_generation=birth_generation,
        code=code,
        fitness=fitness,
        train_pass_ratio=0.0,
        hidden_pass_ratio=0.0,
        ast_nodes=0,
        ast_depth=0,
        shape_fingerprint="",
        lineage_depth=lineage_depth,
        evaluated=False,
    )


def test_compute_generation_metrics_is_deterministic() -> None:
    population = [
        _organism("o1", "def solve(x):\n    return x + 1\n", 0.1),
        _organism("o2", "def solve(x):\n    return x + 2\n", 0.2),
        _organism("o3", "def solve(x):\n    return x - 1\n", 0.3),
    ]

    first = compute_generation_metrics(population, novelty_k=2)
    second = compute_generation_metrics(population, novelty_k=2)
    assert first == second


def test_compute_generation_metrics_entropy_bounds() -> None:
    population = [
        _organism("o1", "def solve(x):\n    return x + 1\n", 0.1),
        _organism("o2", "def solve(x):\n    return x + 1\n", 0.2),
        _organism("o3", "def solve(x):\n    return x + 1\n", 0.3),
    ]

    metrics = compute_generation_metrics(population, novelty_k=2)
    assert isinstance(metrics, EvolutionMetrics)
    assert 0.0 <= metrics.shannon_entropy <= 1.0
    assert metrics.cluster_count == 1


def test_compute_generation_metrics_increases_diversity_for_distinct_shapes() -> None:
    low_diversity = [
        _organism("a1", "def solve(x):\n    return x + 1\n", 0.1),
        _organism("a2", "def solve(x):\n    return x + 1\n", 0.2),
        _organism("a3", "def solve(x):\n    return x + 1\n", 0.3),
    ]
    high_diversity = [
        _organism("b1", "def solve(x):\n    return x + 1\n", 0.1),
        _organism("b2", "def solve(x):\n    return x if x > 0 else -x\n", 0.2),
        _organism(
            "b3",
            (
                "def solve(x):\n"
                "    total = 0\n"
                "    for item in range(x):\n"
                "        total += item\n"
                "    return total\n"
            ),
            0.3,
        ),
    ]

    low_metrics = compute_generation_metrics(low_diversity, novelty_k=2)
    high_metrics = compute_generation_metrics(high_diversity, novelty_k=2)

    assert high_metrics.structural_diversity_ratio > low_metrics.structural_diversity_ratio
    assert high_metrics.shannon_entropy > low_metrics.shannon_entropy
    assert high_metrics.mean_novelty >= low_metrics.mean_novelty


def test_compute_generation_metrics_tracks_lineage_depth() -> None:
    population = [
        _organism("p0", "def solve(x):\n    return x\n", 0.1, parent_ids=(), lineage_depth=0),
        _organism(
            "p1",
            "def solve(x):\n    return x + 1\n",
            0.2,
            parent_ids=("p0",),
            birth_generation=1,
            lineage_depth=1,
        ),
        _organism(
            "p2",
            "def solve(x):\n    return x + 2\n",
            0.3,
            parent_ids=("p1",),
            birth_generation=2,
            lineage_depth=2,
        ),
    ]

    metrics = compute_generation_metrics(population, novelty_k=2)
    assert metrics.max_lineage_depth == 2
    assert metrics.mean_lineage_depth > 0.0
