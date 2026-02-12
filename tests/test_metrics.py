from alife_core.metrics.evolution import (
    EvolutionMetrics,
    code_token_zipf_coefficient,
    compute_generation_metrics,
    cumulative_complexity_delta,
    kolmogorov_complexity_proxy,
    specialization_score,
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


# --- Kolmogorov complexity proxy tests ---


def test_kolmogorov_complexity_proxy_returns_positive_ratio() -> None:
    codes = [
        "def solve(x):\n    return x + 1\n",
        "def solve(x):\n    return x\n",
    ]
    result = kolmogorov_complexity_proxy(codes)
    assert result > 0.0


def test_kolmogorov_complexity_proxy_lower_for_repetitive_code() -> None:
    # Highly repetitive code compresses better (lower ratio)
    repetitive = ["x = 1\n" * 200]
    unique = ["import math\ndef solve(x):\n    return math.sqrt(x) + x**2 - 3*x + 7\n"]
    rep_ratio = kolmogorov_complexity_proxy(repetitive)
    unique_ratio = kolmogorov_complexity_proxy(unique)
    assert rep_ratio < unique_ratio


def test_kolmogorov_complexity_proxy_empty_list() -> None:
    assert kolmogorov_complexity_proxy([]) == 0.0


# --- Cumulative complexity delta tests ---


def test_cumulative_complexity_delta_zero_for_same_depth() -> None:
    assert cumulative_complexity_delta(current_mean_depth=5.0, initial_mean_depth=5.0) == 0.0


def test_cumulative_complexity_delta_positive_for_growth() -> None:
    assert cumulative_complexity_delta(current_mean_depth=8.0, initial_mean_depth=5.0) == 3.0


def test_cumulative_complexity_delta_negative_for_simplification() -> None:
    assert cumulative_complexity_delta(current_mean_depth=3.0, initial_mean_depth=5.0) == -2.0


# --- Test specialization score tests ---


def test_specialization_score_zero_for_identical_results() -> None:
    # All organisms pass the same cases
    pass_matrices = [
        [True, True, False],
        [True, True, False],
        [True, True, False],
    ]
    score = specialization_score(pass_matrices)
    assert score == 0.0


def test_specialization_score_one_for_fully_specialized() -> None:
    # Each organism passes a completely different set
    pass_matrices = [
        [True, False, False],
        [False, True, False],
        [False, False, True],
    ]
    score = specialization_score(pass_matrices)
    assert score > 0.5


def test_specialization_score_between_zero_and_one() -> None:
    pass_matrices = [
        [True, True, False, False],
        [True, False, True, False],
        [False, True, False, True],
    ]
    score = specialization_score(pass_matrices)
    assert 0.0 < score < 1.0


def test_specialization_score_empty_input() -> None:
    assert specialization_score([]) == 0.0


def test_specialization_score_single_organism() -> None:
    assert specialization_score([[True, False]]) == 0.0


# --- Code token Zipf coefficient tests ---


def test_zipf_coefficient_returns_positive_exponent() -> None:
    code = (
        "def solve(x):\n"
        "    total = 0\n"
        "    for i in range(x):\n"
        "        total = total + i\n"
        "    return total\n"
    )
    coeff = code_token_zipf_coefficient(code)
    assert coeff > 0.0


def test_zipf_coefficient_empty_code() -> None:
    assert code_token_zipf_coefficient("") == 0.0


def test_zipf_coefficient_deterministic() -> None:
    code = "def solve(x):\n    return x + 1\n"
    assert code_token_zipf_coefficient(code) == code_token_zipf_coefficient(code)


# --- New metrics appear in EvolutionMetrics ---


def test_new_metrics_present_in_evolution_metrics() -> None:
    population = [
        _organism("o1", "def solve(x):\n    return x + 1\n", 0.1),
        _organism("o2", "def solve(x):\n    return x + 2\n", 0.2),
        _organism("o3", "def solve(x):\n    return x - 1\n", 0.3),
    ]
    metrics = compute_generation_metrics(population, novelty_k=2)
    assert hasattr(metrics, "kolmogorov_complexity_proxy")
    assert hasattr(metrics, "cumulative_complexity_delta")
    assert hasattr(metrics, "code_token_zipf_coefficient")
    assert isinstance(metrics.kolmogorov_complexity_proxy, float)
    assert isinstance(metrics.code_token_zipf_coefficient, float)
