import ast
import gzip
import logging
import math
import statistics
import tokenize
from collections import Counter
from dataclasses import dataclass
from io import BytesIO

from alife_core.models import OrganismState

LOGGER = logging.getLogger(__name__)
_NOVELTY_PAIRWISE_THRESHOLD = 64
_NOVELTY_REFERENCE_SAMPLE_SIZE = 32


def ast_shape_fingerprint(source: str) -> str:
    tree = ast.parse(source)

    class _ShapeNormalizer(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            return ast.copy_location(ast.Constant(value=None), node)

        def visit_Name(self, node: ast.Name) -> ast.AST:
            updated = ast.Name(id="_", ctx=node.ctx)
            return ast.copy_location(updated, node)

        def visit_arg(self, node: ast.arg) -> ast.AST:
            updated = ast.arg(arg="_", annotation=None, type_comment=None)
            return ast.copy_location(updated, node)

    normalized = _ShapeNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, include_attributes=False)


def ast_node_count(source: str) -> int:
    return sum(1 for _ in ast.walk(ast.parse(source)))


def ast_max_depth(source: str) -> int:
    tree = ast.parse(source)

    def _depth(node: ast.AST) -> int:
        child_nodes = list(ast.iter_child_nodes(node))
        if not child_nodes:
            return 1
        return 1 + max(_depth(child) for child in child_nodes)

    return _depth(tree)


def _normalized_levenshtein(left: str, right: str) -> float:
    if left == right:
        return 0.0
    if not left or not right:
        return 1.0

    prev = list(range(len(right) + 1))
    for row, char_left in enumerate(left, start=1):
        current = [row]
        for col, char_right in enumerate(right, start=1):
            insert_cost = current[col - 1] + 1
            delete_cost = prev[col] + 1
            replace_cost = prev[col - 1] + (0 if char_left == char_right else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        prev = current
    return prev[-1] / max(len(left), len(right))


def kolmogorov_complexity_proxy(codes: list[str]) -> float:
    """Mean gzip compression ratio across code strings (proxy for Kolmogorov complexity)."""
    if not codes:
        return 0.0
    ratios: list[float] = []
    for code in codes:
        raw = code.encode("utf-8")
        if not raw:
            continue
        compressed = gzip.compress(raw)
        ratios.append(len(compressed) / len(raw))
    return statistics.fmean(ratios) if ratios else 0.0


def cumulative_complexity_delta(
    current_mean_depth: float,
    initial_mean_depth: float,
) -> float:
    """Difference between current and initial mean AST depth."""
    return current_mean_depth - initial_mean_depth


def specialization_score(pass_matrices: list[list[bool]]) -> float:
    """Measure how specialized organisms are on different test case subsets.

    0.0 = all identical pass patterns, 1.0 = fully specialized (no overlap).
    Uses mean pairwise Jaccard distance across organisms.
    """
    if len(pass_matrices) < 2:
        return 0.0
    n_cases = len(pass_matrices[0]) if pass_matrices[0] else 0
    if n_cases == 0:
        return 0.0

    distances: list[float] = []
    for i in range(len(pass_matrices)):
        for j in range(i + 1, len(pass_matrices)):
            set_a = {k for k, v in enumerate(pass_matrices[i]) if v}
            set_b = {k for k, v in enumerate(pass_matrices[j]) if v}
            union = set_a | set_b
            if not union:
                distances.append(0.0)
                continue
            intersection = set_a & set_b
            distances.append(1.0 - len(intersection) / len(union))
    return statistics.fmean(distances) if distances else 0.0


def code_token_zipf_coefficient(code: str) -> float:
    """Fit power-law exponent to token frequency distribution.

    Life-like systems tend to follow Zipf's law (exponent near 1.0).
    Uses least-squares fit on log-log data.
    """
    if not code.strip():
        return 0.0
    try:
        tokens = list(tokenize.tokenize(BytesIO(code.encode("utf-8")).readline))
    except tokenize.TokenError:
        return 0.0

    token_strings = [
        tok.string
        for tok in tokens
        if tok.type not in (tokenize.ENCODING, tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER)
        and tok.string.strip()
    ]
    if len(token_strings) < 2:
        return 0.0

    freq_counts = Counter(token_strings)
    frequencies = sorted(freq_counts.values(), reverse=True)
    if len(frequencies) < 2:
        return 0.0

    # Least-squares fit of log(freq) = -alpha * log(rank) + c
    log_ranks = [math.log(rank) for rank in range(1, len(frequencies) + 1)]
    log_freqs = [math.log(freq) for freq in frequencies]
    n = len(log_ranks)
    sum_x = sum(log_ranks)
    sum_y = sum(log_freqs)
    sum_xy = sum(x * y for x, y in zip(log_ranks, log_freqs, strict=True))
    sum_xx = sum(x * x for x in log_ranks)
    denominator = n * sum_xx - sum_x * sum_x
    if abs(denominator) < 1e-12:
        return 0.0
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return abs(slope)


@dataclass(frozen=True)
class EvolutionMetrics:
    structural_diversity_ratio: float
    shannon_entropy: float
    simpson_diversity_index: float
    cluster_count: int
    mean_ast_nodes: float
    median_ast_nodes: float
    mean_ast_depth: float
    median_ast_depth: float
    mean_novelty: float
    max_lineage_depth: int
    mean_lineage_depth: float
    kolmogorov_complexity_proxy: float = 0.0
    cumulative_complexity_delta: float = 0.0
    code_token_zipf_coefficient: float = 0.0


def compute_generation_metrics(
    population: list[OrganismState],
    novelty_k: int = 3,
) -> EvolutionMetrics:
    if not population:
        return EvolutionMetrics(
            structural_diversity_ratio=0.0,
            shannon_entropy=0.0,
            simpson_diversity_index=0.0,
            cluster_count=0,
            mean_ast_nodes=0.0,
            median_ast_nodes=0.0,
            mean_ast_depth=0.0,
            median_ast_depth=0.0,
            mean_novelty=0.0,
            max_lineage_depth=0,
            mean_lineage_depth=0.0,
            kolmogorov_complexity_proxy=0.0,
            cumulative_complexity_delta=0.0,
            code_token_zipf_coefficient=0.0,
        )

    fingerprints: list[str] = []
    ast_nodes: list[int] = []
    ast_depths: list[int] = []
    for organism in population:
        fingerprint = (
            organism.shape_fingerprint
            if organism.shape_fingerprint
            else ast_shape_fingerprint(organism.code)
        )
        fingerprints.append(fingerprint)
        ast_nodes.append(
            organism.ast_nodes if organism.ast_nodes > 0 else ast_node_count(organism.code)
        )
        ast_depths.append(
            organism.ast_depth if organism.ast_depth > 0 else ast_max_depth(organism.code)
        )

    counts: dict[str, int] = {}
    for fingerprint in fingerprints:
        counts[fingerprint] = counts.get(fingerprint, 0) + 1
    total = len(fingerprints)
    unique = len(counts)

    probabilities = [count / total for count in counts.values()]
    raw_entropy = -sum(prob * math.log2(prob) for prob in probabilities if prob > 0.0)
    max_entropy = math.log2(total) if total > 1 else 1.0
    normalized_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
    simpson = 1.0 - sum(prob * prob for prob in probabilities)

    k = max(1, novelty_k)
    use_approximate_novelty = total > _NOVELTY_PAIRWISE_THRESHOLD
    if use_approximate_novelty:
        LOGGER.warning(
            "Approximating novelty for population size %s (threshold=%s)",
            total,
            _NOVELTY_PAIRWISE_THRESHOLD,
        )
        reference_fingerprints = fingerprints[:_NOVELTY_REFERENCE_SAMPLE_SIZE]
    else:
        reference_fingerprints = fingerprints

    novelty_scores: list[float] = []
    for idx, fingerprint in enumerate(fingerprints):
        if use_approximate_novelty:
            distances = [
                _normalized_levenshtein(fingerprint, other)
                for jdx, other in enumerate(reference_fingerprints)
                if jdx != idx
            ]
        else:
            distances = [
                _normalized_levenshtein(fingerprint, other)
                for jdx, other in enumerate(fingerprints)
                if jdx != idx
            ]
        if not distances:
            novelty_scores.append(0.0)
            continue
        distances.sort()
        nearest = distances[:k]
        novelty_scores.append(statistics.fmean(nearest))

    lineage_depths = [organism.lineage_depth for organism in population]
    codes = [organism.code for organism in population]

    kc_proxy = kolmogorov_complexity_proxy(codes)
    mean_zipf = (
        statistics.fmean([code_token_zipf_coefficient(code) for code in codes]) if codes else 0.0
    )

    return EvolutionMetrics(
        structural_diversity_ratio=unique / total,
        shannon_entropy=normalized_entropy,
        simpson_diversity_index=simpson,
        cluster_count=unique,
        mean_ast_nodes=statistics.fmean(ast_nodes),
        median_ast_nodes=float(statistics.median(ast_nodes)),
        mean_ast_depth=statistics.fmean(ast_depths),
        median_ast_depth=float(statistics.median(ast_depths)),
        mean_novelty=statistics.fmean(novelty_scores),
        max_lineage_depth=max(lineage_depths) if lineage_depths else 0,
        mean_lineage_depth=statistics.fmean(lineage_depths) if lineage_depths else 0.0,
        kolmogorov_complexity_proxy=kc_proxy,
        code_token_zipf_coefficient=mean_zipf,
    )
