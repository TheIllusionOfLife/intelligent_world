"""Tests for LibCST-based semantic mutation operators."""

import ast
import random

from alife_core.mutation.semantic import (
    mutate_guard_insertion,
    mutate_loop_conversion,
    mutate_variable_extraction,
)


class TestGuardInsertion:
    def test_inserts_guard_into_function(self) -> None:
        source = "def solve(x, y):\n    total = x + y\n    return total\n"
        rng = random.Random(42)
        result = mutate_guard_insertion(source, rng)
        assert result != source
        # Must be syntactically valid
        ast.parse(result)
        # Should contain an if/return pattern
        assert "if" in result and "return" in result

    def test_returns_original_when_no_function(self) -> None:
        source = "x = 1\ny = 2\n"
        rng = random.Random(0)
        result = mutate_guard_insertion(source, rng)
        assert result == source

    def test_produces_valid_python(self) -> None:
        source = (
            "def solve(nums, target):\n"
            "    for i in range(len(nums)):\n"
            "        for j in range(i + 1, len(nums)):\n"
            "            if nums[i] + nums[j] == target:\n"
            "                return (i + 1, j + 1)\n"
            "    return (1, 1)\n"
        )
        for seed in range(20):
            result = mutate_guard_insertion(source, random.Random(seed))
            ast.parse(result)


class TestLoopConversion:
    def test_converts_for_to_while(self) -> None:
        source = (
            "def solve(x):\n"
            "    total = 0\n"
            "    for i in range(x):\n"
            "        total += i\n"
            "    return total\n"
        )
        rng = random.Random(42)
        result = mutate_loop_conversion(source, rng)
        # Must be syntactically valid
        ast.parse(result)
        # Should differ from original
        if result != source:
            assert "while" in result

    def test_returns_original_when_no_loops(self) -> None:
        source = "def solve(x):\n    return x + 1\n"
        rng = random.Random(0)
        result = mutate_loop_conversion(source, rng)
        assert result == source

    def test_produces_valid_python(self) -> None:
        source = (
            "def solve(x):\n"
            "    result = []\n"
            "    for i in range(x):\n"
            "        result.append(i * 2)\n"
            "    return result\n"
        )
        for seed in range(20):
            result = mutate_loop_conversion(source, random.Random(seed))
            ast.parse(result)


class TestVariableExtraction:
    def test_extracts_repeated_subexpression(self) -> None:
        source = "def solve(x):\n    a = x + 1\n    b = x + 1\n    return a + b\n"
        rng = random.Random(42)
        result = mutate_variable_extraction(source, rng)
        ast.parse(result)

    def test_returns_original_when_no_candidates(self) -> None:
        source = "def solve(x):\n    return x\n"
        rng = random.Random(0)
        result = mutate_variable_extraction(source, rng)
        assert result == source

    def test_produces_valid_python(self) -> None:
        source = "def solve(x, y):\n    if x + y > 10:\n        return x + y\n    return 0\n"
        for seed in range(20):
            result = mutate_variable_extraction(source, random.Random(seed))
            ast.parse(result)


class TestSemanticMutationConfig:
    def test_enable_semantic_mutation_flag(self) -> None:
        from alife_core.models import RunConfig

        config = RunConfig(enable_semantic_mutation=True)
        assert config.enable_semantic_mutation is True

    def test_enable_semantic_mutation_defaults_false(self) -> None:
        from alife_core.models import RunConfig

        config = RunConfig()
        assert config.enable_semantic_mutation is False


class TestSemanticMutationValidity:
    """Batch validity tests: semantic operators should produce >=80% valid code."""

    def test_guard_insertion_validity_rate(self) -> None:
        source = (
            "def solve(nums, target):\n"
            "    for i in range(len(nums)):\n"
            "        for j in range(i + 1, len(nums)):\n"
            "            if nums[i] + nums[j] == target:\n"
            "                return (i + 1, j + 1)\n"
            "    return (1, 1)\n"
        )
        valid = 0
        total = 50
        for seed in range(total):
            result = mutate_guard_insertion(source, random.Random(seed))
            try:
                ast.parse(result)
                valid += 1
            except SyntaxError:
                pass
        assert valid / total >= 0.8

    def test_loop_conversion_validity_rate(self) -> None:
        source = (
            "def solve(x):\n"
            "    total = 0\n"
            "    for i in range(x):\n"
            "        total += i\n"
            "    return total\n"
        )
        valid = 0
        total = 50
        for seed in range(total):
            result = mutate_loop_conversion(source, random.Random(seed))
            try:
                ast.parse(result)
                valid += 1
            except SyntaxError:
                pass
        assert valid / total >= 0.8

    def test_variable_extraction_validity_rate(self) -> None:
        source = "def solve(x, y):\n    if x + y > 10:\n        return x + y\n    return 0\n"
        valid = 0
        total = 50
        for seed in range(total):
            result = mutate_variable_extraction(source, random.Random(seed))
            try:
                ast.parse(result)
                valid += 1
            except SyntaxError:
                pass
        assert valid / total >= 0.8
