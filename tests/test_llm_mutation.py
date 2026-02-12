"""Tests for LLM-assisted mutation operator."""

import ast
import random

from alife_core.mutation.llm import mutate_with_llm


class TestMutateWithLlm:
    def test_returns_llm_output_when_valid(self, monkeypatch) -> None:
        """LLM mutation returns the LLM-generated code when it's valid Python."""
        import subprocess

        improved_code = (
            "def solve(x, y):\n    left = 0\n    right = len(x) - 1\n    return (1, 1)\n"
        )

        class FakeResult:
            returncode = 0
            stdout = f"```python\n{improved_code}```"
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: FakeResult())

        source = "def solve(x, y):\n    return (1, 1)\n"
        rng = random.Random(42)
        result = mutate_with_llm(source, rng, model="test-model", timeout=5.0)

        assert result is not None
        ast.parse(result)

    def test_returns_none_on_ollama_failure(self, monkeypatch) -> None:
        """LLM mutation returns None when Ollama returns non-zero exit code."""
        import subprocess

        class FakeResult:
            returncode = 1
            stdout = ""
            stderr = "model not found"

        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: FakeResult())

        source = "def solve(x, y):\n    return (1, 1)\n"
        rng = random.Random(42)
        result = mutate_with_llm(source, rng, model="test-model", timeout=5.0)

        assert result is None

    def test_returns_none_on_timeout(self, monkeypatch) -> None:
        """LLM mutation returns None when Ollama times out."""
        import subprocess

        def raise_timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired("ollama", 5.0)

        monkeypatch.setattr(subprocess, "run", raise_timeout)

        source = "def solve(x, y):\n    return (1, 1)\n"
        rng = random.Random(42)
        result = mutate_with_llm(source, rng, model="test-model", timeout=5.0)

        assert result is None

    def test_returns_none_on_file_not_found(self, monkeypatch) -> None:
        """LLM mutation returns None when ollama binary is not found."""
        import subprocess

        def raise_fnf(*args, **kwargs):
            raise FileNotFoundError("ollama")

        monkeypatch.setattr(subprocess, "run", raise_fnf)

        source = "def solve(x, y):\n    return (1, 1)\n"
        rng = random.Random(42)
        result = mutate_with_llm(source, rng, model="test-model", timeout=5.0)

        assert result is None

    def test_returns_none_on_invalid_python_output(self, monkeypatch) -> None:
        """LLM mutation returns None when Ollama returns unparseable code."""
        import subprocess

        class FakeResult:
            returncode = 0
            stdout = "This is not valid python code }{]["
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: FakeResult())

        source = "def solve(x, y):\n    return (1, 1)\n"
        rng = random.Random(42)
        result = mutate_with_llm(source, rng, model="test-model", timeout=5.0)

        assert result is None

    def test_returns_none_on_empty_output(self, monkeypatch) -> None:
        """LLM mutation returns None when Ollama returns empty output."""
        import subprocess

        class FakeResult:
            returncode = 0
            stdout = ""
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: FakeResult())

        source = "def solve(x, y):\n    return (1, 1)\n"
        rng = random.Random(42)
        result = mutate_with_llm(source, rng, model="test-model", timeout=5.0)

        assert result is None

    def test_passes_model_and_timeout_to_subprocess(self, monkeypatch) -> None:
        """LLM mutation passes the correct model and timeout to subprocess."""
        import subprocess

        captured_args: dict = {}

        def capture_run(*args, **kwargs):
            captured_args["args"] = args
            captured_args["kwargs"] = kwargs

            class FakeResult:
                returncode = 0
                stdout = "def solve(x, y):\n    return (1, 1)\n"
                stderr = ""

            return FakeResult()

        monkeypatch.setattr(subprocess, "run", capture_run)

        source = "def solve(x, y):\n    return (1, 1)\n"
        rng = random.Random(42)
        mutate_with_llm(source, rng, model="my-model:7b", timeout=15.0)

        cmd = captured_args["args"][0]
        assert "my-model:7b" in cmd
        assert captured_args["kwargs"]["timeout"] == 15.0

    def test_returns_none_when_source_unchanged(self, monkeypatch) -> None:
        """LLM mutation returns None when output is identical to input."""
        import subprocess

        source = "def solve(x, y):\n    return (1, 1)\n"

        class FakeResult:
            returncode = 0
            stdout = source
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: FakeResult())

        rng = random.Random(42)
        result = mutate_with_llm(source, rng, model="test-model", timeout=5.0)

        assert result is None


class TestLlmMutationConfig:
    def test_config_fields_exist(self) -> None:
        from alife_core.models import RunConfig

        config = RunConfig(llm_mutation_rate=0.1, llm_mutation_budget=10)
        assert config.llm_mutation_rate == 0.1
        assert config.llm_mutation_budget == 10

    def test_config_defaults(self) -> None:
        from alife_core.models import RunConfig

        config = RunConfig()
        assert config.llm_mutation_rate == 0.05
        assert config.llm_mutation_budget == 20


class TestLlmMutationIntegration:
    def test_mutate_code_uses_llm_when_enabled_and_budget_available(self, monkeypatch) -> None:
        """When LLM mutation is enabled and budget > 0, _mutate_code can invoke it."""
        from alife_core import runtime

        llm_calls: list[str] = []

        def fake_llm_mutate(source, rng, model, timeout):
            llm_calls.append(source)
            return source.replace("return (1, 1)", "return (1, 2)")

        monkeypatch.setattr("alife_core.mutation.llm.mutate_with_llm", fake_llm_mutate)

        source = "def two_sum_sorted(numbers, target):\n    return (1, 1)\n"
        rng = random.Random(42)

        # Force LLM path by calling with high rate
        result, llm_used = runtime._mutate_code(
            source,
            rng,
            intensity=1,
            enable_llm=True,
            llm_mutation_rate=1.0,  # Always trigger
            llm_budget_remaining=5,
            llm_model="test-model",
            llm_timeout=5.0,
        )

        assert llm_calls, "LLM mutation should have been called"
        assert result != source
        assert llm_used == 1

    def test_mutate_code_skips_llm_when_budget_exhausted(self, monkeypatch) -> None:
        """When LLM budget is 0, _mutate_code should not call LLM mutation."""
        from alife_core import runtime

        llm_calls: list[str] = []

        def fake_llm_mutate(source, rng, model, timeout):
            llm_calls.append(source)
            return source.replace("return (1, 1)", "return (1, 2)")

        monkeypatch.setattr("alife_core.mutation.llm.mutate_with_llm", fake_llm_mutate)

        source = "def two_sum_sorted(numbers, target):\n    return (1, 1)\n"
        rng = random.Random(42)

        _result, llm_used = runtime._mutate_code(
            source,
            rng,
            intensity=1,
            enable_llm=True,
            llm_mutation_rate=1.0,
            llm_budget_remaining=0,  # Budget exhausted
            llm_model="test-model",
            llm_timeout=5.0,
        )

        assert not llm_calls, "LLM mutation should NOT have been called with 0 budget"
        assert llm_used == 0

    def test_mutate_code_skips_llm_when_not_enabled(self, monkeypatch) -> None:
        """When enable_llm is False, LLM mutation should not be called."""
        from alife_core import runtime

        llm_calls: list[str] = []

        def fake_llm_mutate(source, rng, model, timeout):
            llm_calls.append(source)
            return source.replace("return (1, 1)", "return (1, 2)")

        monkeypatch.setattr("alife_core.mutation.llm.mutate_with_llm", fake_llm_mutate)

        source = "def two_sum_sorted(numbers, target):\n    return (1, 1)\n"
        rng = random.Random(42)

        _result, llm_used = runtime._mutate_code(
            source,
            rng,
            intensity=1,
            enable_llm=False,
            llm_mutation_rate=1.0,
            llm_budget_remaining=10,
            llm_model="test-model",
            llm_timeout=5.0,
        )

        assert not llm_calls, "LLM mutation should NOT be called when enable_llm=False"
        assert llm_used == 0

    def test_mutate_code_falls_back_to_ast_when_llm_returns_none(self, monkeypatch) -> None:
        """When LLM mutation returns None, _mutate_code falls back to AST mutation."""
        from alife_core import runtime

        def fake_llm_mutate(source, rng, model, timeout):
            return None  # LLM failed

        monkeypatch.setattr("alife_core.mutation.llm.mutate_with_llm", fake_llm_mutate)

        source = "def two_sum_sorted(numbers, target):\n    return (1, 1)\n"
        rng = random.Random(42)

        result, llm_used = runtime._mutate_code(
            source,
            rng,
            intensity=1,
            enable_llm=True,
            llm_mutation_rate=1.0,
            llm_budget_remaining=5,
            llm_model="test-model",
            llm_timeout=5.0,
        )

        # LLM was attempted but failed, should still get a result from AST mutation
        assert isinstance(result, str)
        assert llm_used == 1  # Call was made even though it failed
