from alife_core.bootstrap import BootstrapError, generate_seed
from alife_core.models import RunConfig
from alife_core.tasks.builtin import load_builtin_tasks


def test_generate_seed_uses_static_backend() -> None:
    task = load_builtin_tasks()["two_sum_sorted"]
    config = RunConfig(bootstrap_backend="static")

    seed = generate_seed(task, config)

    assert "def two_sum_sorted" in seed


def test_generate_seed_raises_when_ollama_unavailable_and_no_fallback(monkeypatch) -> None:
    from alife_core import bootstrap

    task = load_builtin_tasks()["two_sum_sorted"]
    config = RunConfig(
        bootstrap_backend="ollama",
        ollama_model="gpt-oss:20b",
        bootstrap_fallback_to_static=False,
    )

    def fail_ollama(*_args, **_kwargs) -> str:
        raise BootstrapError("failure")

    monkeypatch.setattr(bootstrap, "_generate_ollama_seed", fail_ollama)

    try:
        generate_seed(task, config)
    except BootstrapError as exc:
        assert "failure" in str(exc)
    else:
        raise AssertionError("expected BootstrapError")


def test_generate_seed_falls_back_to_static_when_enabled(monkeypatch) -> None:
    from alife_core import bootstrap

    task = load_builtin_tasks()["run_length_encode"]
    config = RunConfig(
        bootstrap_backend="ollama",
        ollama_model="gpt-oss:20b",
        bootstrap_fallback_to_static=True,
    )

    def fail_ollama(*_args, **_kwargs) -> str:
        raise BootstrapError("failure")

    monkeypatch.setattr(bootstrap, "_generate_ollama_seed", fail_ollama)

    try:
        generate_seed(task, config)
    except BootstrapError as exc:
        assert "failure" in str(exc)
    else:
        raise AssertionError("expected BootstrapError")


def test_extract_python_code_joins_multiple_python_blocks() -> None:
    from alife_core.bootstrap import _extract_python_code

    text = (
        "preface\n"
        "```python\n"
        "def a():\n    return 1\n"
        "```\n"
        "middle\n"
        "```python\n"
        "def b():\n    return 2\n"
        "```\n"
    )

    code = _extract_python_code(text)

    assert "def a" in code
    assert "def b" in code


def test_generate_ollama_seed_requires_function_name(monkeypatch) -> None:
    from alife_core import bootstrap

    task = load_builtin_tasks()["two_sum_sorted"]
    config = RunConfig(
        bootstrap_backend="ollama",
        ollama_model="gpt-oss:20b",
    )

    class Completed:
        returncode = 0
        stdout = "def wrong_name(x):\n    return x\n"
        stderr = ""

    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *args, **kwargs: Completed())

    try:
        bootstrap._generate_ollama_seed(task, config)
    except BootstrapError as exc:
        assert "function name" in str(exc)
    else:
        raise AssertionError("expected BootstrapError for missing function name")


def test_generate_ollama_seed_invokes_expected_command(monkeypatch) -> None:
    from alife_core import bootstrap

    task = load_builtin_tasks()["two_sum_sorted"]
    config = RunConfig(
        bootstrap_backend="ollama",
        ollama_model="gpt-oss:20b",
    )
    called = {"cmd": None}

    class Completed:
        returncode = 0
        stdout = "def two_sum_sorted(numbers, target):\n    return (1, 2)\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        called["cmd"] = cmd
        return Completed()

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    bootstrap._generate_ollama_seed(task, config)

    assert called["cmd"][:3] == ["ollama", "run", "gpt-oss:20b"]
