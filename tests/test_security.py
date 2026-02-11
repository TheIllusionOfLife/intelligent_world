from alife_core.mutation.validation import validate_candidate


def test_validation_rejects_forbidden_import() -> None:
    code = "import os\n\ndef solve(x):\n    return x\n"

    result = validate_candidate(code)

    assert result.is_valid is False
    assert result.stage == "ast_policy"
    assert "forbidden import" in result.reason


def test_validation_rejects_syntax_error() -> None:
    code = "def solve(x):\nreturn x\n"

    result = validate_candidate(code)

    assert result.is_valid is False
    assert result.stage == "parse"


def test_validation_rejects_forbidden_attribute_call() -> None:
    code = "import builtins\n\ndef solve(x):\n    return builtins.exec('x=1')\n"

    result = validate_candidate(code)

    assert result.is_valid is False
    assert result.stage == "ast_policy"
    assert "forbidden call" in result.reason


def test_validation_rejects_getattr_builtin_bypass() -> None:
    code = "def solve(x):\n    return getattr(__builtins__, 'open')('/tmp/x').read()\n"

    result = validate_candidate(code)

    assert result.is_valid is False
    assert result.stage == "ast_policy"


def test_validation_rejects_dunder_attribute_access() -> None:
    code = "def solve(x):\n    return x.__class__\n"

    result = validate_candidate(code)

    assert result.is_valid is False
    assert result.stage == "ast_policy"
