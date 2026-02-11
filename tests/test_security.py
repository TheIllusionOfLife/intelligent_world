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
