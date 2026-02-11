import ast

from alife_core.models import ValidationResult

_FORBIDDEN_IMPORTS = {"os", "sys", "subprocess", "socket", "pathlib"}
_FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__"}


def _is_forbidden_import(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in _FORBIDDEN_IMPORTS:
                return True
    if isinstance(node, ast.ImportFrom) and node.module:
        root = node.module.split(".")[0]
        if root in _FORBIDDEN_IMPORTS:
            return True
    return False


def _is_forbidden_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in _FORBIDDEN_CALLS
    return False


def validate_candidate(code: str) -> ValidationResult:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ValidationResult(is_valid=False, stage="parse", reason=str(exc))

    try:
        compile(tree, "<candidate>", "exec")
    except SyntaxError as exc:
        return ValidationResult(is_valid=False, stage="compile", reason=str(exc))

    for node in ast.walk(tree):
        if _is_forbidden_import(node):
            return ValidationResult(
                is_valid=False,
                stage="ast_policy",
                reason="forbidden import detected",
            )
        if _is_forbidden_call(node):
            return ValidationResult(
                is_valid=False,
                stage="ast_policy",
                reason="forbidden call detected",
            )

    return ValidationResult(is_valid=True, stage="ok", reason="")
