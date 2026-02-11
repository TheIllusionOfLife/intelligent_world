import ast

from alife_core.models import ValidationResult

_FORBIDDEN_IMPORTS = {
    "ctypes",
    "importlib",
    "io",
    "os",
    "pathlib",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
}
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


def _attribute_path(node: ast.Attribute) -> str:
    parts = [node.attr]
    current = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _is_forbidden_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in _FORBIDDEN_CALLS
    if isinstance(node.func, ast.Attribute):
        path = _attribute_path(node.func)
        return node.func.attr in _FORBIDDEN_CALLS or path in _FORBIDDEN_CALLS
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
