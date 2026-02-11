"""Bootstrap interfaces for initial organism generation."""

from __future__ import annotations

import ast
import subprocess

from alife_core.models import RunConfig, TaskSpec


class BootstrapError(RuntimeError):
    """Raised when seed generation fails."""


def static_seed(task: TaskSpec) -> str:
    if task.name == "two_sum_sorted":
        return (
            "def two_sum_sorted(numbers, target):\n"
            "    for i in range(len(numbers)):\n"
            "        for j in range(i + 1, len(numbers)):\n"
            "            if numbers[i] + numbers[j] == target:\n"
            "                return (i + 1, j + 1)\n"
            "    return (1, 1)\n"
        )
    if task.name == "run_length_encode":
        return (
            "def run_length_encode(s):\n"
            "    if not s:\n"
            "        return ''\n"
            "    out = []\n"
            "    count = 1\n"
            "    for i in range(1, len(s) + 1):\n"
            "        if i < len(s) and s[i] == s[i - 1]:\n"
            "            count += 1\n"
            "        else:\n"
            "            out.append(s[i - 1] if count == 1 else s[i - 1] + str(count))\n"
            "            count = 1\n"
            "    return ''.join(out)\n"
        )
    if task.name == "slugify":
        return (
            "def slugify(text):\n"
            "    text = text.lower()\n"
            "    chars = []\n"
            "    prev_dash = False\n"
            "    for ch in text:\n"
            "        if ch.isalnum():\n"
            "            chars.append(ch)\n"
            "            prev_dash = False\n"
            "        elif not prev_dash:\n"
            "            chars.append('-')\n"
            "            prev_dash = True\n"
            "    return ''.join(chars).strip('-')\n"
        )
    raise BootstrapError(f"Unknown task seed: {task.name}")


def _extract_python_code(text: str) -> str:
    marker = "```python"
    if marker in text:
        blocks = []
        for part in text.split(marker)[1:]:
            blocks.append(part.split("```", maxsplit=1)[0].strip())
        return "\n\n".join(block for block in blocks if block).strip() + "\n"
    if "```" in text:
        parts = text.split("```")
        blocks = [part.strip() for part in parts[1::2] if part.strip()]
        return "\n\n".join(block for block in blocks if block).strip() + "\n"
    return text.strip() + "\n"


def _generate_ollama_seed(task: TaskSpec, config: RunConfig) -> str:
    prompt = (
        "Write only valid Python code for this function.\n"
        f"Function name: {task.function_name}\n"
        f"Task: {task.prompt}\n"
        "Constraints:\n"
        "- Simplest naive implementation only\n"
        "- Use only loops/conditionals\n"
        "- No imports\n"
        "- No helper functions\n"
        "- No explanations or markdown\n"
    )

    try:
        completed = subprocess.run(
            ["ollama", "run", config.ollama_model, prompt],
            check=False,
            capture_output=True,
            text=True,
            timeout=config.bootstrap_timeout_seconds,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise BootstrapError(f"Ollama invocation failed: {exc}") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise BootstrapError(f"Ollama returned non-zero status: {stderr or completed.returncode}")

    response = completed.stdout.strip()
    if not response:
        raise BootstrapError("Ollama returned empty output")

    code = _extract_python_code(response)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise BootstrapError(f"Ollama returned invalid python: {exc}") from exc

    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    if task.function_name not in function_names:
        raise BootstrapError(f"Ollama output missing required function name: {task.function_name}")

    return code


def generate_seed(task: TaskSpec, config: RunConfig) -> str:
    if config.bootstrap_backend == "static":
        return static_seed(task)
    return _generate_ollama_seed(task, config)
