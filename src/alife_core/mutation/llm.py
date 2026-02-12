"""LLM-assisted mutation operator using Ollama."""

from __future__ import annotations

import ast
import logging
import random
import subprocess

LOGGER = logging.getLogger(__name__)

_MUTATION_PROMPT_TEMPLATE = (
    "You are improving a Python function. Make ONE small improvement.\n"
    "Rules:\n"
    "- Return ONLY valid Python code, no explanations\n"
    "- Keep the same function name and signature\n"
    "- No imports, no helper functions\n"
    "- Make a single targeted change (fix a bug, optimize a loop, improve logic)\n"
    "\n"
    "Current code:\n"
    "```python\n"
    "{source}"
    "```\n"
    "\n"
    "Improved code:\n"
)


def _extract_code(text: str) -> str:
    """Extract Python code from LLM response, handling markdown fences."""
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


def mutate_with_llm(
    source: str,
    rng: random.Random,
    *,
    model: str,
    timeout: float,
) -> str | None:
    """Attempt to mutate source code using an LLM via Ollama.

    Returns the mutated code if successful, or None if the mutation
    failed for any reason (timeout, invalid output, etc.).
    """
    _ = rng  # Reserved for future stochastic prompt variation

    prompt = _MUTATION_PROMPT_TEMPLATE.format(source=source)

    try:
        completed = subprocess.run(
            ["ollama", "run", model, prompt],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        LOGGER.debug("LLM mutation failed: %s", exc)
        return None

    if completed.returncode != 0:
        LOGGER.debug("LLM mutation returned non-zero: %s", completed.stderr.strip())
        return None

    response = completed.stdout.strip()
    if not response:
        LOGGER.debug("LLM mutation returned empty output")
        return None

    code = _extract_code(response)

    try:
        ast.parse(code)
    except SyntaxError:
        LOGGER.debug("LLM mutation returned invalid Python")
        return None

    # Don't return unchanged code
    if code.strip() == source.strip():
        return None

    return code
