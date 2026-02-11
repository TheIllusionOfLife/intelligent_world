import ast
import hashlib
import logging
import math
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import yaml

from alife_core.agent.curriculum import should_unlock_next_task
from alife_core.agent.lifecycle import AgentState, apply_step_outcome
from alife_core.evaluator.core import evaluate_candidate
from alife_core.logging.events import write_event, write_run_start
from alife_core.models import RunConfig, TaskSpec
from alife_core.mutation.validation import validate_candidate
from alife_core.tasks.builtin import load_builtin_tasks

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    log_path: Path
    completed_tasks: list[str]


def load_run_config(config_path: Path, seed_override: int | None = None) -> RunConfig:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if loaded is None:
        values: dict[str, object] = {}
    elif isinstance(loaded, dict):
        values = loaded
    else:
        raise ValueError(f"Config must be a mapping, got: {type(loaded).__name__}")

    config = RunConfig(**values)
    if seed_override is not None:
        config = replace(config, seed=seed_override)
    return config


def _temperature_for_step(config: RunConfig, step: int) -> float:
    return max(1e-9, config.initial_temperature * (config.cooling_rate**step))


def _effective_w2(config: RunConfig, step: int) -> float:
    return max(config.w2_floor, config.w2_ast_edit_cost * (config.decay_factor**step))


def _ast_node_count(source: str) -> int:
    return sum(1 for _ in ast.walk(ast.parse(source)))


def compute_ast_edit_cost(previous_code: str, candidate_code: str) -> float:
    if previous_code == candidate_code:
        return 0.0
    previous_count = _ast_node_count(previous_code)
    candidate_count = _ast_node_count(candidate_code)
    return float(abs(candidate_count - previous_count) + 1)


def _mutate_code(current: str, rng: random.Random, intensity: int = 1) -> str:
    tree = ast.parse(current)
    edits = 0
    for node in ast.walk(tree):
        if edits >= intensity:
            break
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add) and rng.random() < 0.4:
            node.op = ast.Sub() if rng.random() < 0.5 else ast.Mult()
            edits += 1
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and rng.random() < 0.6:
            node.value = node.value + rng.choice([-2, -1, 1, 2])
            edits += 1
    return ast.unparse(tree) + "\n"


def _bootstrap_seed(task: TaskSpec) -> str:
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
    raise ValueError(f"Unknown task seed: {task.name}")


def _resolve_docker_digest(config: RunConfig) -> str:
    if config.sandbox_backend != "docker":
        return "process-backend"
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", "--format={{.Id}}", config.docker_image],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        LOGGER.warning("Unable to resolve docker image digest for %s", config.docker_image)
        return "unknown"
    digest = completed.stdout.strip()
    return digest if digest else "unknown"


def _resolve_git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        LOGGER.warning("Unable to resolve framework git SHA")
        return "unknown"
    sha = completed.stdout.strip()
    return sha if sha else "unknown"


def run_experiment(task_name: str, config: RunConfig, output_root: Path) -> RunSummary:
    tasks = load_builtin_tasks()
    if task_name not in tasks:
        raise ValueError(f"Unknown task: {task_name}")

    run_id = datetime.now(UTC).strftime("run-%Y%m%dT%H%M%S%fZ") + f"-{uuid4().hex[:6]}"
    logs_dir = output_root / "logs"
    organisms_dir = output_root / "organisms"
    log_path = logs_dir / f"{run_id}.jsonl"

    parameters = asdict(config)
    config_hash = hashlib.sha256(str(sorted(parameters.items())).encode("utf-8")).hexdigest()
    write_run_start(
        log_path=log_path,
        run_id=run_id,
        config=config,
        framework_git_sha=_resolve_git_sha(),
        docker_image_digest=_resolve_docker_digest(config),
        python_version=sys.version.split()[0],
        cpu_architecture=platform.machine(),
        parameters={**parameters, "config_hash": config_hash},
    )

    rng = random.Random(config.seed)
    completed_tasks: list[str] = []
    task = tasks[task_name]

    current_code = _bootstrap_seed(task)
    validation = validate_candidate(current_code)
    if not validation.is_valid:
        raise RuntimeError(f"Bootstrap failed validation for {task.name}: {validation.reason}")

    current_eval = evaluate_candidate(current_code, task=task, edit_cost=0.0, config=config)
    state = AgentState(
        energy=config.initial_energy,
        stagnation_steps=0,
        best_fitness=current_eval.fitness,
    )

    write_event(
        log_path,
        {
            "type": "step",
            "task": task.name,
            "step": 0,
            "accepted": True,
            "mutation_rejected": False,
            "energy": state.energy,
            "fitness": current_eval.fitness,
            "fitness_delta": 0.0,
            "train_pass_ratio": current_eval.train_pass_ratio,
            "hidden_pass_ratio": current_eval.hidden_pass_ratio,
            "goodhart_warning": (
                (current_eval.train_pass_ratio - current_eval.hidden_pass_ratio)
                > config.goodhart_gap_threshold
            ),
            "temperature": _temperature_for_step(config, 0),
            "w2_effective": _effective_w2(config, 0),
        },
    )

    for step in range(1, config.max_steps + 1):
        mutation_size = 1 + (state.stagnation_steps // max(1, config.mutation_stagnation_window))
        candidate_code = _mutate_code(current_code, rng, intensity=mutation_size)
        validation = validate_candidate(candidate_code)
        if not validation.is_valid:
            write_event(
                log_path,
                {
                    "type": "step",
                    "task": task.name,
                    "step": step,
                    "accepted": False,
                    "mutation_rejected": True,
                    "rejection_stage": validation.stage,
                    "rejection_reason": validation.reason,
                    "energy": state.energy,
                    "fitness": current_eval.fitness,
                    "fitness_delta": 0.0,
                    "train_pass_ratio": current_eval.train_pass_ratio,
                    "hidden_pass_ratio": current_eval.hidden_pass_ratio,
                    "goodhart_warning": (
                        (current_eval.train_pass_ratio - current_eval.hidden_pass_ratio)
                        > config.goodhart_gap_threshold
                    ),
                    "temperature": _temperature_for_step(config, step),
                    "w2_effective": _effective_w2(config, step),
                },
            )
            continue

        edit_cost = compute_ast_edit_cost(current_code, candidate_code)
        step_config = replace(config, w2_ast_edit_cost=_effective_w2(config, step))
        candidate_eval = evaluate_candidate(
            candidate_code,
            task=task,
            edit_cost=edit_cost,
            config=step_config,
        )

        previous_fitness = current_eval.fitness
        accepted = candidate_eval.fitness >= previous_fitness
        if not accepted:
            loss = max(0.0, previous_fitness - candidate_eval.fitness)
            temperature = _temperature_for_step(config, step)
            accepted = rng.random() < math.exp(-loss / temperature)

        state = apply_step_outcome(
            state=state,
            accepted=accepted,
            previous_fitness=previous_fitness,
            evaluation=candidate_eval,
            mutation_rejected=False,
            config=config,
        )

        if accepted:
            current_code = candidate_code
            current_eval = candidate_eval
            archive_path = organisms_dir / "archive" / run_id / task.name / f"{step}.py"
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_text(current_code, encoding="utf-8")

            current_path = organisms_dir / "current" / f"{task.name}.py"
            current_path.parent.mkdir(parents=True, exist_ok=True)
            current_path.write_text(current_code, encoding="utf-8")

        write_event(
            log_path,
            {
                "type": "step",
                "task": task.name,
                "step": step,
                "accepted": accepted,
                "mutation_rejected": False,
                "energy": state.energy,
                "fitness": current_eval.fitness,
                "fitness_delta": (candidate_eval.fitness - previous_fitness) if accepted else 0.0,
                "train_pass_ratio": current_eval.train_pass_ratio,
                "hidden_pass_ratio": current_eval.hidden_pass_ratio,
                "goodhart_warning": (
                    (current_eval.train_pass_ratio - current_eval.hidden_pass_ratio)
                    > config.goodhart_gap_threshold
                ),
                "temperature": _temperature_for_step(config, step),
                "w2_effective": _effective_w2(config, step),
            },
        )

        if should_unlock_next_task(current_eval, config):
            completed_tasks.append(task.name)
            break

        if state.energy <= 0 or state.stagnation_steps >= config.n_stagnation:
            break

    return RunSummary(run_id=run_id, log_path=log_path, completed_tasks=completed_tasks)
