import ast
import copy
import hashlib
import logging
import math
import os
import platform
import random
import statistics
import subprocess
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import yaml

from alife_core.agent.curriculum import should_unlock_next_task
from alife_core.agent.lifecycle import AgentState, apply_step_outcome
from alife_core.bootstrap import BootstrapError, generate_seed, static_seed
from alife_core.evaluator.core import evaluate_candidate
from alife_core.logging.events import write_event, write_run_start
from alife_core.models import BootstrapBackend, EvolutionMode, OrganismState, RunConfig
from alife_core.mutation.validation import validate_candidate
from alife_core.tasks.builtin import load_builtin_tasks

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    log_path: Path
    completed_tasks: list[str]


def load_run_config(
    config_path: Path,
    seed_override: int | None = None,
    bootstrap_backend_override: BootstrapBackend | None = None,
    ollama_model_override: str | None = None,
    run_curriculum_override: bool | None = None,
    allow_unsafe_process_backend_override: bool | None = None,
    evolution_mode_override: EvolutionMode | None = None,
    population_size_override: int | None = None,
    elite_count_override: int | None = None,
    tournament_k_override: int | None = None,
    crossover_rate_override: float | None = None,
    mutation_rate_override: float | None = None,
    max_generations_override: int | None = None,
    population_workers_override: int | None = None,
) -> RunConfig:
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
    if bootstrap_backend_override is not None:
        config = replace(config, bootstrap_backend=bootstrap_backend_override)
    if ollama_model_override is not None:
        config = replace(config, ollama_model=ollama_model_override)
    if run_curriculum_override is not None:
        config = replace(config, run_curriculum=run_curriculum_override)
    if allow_unsafe_process_backend_override is not None:
        config = replace(config, allow_unsafe_process_backend=allow_unsafe_process_backend_override)
    if evolution_mode_override is not None:
        config = replace(config, evolution_mode=evolution_mode_override)
    if population_size_override is not None:
        config = replace(config, population_size=population_size_override)
    if elite_count_override is not None:
        config = replace(config, elite_count=elite_count_override)
    if tournament_k_override is not None:
        config = replace(config, tournament_k=tournament_k_override)
    if crossover_rate_override is not None:
        config = replace(config, crossover_rate=crossover_rate_override)
    if mutation_rate_override is not None:
        config = replace(config, mutation_rate=mutation_rate_override)
    if max_generations_override is not None:
        config = replace(config, max_generations=max_generations_override)
    if population_workers_override is not None:
        config = replace(config, population_workers=population_workers_override)
    _validate_run_config(config)
    return config


def _validate_run_config(config: RunConfig) -> None:
    if not 0.0 <= config.crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be within [0.0, 1.0]")
    if not 0.0 <= config.mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be within [0.0, 1.0]")
    if config.population_workers < 1:
        raise ValueError("population_workers must be >= 1")
    if config.tournament_k < 1:
        raise ValueError("tournament_k must be >= 1")
    if config.max_generations < 0:
        raise ValueError("max_generations must be >= 0")
    if config.population_size < 2 and config.evolution_mode == "population":
        raise ValueError("population_size must be at least 2 for population mode")


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


def _mutate_statement_swap(tree: ast.AST, rng: random.Random) -> bool:
    for node in ast.walk(tree):
        if (
            hasattr(node, "body")
            and isinstance(node.body, list)
            and len(node.body) >= 2
            and all(isinstance(item, ast.stmt) for item in node.body)
            and rng.random() < 0.2
        ):
            left = rng.choice(range(0, len(node.body) - 1))
            node.body[left], node.body[left + 1] = node.body[left + 1], node.body[left]
            return True
    return False


def _mutate_binop(tree: ast.AST, rng: random.Random) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add) and rng.random() < 0.4:
            node.op = ast.Sub() if rng.random() < 0.5 else ast.Mult()
            return True
    return False


def _mutate_compare(tree: ast.AST, rng: random.Random) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and rng.random() < 0.5:
            op = node.ops[0]
            if isinstance(op, ast.Lt):
                node.ops[0] = ast.Gt()
            elif isinstance(op, ast.Gt):
                node.ops[0] = ast.Lt()
            elif isinstance(op, ast.LtE):
                node.ops[0] = ast.GtE()
            elif isinstance(op, ast.GtE):
                node.ops[0] = ast.LtE()
            elif isinstance(op, ast.Eq):
                node.ops[0] = ast.NotEq()
            elif isinstance(op, ast.NotEq):
                node.ops[0] = ast.Eq()
            else:
                continue
            return True
    return False


def _mutate_boolop(tree: ast.AST, rng: random.Random) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.BoolOp) and rng.random() < 0.4:
            if isinstance(node.op, ast.And):
                node.op = ast.Or()
                return True
            if isinstance(node.op, ast.Or):
                node.op = ast.And()
                return True
    return False


def _mutate_constant(tree: ast.AST, rng: random.Random) -> bool:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and rng.random() < 0.6
        ):
            node.value = node.value + rng.choice([-2, -1, 1, 2])
            return True
    return False


_MUTATORS = {
    "statement_swap": _mutate_statement_swap,
    "binop": _mutate_binop,
    "compare": _mutate_compare,
    "boolop": _mutate_boolop,
    "constant": _mutate_constant,
}


def _mutate_code(
    current: str,
    rng: random.Random,
    intensity: int = 1,
    prefer_structural: bool = False,
) -> str:
    tree = ast.parse(current)
    order = (
        ["statement_swap", "compare", "boolop", "binop", "constant"]
        if prefer_structural
        else ["binop", "constant", "compare", "boolop", "statement_swap"]
    )
    for _ in range(intensity):
        chosen = rng.choice(order)
        start = order.index(chosen)
        probe_order = order[start:] + order[:start]
        for mutator_name in probe_order:
            if _MUTATORS[mutator_name](tree, rng):
                break
    return ast.unparse(tree) + "\n"


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


def _ensure_docker_daemon_available(config: RunConfig) -> None:
    if config.sandbox_backend != "docker":
        return
    try:
        completed = subprocess.run(
            ["docker", "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise RuntimeError("Docker daemon unavailable for docker sandbox backend") from exc
    if completed.returncode != 0:
        raise RuntimeError("Docker daemon unavailable for docker sandbox backend")


def run_single_agent_experiment(task_name: str, config: RunConfig, output_root: Path) -> RunSummary:
    _validate_run_config(config)
    _ensure_docker_daemon_available(config)
    if (
        config.sandbox_backend == "process"
        and config.bootstrap_backend == "ollama"
        and not config.allow_unsafe_process_backend
    ):
        raise ValueError(
            "process backend with ollama bootstrap is unsafe; "
            "set allow_unsafe_process_backend=true to opt in explicitly"
        )

    tasks = load_builtin_tasks()
    if task_name not in tasks:
        raise ValueError(f"Unknown task: {task_name}")
    task_names = list(tasks)

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
    state: AgentState | None = None
    current_task_index = task_names.index(task_name)
    while current_task_index < len(task_names):
        task = tasks[task_names[current_task_index]]
        mutation_outcomes: deque[tuple[bool, bool]] = deque(maxlen=max(1, config.viability_window))
        mutation_fallback_active = False

        bootstrap_fallback_used = False
        bootstrap_error = ""
        bootstrap_backend_used = config.bootstrap_backend
        try:
            current_code = generate_seed(task, config)
        except BootstrapError as exc:
            if not config.bootstrap_fallback_to_static:
                raise RuntimeError(f"Bootstrap failed for {task.name}: {exc}") from exc
            LOGGER.warning("Bootstrap failed with backend %s: %s", config.bootstrap_backend, exc)
            current_code = static_seed(task)
            bootstrap_backend_used = "static"
            bootstrap_fallback_used = True
            bootstrap_error = str(exc)

        validation = validate_candidate(current_code)
        if (
            not validation.is_valid
            and config.bootstrap_backend == "ollama"
            and config.bootstrap_fallback_to_static
            and bootstrap_backend_used != "static"
        ):
            LOGGER.warning(
                "Bootstrap output invalid; falling back to static seed for %s", task.name
            )
            current_code = static_seed(task)
            bootstrap_backend_used = "static"
            bootstrap_fallback_used = True
            bootstrap_error = validation.reason
            validation = validate_candidate(current_code)

        if not validation.is_valid:
            raise RuntimeError(f"Bootstrap failed validation for {task.name}: {validation.reason}")

        current_eval = evaluate_candidate(current_code, task=task, edit_cost=0.0, config=config)
        if state is None:
            state = AgentState(
                energy=config.initial_energy,
                stagnation_steps=0,
                best_fitness=current_eval.fitness,
            )
        else:
            state = replace(
                state,
                energy=round(max(0.0, state.energy - config.base_survival_cost), 10),
                stagnation_steps=0,
                best_fitness=current_eval.fitness,
            )
        archive_path = organisms_dir / "archive" / run_id / task.name / "0.py"
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
                "bootstrap_backend": bootstrap_backend_used,
                "bootstrap_model": config.ollama_model
                if bootstrap_backend_used == "ollama"
                else "",
                "bootstrap_fallback_used": bootstrap_fallback_used,
                "bootstrap_error": bootstrap_error,
                "rolling_validity_rate": 0.0,
                "rolling_improvement_rate": 0.0,
                "mutation_fallback_active": mutation_fallback_active,
                "mutation_mode": "bootstrap",
            },
        )

        unlocked = False
        if should_unlock_next_task(current_eval, config):
            completed_tasks.append(task.name)
            unlocked = True

        for step in range(1, config.max_steps + 1):
            if unlocked:
                break

            mutation_size = 1 + (
                state.stagnation_steps // max(1, config.mutation_stagnation_window)
            )
            mutation_mode = "normal"
            if mutation_fallback_active:
                mutation_size += 1
                mutation_mode = "fallback_template"
            candidate_code = _mutate_code(
                current_code,
                rng,
                intensity=mutation_size,
                prefer_structural=mutation_fallback_active,
            )
            validation = validate_candidate(candidate_code)

            if not validation.is_valid:
                mutation_outcomes.append((False, False))
                valid_count = sum(1 for valid, _ in mutation_outcomes if valid)
                improved_count = sum(
                    1 for valid, improved in mutation_outcomes if valid and improved
                )
                rolling_validity_rate = valid_count / len(mutation_outcomes)
                rolling_improvement_rate = 0.0 if valid_count == 0 else improved_count / valid_count
                mutation_fallback_active = (
                    len(mutation_outcomes) >= config.viability_window
                    and rolling_improvement_rate < config.viability_min_improvement_rate
                )
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
                        "rolling_validity_rate": rolling_validity_rate,
                        "rolling_improvement_rate": rolling_improvement_rate,
                        "mutation_fallback_active": mutation_fallback_active,
                        "mutation_mode": mutation_mode,
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
            improved = candidate_eval.fitness > previous_fitness
            mutation_outcomes.append((True, improved))
            valid_count = sum(1 for valid, _ in mutation_outcomes if valid)
            improved_count = sum(
                1 for valid, is_improved in mutation_outcomes if valid and is_improved
            )
            rolling_validity_rate = valid_count / len(mutation_outcomes)
            rolling_improvement_rate = 0.0 if valid_count == 0 else improved_count / valid_count
            mutation_fallback_active = (
                len(mutation_outcomes) >= config.viability_window
                and rolling_improvement_rate < config.viability_min_improvement_rate
            )

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
                    "fitness_delta": (candidate_eval.fitness - previous_fitness)
                    if accepted
                    else 0.0,
                    "train_pass_ratio": current_eval.train_pass_ratio,
                    "hidden_pass_ratio": current_eval.hidden_pass_ratio,
                    "goodhart_warning": (
                        (current_eval.train_pass_ratio - current_eval.hidden_pass_ratio)
                        > config.goodhart_gap_threshold
                    ),
                    "temperature": _temperature_for_step(config, step),
                    "w2_effective": _effective_w2(config, step),
                    "rolling_validity_rate": rolling_validity_rate,
                    "rolling_improvement_rate": rolling_improvement_rate,
                    "mutation_fallback_active": mutation_fallback_active,
                    "mutation_mode": mutation_mode,
                },
            )

            if should_unlock_next_task(current_eval, config):
                completed_tasks.append(task.name)
                unlocked = True
                break

            if state.energy <= 0 or state.stagnation_steps >= config.n_stagnation:
                break

        if not unlocked or not config.run_curriculum or state.energy <= 0:
            break
        current_task_index += 1

    return RunSummary(run_id=run_id, log_path=log_path, completed_tasks=completed_tasks)


def _ast_shape_fingerprint(source: str) -> str:
    tree = ast.parse(source)

    class _ShapeNormalizer(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            return ast.copy_location(ast.Constant(value=None), node)

        def visit_Name(self, node: ast.Name) -> ast.AST:
            updated = ast.Name(id="_", ctx=node.ctx)
            return ast.copy_location(updated, node)

        def visit_arg(self, node: ast.arg) -> ast.AST:
            updated = ast.arg(arg="_", annotation=None, type_comment=None)
            return ast.copy_location(updated, node)

    normalized = _ShapeNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, include_attributes=False)


def _diversity_score(population: list[OrganismState]) -> float:
    if not population:
        return 0.0
    unique_shapes = len({_ast_shape_fingerprint(item.code) for item in population})
    return unique_shapes / len(population)


def _tournament_select(
    population: list[OrganismState],
    rng: random.Random,
    tournament_k: int,
) -> OrganismState:
    if not population:
        raise ValueError("Population cannot be empty")
    k = min(len(population), max(1, tournament_k))
    candidates = rng.sample(population, k)
    return max(candidates, key=lambda item: item.fitness)


def _crossover_code(parent_a: str, parent_b: str, rng: random.Random) -> str:
    try:
        tree_a = ast.parse(parent_a)
        tree_b = ast.parse(parent_b)
    except SyntaxError:
        return parent_a

    funcs_a = [node for node in tree_a.body if isinstance(node, ast.FunctionDef)]
    funcs_b = [node for node in tree_b.body if isinstance(node, ast.FunctionDef)]
    if not funcs_a or not funcs_b:
        return parent_a
    function_a = funcs_a[0]
    function_b = funcs_b[0]
    if not function_a.body or not function_b.body:
        return parent_a

    if len(function_a.body) == 1 or len(function_b.body) == 1:
        new_body = copy.deepcopy(function_b.body)
    else:
        cut_a = rng.randint(1, len(function_a.body) - 1)
        cut_b = rng.randint(1, len(function_b.body) - 1)
        new_body = copy.deepcopy(function_a.body[:cut_a]) + copy.deepcopy(function_b.body[cut_b:])

    crossed_tree = copy.deepcopy(tree_a)
    crossed_funcs = [node for node in crossed_tree.body if isinstance(node, ast.FunctionDef)]
    if not crossed_funcs:
        return parent_a
    crossed_funcs[0].body = new_body if new_body else copy.deepcopy(function_a.body)
    ast.fix_missing_locations(crossed_tree)
    return ast.unparse(crossed_tree) + "\n"


def _evaluate_population(
    codes: list[str],
    task,
    config: RunConfig,
) -> list[OrganismState]:
    def _evaluate_one(code: str) -> OrganismState:
        result = evaluate_candidate(code, task=task, edit_cost=0.0, config=config)
        return OrganismState(
            code=code,
            fitness=result.fitness,
            train_pass_ratio=result.train_pass_ratio,
            hidden_pass_ratio=result.hidden_pass_ratio,
        )

    workers = min(max(1, config.population_workers), max(1, len(codes)), (os.cpu_count() or 1))
    LOGGER.info("Evaluating population with %s workers", workers)
    if workers == 1:
        return [_evaluate_one(code) for code in codes]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_evaluate_one, codes))


def _initialize_population_for_task(task, config: RunConfig, rng: random.Random) -> list[str]:
    try:
        bootstrap_code = generate_seed(task, config)
    except BootstrapError as exc:
        if not config.bootstrap_fallback_to_static:
            raise RuntimeError(f"Bootstrap failed for {task.name}: {exc}") from exc
        bootstrap_code = static_seed(task)
    if not validate_candidate(bootstrap_code).is_valid:
        bootstrap_code = static_seed(task)

    population_codes = [bootstrap_code]
    attempts = 0
    while len(population_codes) < max(1, config.population_size):
        attempts += 1
        if attempts > config.population_size * 20:
            population_codes.append(bootstrap_code)
            continue
        candidate = _mutate_code(bootstrap_code, rng, intensity=1)
        if validate_candidate(candidate).is_valid:
            population_codes.append(candidate)
    return population_codes


def run_population_experiment(task_name: str, config: RunConfig, output_root: Path) -> RunSummary:
    _validate_run_config(config)
    _ensure_docker_daemon_available(config)
    if (
        config.sandbox_backend == "process"
        and config.bootstrap_backend == "ollama"
        and not config.allow_unsafe_process_backend
    ):
        raise ValueError(
            "process backend with ollama bootstrap is unsafe; "
            "set allow_unsafe_process_backend=true to opt in explicitly"
        )
    tasks = load_builtin_tasks()
    if task_name not in tasks:
        raise ValueError(f"Unknown task: {task_name}")
    if config.population_size < 2:
        raise ValueError("population_size must be at least 2 for population mode")
    if config.elite_count < 1 or config.elite_count >= config.population_size:
        raise ValueError("elite_count must be >= 1 and < population_size")

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
    current_task_names = list(tasks)
    current_task_index = current_task_names.index(task_name)

    while current_task_index < len(current_task_names):
        task = tasks[current_task_names[current_task_index]]
        recent_diversity: deque[float] = deque(maxlen=max(1, config.diversity_window))
        population_codes = _initialize_population_for_task(task, config, rng)
        population = _evaluate_population(population_codes, task, config)

        unlocked = False
        # Evaluate at generation 0 (initial population) and after each reproduction step.
        # This is intentionally max_generations + 1 checkpoints so the final evolved
        # population is scored and unlock-checked before exiting.
        for generation in range(config.max_generations + 1):
            best = max(population, key=lambda item: item.fitness)
            fitnesses = [item.fitness for item in population]
            diversity = _diversity_score(population)
            recent_diversity.append(diversity)

            write_event(
                log_path,
                {
                    "type": "generation_start",
                    "task": task.name,
                    "generation": generation,
                    "population_size": len(population),
                    "step": generation,
                    "energy": None,
                    "mode": "population",
                    "run_id": run_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            write_event(
                log_path,
                {
                    "type": "diversity_snapshot",
                    "task": task.name,
                    "generation": generation,
                    "diversity_score": diversity,
                    "min_diversity_score": config.min_diversity_score,
                    "step": generation,
                    "energy": None,
                    "mode": "population",
                    "run_id": run_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            write_event(
                log_path,
                {
                    "type": "generation_end",
                    "task": task.name,
                    "generation": generation,
                    "best_fitness": best.fitness,
                    "mean_fitness": statistics.fmean(fitnesses),
                    "median_fitness": statistics.median(fitnesses),
                    "best_train_pass_ratio": best.train_pass_ratio,
                    "best_hidden_pass_ratio": best.hidden_pass_ratio,
                    "step": generation,
                    "energy": None,
                    "mode": "population",
                    "run_id": run_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

            archive_path = (
                organisms_dir / "archive" / run_id / task.name / f"gen-{generation}-best.py"
            )
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path.write_text(best.code, encoding="utf-8")
            current_path = organisms_dir / "current" / f"{task.name}.py"
            current_path.parent.mkdir(parents=True, exist_ok=True)
            current_path.write_text(best.code, encoding="utf-8")

            if (
                best.train_pass_ratio >= config.pass_ratio_threshold
                and best.fitness >= config.fitness_threshold
            ):
                completed_tasks.append(task.name)
                unlocked = True
                break

            if len(recent_diversity) >= config.diversity_window and all(
                score <= config.min_diversity_score for score in recent_diversity
            ):
                write_event(
                    log_path,
                    {
                        "type": "extinction_or_convergence",
                        "task": task.name,
                        "generation": generation,
                        "reason": "low_diversity",
                        "diversity_window": list(recent_diversity),
                        "step": generation,
                        "energy": None,
                        "mode": "population",
                        "run_id": run_id,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )
                break

            if generation == config.max_generations:
                break

            sorted_population = sorted(population, key=lambda item: item.fitness, reverse=True)
            elites = sorted_population[: config.elite_count]
            next_codes = [item.code for item in elites]
            while len(next_codes) < config.population_size:
                parent_a = _tournament_select(population, rng, config.tournament_k)
                child_code = parent_a.code
                if rng.random() < config.crossover_rate:
                    parent_b = _tournament_select(population, rng, config.tournament_k)
                    child_code = _crossover_code(parent_a.code, parent_b.code, rng)
                    write_event(
                        log_path,
                        {
                            "type": "crossover",
                            "task": task.name,
                            "generation": generation,
                            "step": generation,
                            "energy": None,
                            "mode": "population",
                            "run_id": run_id,
                            "timestamp": datetime.now(UTC).isoformat(),
                        },
                    )
                if rng.random() < config.mutation_rate:
                    child_code = _mutate_code(child_code, rng, intensity=1)

                validation = validate_candidate(child_code)
                if not validation.is_valid:
                    child_code = parent_a.code
                next_codes.append(child_code)

            population = _evaluate_population(next_codes, task, config)

        if not unlocked or not config.run_curriculum:
            break
        current_task_index += 1

    return RunSummary(run_id=run_id, log_path=log_path, completed_tasks=completed_tasks)


def run_experiment(task_name: str, config: RunConfig, output_root: Path) -> RunSummary:
    _validate_run_config(config)
    if config.evolution_mode == "population":
        return run_population_experiment(
            task_name=task_name, config=config, output_root=output_root
        )
    return run_single_agent_experiment(task_name=task_name, config=config, output_root=output_root)
