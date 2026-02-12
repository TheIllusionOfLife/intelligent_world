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
from alife_core.metrics.evolution import (
    ast_max_depth,
    ast_node_count,
    ast_shape_fingerprint,
    compute_generation_metrics,
)
from alife_core.models import BootstrapBackend, EvolutionMode, OrganismState, RunConfig, TaskSpec
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
    if config.novelty_k < 1:
        raise ValueError("novelty_k must be >= 1")
    if config.convergence_patience < 1:
        raise ValueError("convergence_patience must be >= 1")
    if not 0.0 <= config.convergence_entropy_floor <= 1.0:
        raise ValueError("convergence_entropy_floor must be within [0.0, 1.0]")
    if config.convergence_fitness_delta_floor < 0.0:
        raise ValueError("convergence_fitness_delta_floor must be >= 0.0")
    if config.evolution_mode == "population":
        if config.population_size < 2:
            raise ValueError("population_size must be at least 2 for population mode")
        if not 1 <= config.elite_count < config.population_size:
            raise ValueError("elite_count must be >= 1 and < population_size")


def _temperature_for_step(config: RunConfig, step: int) -> float:
    return max(1e-9, config.initial_temperature * (config.cooling_rate**step))


def _effective_w2(config: RunConfig, step: int) -> float:
    return max(config.w2_floor, config.w2_ast_edit_cost * (config.decay_factor**step))


def compute_ast_edit_cost(previous_code: str, candidate_code: str) -> float:
    if previous_code == candidate_code:
        return 0.0
    previous_count = ast_node_count(previous_code)
    candidate_count = ast_node_count(candidate_code)
    return float(abs(candidate_count - previous_count) + 1)


def _emit_event(
    *,
    log_path: Path,
    config: RunConfig,
    run_id: str,
    event_type: str,
    mode: str,
    task: str | None,
    step: int,
    payload: dict[str, object],
) -> None:
    write_event(
        log_path,
        {
            "schema_version": config.event_schema_version,
            "event_type": event_type,
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "mode": mode,
            "task": task,
            "step": step,
            "payload": payload,
        },
    )


def _initialize_run(
    *,
    config: RunConfig,
    output_root: Path,
) -> tuple[str, Path]:
    run_id = datetime.now(UTC).strftime("run-%Y%m%dT%H%M%S%fZ") + f"-{uuid4().hex[:6]}"
    log_path = output_root / "logs" / f"{run_id}.jsonl"
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
    return run_id, log_path


def _mutate_statement_swap(tree: ast.AST, rng: random.Random) -> bool:
    candidates = [
        node
        for node in ast.walk(tree)
        if hasattr(node, "body")
        and isinstance(node.body, list)
        and len(node.body) >= 2
        and all(isinstance(item, ast.stmt) for item in node.body)
    ]
    if not candidates:
        return False
    target = rng.choice(candidates)
    left = rng.choice(range(0, len(target.body) - 1))
    target.body[left], target.body[left + 1] = target.body[left + 1], target.body[left]
    return True


def _mutate_binop(tree: ast.AST, rng: random.Random) -> bool:
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
    ]
    if not candidates:
        return False
    target = rng.choice(candidates)
    target.op = ast.Sub() if rng.random() < 0.5 else ast.Mult()
    return True


_COMPARE_SWAP: dict[type, type] = {
    ast.Lt: ast.Gt,
    ast.Gt: ast.Lt,
    ast.LtE: ast.GtE,
    ast.GtE: ast.LtE,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}


def _mutate_compare(tree: ast.AST, rng: random.Random) -> bool:
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and type(node.ops[0]) in _COMPARE_SWAP
    ]
    if not candidates:
        return False
    target = rng.choice(candidates)
    target.ops[0] = _COMPARE_SWAP[type(target.ops[0])]()
    return True


def _mutate_boolop(tree: ast.AST, rng: random.Random) -> bool:
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or))
    ]
    if not candidates:
        return False
    target = rng.choice(candidates)
    target.op = ast.Or() if isinstance(target.op, ast.And) else ast.And()
    return True


def _mutate_constant(tree: ast.AST, rng: random.Random) -> bool:
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ]
    if not candidates:
        return False
    target = rng.choice(candidates)
    target.value = target.value + rng.choice([-2, -1, 1, 2])
    return True


_MUTATORS = {
    "statement_swap": _mutate_statement_swap,
    "binop": _mutate_binop,
    "compare": _mutate_compare,
    "boolop": _mutate_boolop,
    "constant": _mutate_constant,
}


_SEMANTIC_MUTATORS: list[str] = ["guard_insertion", "loop_conversion", "variable_extraction"]


def _mutate_code(
    current: str,
    rng: random.Random,
    intensity: int = 1,
    prefer_structural: bool = False,
    enable_semantic: bool = False,
) -> str:
    # Try semantic mutation first if enabled (30% chance per intensity round)
    if enable_semantic:
        from alife_core.mutation.semantic import (
            mutate_guard_insertion,
            mutate_loop_conversion,
            mutate_variable_extraction,
        )

        _semantic_dispatch = {
            "guard_insertion": mutate_guard_insertion,
            "loop_conversion": mutate_loop_conversion,
            "variable_extraction": mutate_variable_extraction,
        }
        for _ in range(intensity):
            if rng.random() < 0.3:
                op_name = rng.choice(_SEMANTIC_MUTATORS)
                result = _semantic_dispatch[op_name](current, rng)
                if result != current:
                    return result

    tree = ast.parse(current)
    order = (
        ["statement_swap", "compare", "boolop", "binop", "constant"]
        if prefer_structural
        else ["compare", "boolop", "binop", "statement_swap", "constant"]
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

    run_id, log_path = _initialize_run(config=config, output_root=output_root)
    organisms_dir = output_root / "organisms"

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

        _emit_event(
            log_path=log_path,
            config=config,
            run_id=run_id,
            event_type="step.evaluated",
            mode="single_agent",
            task=task.name,
            step=0,
            payload={
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
                "hard_failure": current_eval.hard_failure,
                "execution_status": current_eval.execution_status,
                "error_type": current_eval.error_type,
                "error_detail": current_eval.error_detail,
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
                enable_semantic=config.enable_semantic_mutation,
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
                _emit_event(
                    log_path=log_path,
                    config=config,
                    run_id=run_id,
                    event_type="mutation.rejected",
                    mode="single_agent",
                    task=task.name,
                    step=step,
                    payload={
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

            _emit_event(
                log_path=log_path,
                config=config,
                run_id=run_id,
                event_type="step.evaluated",
                mode="single_agent",
                task=task.name,
                step=step,
                payload={
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
                    "hard_failure": candidate_eval.hard_failure,
                    "execution_status": candidate_eval.execution_status,
                    "error_type": candidate_eval.error_type,
                    "error_detail": candidate_eval.error_detail,
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

    _emit_event(
        log_path=log_path,
        config=config,
        run_id=run_id,
        event_type="run.completed",
        mode="single_agent",
        task=task_name,
        step=0,
        payload={"completed_tasks": completed_tasks},
    )
    return RunSummary(run_id=run_id, log_path=log_path, completed_tasks=completed_tasks)


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
    organisms: list[OrganismState] | list[str],
    task: TaskSpec,
    config: RunConfig,
) -> list[OrganismState]:
    normalized: list[OrganismState] = []
    if organisms and isinstance(organisms[0], str):
        for idx, code in enumerate(organisms, start=1):
            normalized.append(
                OrganismState(
                    organism_id=f"legacy-{idx}",
                    parent_ids=(),
                    birth_generation=0,
                    code=code,
                    fitness=0.0,
                    train_pass_ratio=0.0,
                    hidden_pass_ratio=0.0,
                    lineage_depth=0,
                    evaluated=False,
                )
            )
    else:
        normalized = list(organisms)  # type: ignore[arg-type]

    def _evaluate_one(organism: OrganismState) -> OrganismState:
        result = evaluate_candidate(organism.code, task=task, edit_cost=0.0, config=config)
        return OrganismState(
            code=organism.code,
            fitness=result.fitness,
            train_pass_ratio=result.train_pass_ratio,
            hidden_pass_ratio=result.hidden_pass_ratio,
            organism_id=organism.organism_id,
            parent_ids=organism.parent_ids,
            birth_generation=organism.birth_generation,
            ast_nodes=ast_node_count(organism.code),
            ast_depth=ast_max_depth(organism.code),
            shape_fingerprint=ast_shape_fingerprint(organism.code),
            lineage_depth=organism.lineage_depth,
            evaluated=True,
        )

    result_population = list(normalized)
    pending: list[tuple[int, OrganismState]] = [
        (idx, organism) for idx, organism in enumerate(normalized) if not organism.evaluated
    ]
    if not pending:
        return result_population

    workers = min(
        max(1, config.population_workers),
        max(1, len(pending)),
        (os.cpu_count() or 1),
    )
    LOGGER.info("Evaluating population with %s workers", workers)
    if workers == 1:
        evaluated_items = [_evaluate_one(organism) for _, organism in pending]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            evaluated_items = list(executor.map(_evaluate_one, [item for _, item in pending]))

    for (idx, _), evaluated in zip(pending, evaluated_items, strict=True):
        result_population[idx] = evaluated
    return result_population


def _initialize_population_for_task(
    task: TaskSpec,
    config: RunConfig,
    rng: random.Random,
    id_counter: list[int],
) -> list[OrganismState]:
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
    while len(population_codes) < config.population_size:
        attempts += 1
        if attempts > config.population_size * 20:
            population_codes.append(bootstrap_code)
            continue
        candidate = _mutate_code(
            bootstrap_code,
            rng,
            intensity=1,
            enable_semantic=config.enable_semantic_mutation,
        )
        if validate_candidate(candidate).is_valid:
            population_codes.append(candidate)
    organisms: list[OrganismState] = []
    for code in population_codes:
        id_counter[0] += 1
        organisms.append(
            OrganismState(
                organism_id=f"org-{id_counter[0]}",
                parent_ids=(),
                birth_generation=0,
                code=code,
                fitness=0.0,
                train_pass_ratio=0.0,
                hidden_pass_ratio=0.0,
                lineage_depth=0,
                evaluated=False,
            )
        )
    return organisms


def run_population_experiment(task_name: str, config: RunConfig, output_root: Path) -> RunSummary:
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
    run_id, log_path = _initialize_run(config=config, output_root=output_root)
    organisms_dir = output_root / "organisms"

    rng = random.Random(config.seed)
    completed_tasks: list[str] = []
    current_task_names = list(tasks)
    current_task_index = current_task_names.index(task_name)
    id_counter = [0]

    while current_task_index < len(current_task_names):
        task = tasks[current_task_names[current_task_index]]
        recent_diversity: deque[float] = deque(maxlen=max(1, config.diversity_window))
        recent_best_fitness: deque[float] = deque(maxlen=max(1, config.convergence_patience))
        seed_population = _initialize_population_for_task(task, config, rng, id_counter)
        population = _evaluate_population(seed_population, task, config)

        unlocked = False
        for generation in range(config.max_generations + 1):
            best = max(population, key=lambda item: item.fitness)
            fitnesses = [item.fitness for item in population]
            metrics = compute_generation_metrics(population, novelty_k=config.novelty_k)
            recent_diversity.append(metrics.structural_diversity_ratio)
            recent_best_fitness.append(best.fitness)

            _emit_event(
                log_path=log_path,
                config=config,
                run_id=run_id,
                event_type="generation.started",
                mode="population",
                task=task.name,
                step=generation,
                payload={"generation": generation, "population_size": len(population)},
            )
            _emit_event(
                log_path=log_path,
                config=config,
                run_id=run_id,
                event_type="generation.metrics",
                mode="population",
                task=task.name,
                step=generation,
                payload={
                    "generation": generation,
                    "best_fitness": best.fitness,
                    "mean_fitness": statistics.fmean(fitnesses),
                    "median_fitness": statistics.median(fitnesses),
                    "best_train_pass_ratio": best.train_pass_ratio,
                    "best_hidden_pass_ratio": best.hidden_pass_ratio,
                    "structural_diversity_ratio": metrics.structural_diversity_ratio,
                    "shannon_entropy": metrics.shannon_entropy,
                    "simpson_diversity_index": metrics.simpson_diversity_index,
                    "cluster_count": metrics.cluster_count,
                    "mean_ast_nodes": metrics.mean_ast_nodes,
                    "median_ast_nodes": metrics.median_ast_nodes,
                    "mean_ast_depth": metrics.mean_ast_depth,
                    "median_ast_depth": metrics.median_ast_depth,
                    "mean_novelty": metrics.mean_novelty,
                    "max_lineage_depth": metrics.max_lineage_depth,
                    "mean_lineage_depth": metrics.mean_lineage_depth,
                    "kolmogorov_complexity_proxy": metrics.kolmogorov_complexity_proxy,
                    "cumulative_complexity_delta": metrics.cumulative_complexity_delta,
                    "code_token_zipf_coefficient": metrics.code_token_zipf_coefficient,
                },
            )
            _emit_event(
                log_path=log_path,
                config=config,
                run_id=run_id,
                event_type="generation.ended",
                mode="population",
                task=task.name,
                step=generation,
                payload={"generation": generation},
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

            low_diversity = len(recent_diversity) >= config.diversity_window and all(
                score <= config.min_diversity_score for score in recent_diversity
            )
            fitness_stalled = (
                len(recent_best_fitness) >= config.convergence_patience
                and (max(recent_best_fitness) - min(recent_best_fitness))
                <= config.convergence_fitness_delta_floor
            )
            low_entropy = metrics.shannon_entropy <= config.convergence_entropy_floor
            if low_diversity or (fitness_stalled and low_entropy):
                _emit_event(
                    log_path=log_path,
                    config=config,
                    run_id=run_id,
                    event_type="evolution.converged",
                    mode="population",
                    task=task.name,
                    step=generation,
                    payload={
                        "generation": generation,
                        "reason": "low_diversity" if low_diversity else "stalled_low_entropy",
                        "diversity_window": list(recent_diversity),
                        "best_fitness_window": list(recent_best_fitness),
                        "shannon_entropy": metrics.shannon_entropy,
                    },
                )
                break

            if generation == config.max_generations:
                break

            sorted_population = sorted(population, key=lambda item: item.fitness, reverse=True)
            elites = sorted_population[: config.elite_count]
            next_organisms: list[OrganismState] = []
            for elite in elites:
                id_counter[0] += 1
                next_organisms.append(
                    OrganismState(
                        organism_id=f"org-{id_counter[0]}",
                        parent_ids=(elite.organism_id,),
                        birth_generation=generation + 1,
                        code=elite.code,
                        fitness=elite.fitness,
                        train_pass_ratio=elite.train_pass_ratio,
                        hidden_pass_ratio=elite.hidden_pass_ratio,
                        ast_nodes=elite.ast_nodes,
                        ast_depth=elite.ast_depth,
                        shape_fingerprint=elite.shape_fingerprint,
                        lineage_depth=elite.lineage_depth + 1,
                        evaluated=True,
                    )
                )

            while len(next_organisms) < config.population_size:
                parent_a = _tournament_select(population, rng, config.tournament_k)
                child_code = parent_a.code
                parent_ids = (parent_a.organism_id,)
                child_lineage_depth = parent_a.lineage_depth + 1
                if rng.random() < config.crossover_rate:
                    parent_b = _tournament_select(population, rng, config.tournament_k)
                    child_code = _crossover_code(parent_a.code, parent_b.code, rng)
                    parent_ids = (parent_a.organism_id, parent_b.organism_id)
                    child_lineage_depth = max(parent_a.lineage_depth, parent_b.lineage_depth) + 1
                    _emit_event(
                        log_path=log_path,
                        config=config,
                        run_id=run_id,
                        event_type="crossover.applied",
                        mode="population",
                        task=task.name,
                        step=generation,
                        payload={"generation": generation, "parent_ids": parent_ids},
                    )
                if rng.random() < config.mutation_rate:
                    child_code = _mutate_code(
                        child_code,
                        rng,
                        intensity=1,
                        enable_semantic=config.enable_semantic_mutation,
                    )

                validation = validate_candidate(child_code)
                if not validation.is_valid:
                    child_code = parent_a.code
                    parent_ids = (parent_a.organism_id,)
                    child_lineage_depth = parent_a.lineage_depth + 1
                id_counter[0] += 1
                next_organisms.append(
                    OrganismState(
                        organism_id=f"org-{id_counter[0]}",
                        parent_ids=parent_ids,
                        birth_generation=generation + 1,
                        code=child_code,
                        fitness=0.0,
                        train_pass_ratio=0.0,
                        hidden_pass_ratio=0.0,
                        lineage_depth=child_lineage_depth,
                        evaluated=False,
                    )
                )

            population = _evaluate_population(next_organisms, task, config)

        if not unlocked or not config.run_curriculum:
            break
        current_task_index += 1

    _emit_event(
        log_path=log_path,
        config=config,
        run_id=run_id,
        event_type="run.completed",
        mode="population",
        task=task_name,
        step=0,
        payload={"completed_tasks": completed_tasks},
    )
    return RunSummary(run_id=run_id, log_path=log_path, completed_tasks=completed_tasks)


def run_experiment(task_name: str, config: RunConfig, output_root: Path) -> RunSummary:
    _validate_run_config(config)
    _ensure_docker_daemon_available(config)
    if config.evolution_mode == "population":
        return run_population_experiment(
            task_name=task_name, config=config, output_root=output_root
        )
    return run_single_agent_experiment(task_name=task_name, config=config, output_root=output_root)
