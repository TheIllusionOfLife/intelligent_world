from dataclasses import dataclass
from typing import Literal

Case = tuple[tuple, object]
SandboxBackend = Literal["docker", "process"]
BootstrapBackend = Literal["static", "ollama"]
EvolutionMode = Literal["single_agent", "population"]


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    w1_pass_ratio: float = 0.9
    w2_ast_edit_cost: float = 0.1
    base_survival_cost: float = 0.01
    pass_ratio_threshold: float = 0.9
    fitness_threshold: float = 0.7
    exec_timeout_seconds: float = 1.0
    sandbox_backend: SandboxBackend = "docker"
    docker_image: str = "python:3.12-slim"

    initial_energy: float = 1.0
    max_steps: int = 200
    n_stagnation: int = 100
    improvement_multiplier: float = 1.0
    degradation_multiplier: float = 1.0

    initial_temperature: float = 1.0
    cooling_rate: float = 0.995

    w2_floor: float = 0.02
    decay_factor: float = 0.999

    mutation_stagnation_window: int = 20
    goodhart_gap_threshold: float = 0.2

    bootstrap_backend: BootstrapBackend = "ollama"
    ollama_model: str = "gpt-oss:20b"
    bootstrap_timeout_seconds: float = 20.0
    bootstrap_fallback_to_static: bool = True
    allow_unsafe_process_backend: bool = False

    run_curriculum: bool = False
    viability_window: int = 20
    viability_min_improvement_rate: float = 0.05

    evolution_mode: EvolutionMode = "single_agent"
    population_size: int = 8
    elite_count: int = 2
    tournament_k: int = 3
    crossover_rate: float = 0.7
    mutation_rate: float = 0.9
    max_generations: int = 50
    population_workers: int = 4
    diversity_window: int = 5
    min_diversity_score: float = 0.2


@dataclass(frozen=True)
class TaskSpec:
    name: str
    prompt: str
    function_name: str
    train_cases: tuple[Case, ...] = ()
    hidden_cases: tuple[Case, ...] = ()


@dataclass(frozen=True)
class EvaluationResult:
    train_pass_ratio: float
    hidden_pass_ratio: float
    ast_edit_cost: float
    fitness: float
    train_failures: int
    hidden_failures: int


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    stage: str
    reason: str = ""


@dataclass(frozen=True)
class OrganismState:
    code: str
    fitness: float
    train_pass_ratio: float
    hidden_pass_ratio: float
