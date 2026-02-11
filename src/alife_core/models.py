from dataclasses import dataclass
from typing import Literal

Case = tuple[tuple, object]
SandboxBackend = Literal["docker", "process"]


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
