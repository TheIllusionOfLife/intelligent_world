from dataclasses import dataclass, replace

from alife_core.models import EvaluationResult, RunConfig


@dataclass(frozen=True)
class AgentState:
    energy: float
    stagnation_steps: int
    best_fitness: float


def apply_step_outcome(
    state: AgentState,
    accepted: bool,
    previous_fitness: float,
    evaluation: EvaluationResult | None,
    mutation_rejected: bool,
    config: RunConfig,
) -> AgentState:
    if mutation_rejected:
        return state

    next_energy = state.energy - config.base_survival_cost
    next_stagnation = state.stagnation_steps + 1
    next_best = state.best_fitness

    if accepted and evaluation is not None:
        delta = evaluation.fitness - previous_fitness
        if delta >= 0:
            next_energy += config.improvement_multiplier * delta
        else:
            next_energy -= config.degradation_multiplier * abs(delta)

        if evaluation.fitness > state.best_fitness:
            next_best = evaluation.fitness
            next_stagnation = 0

    return replace(
        state,
        energy=round(next_energy, 10),
        stagnation_steps=next_stagnation,
        best_fitness=next_best,
    )
