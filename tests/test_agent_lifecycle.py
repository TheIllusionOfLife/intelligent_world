from alife_core.agent.lifecycle import AgentState, apply_step_outcome
from alife_core.models import EvaluationResult, RunConfig


def test_rejected_mutation_does_not_consume_energy_or_stagnation() -> None:
    state = AgentState(energy=1.0, stagnation_steps=5, best_fitness=0.7)
    config = RunConfig(base_survival_cost=0.01)

    next_state = apply_step_outcome(
        state=state,
        accepted=False,
        evaluation=None,
        mutation_rejected=True,
        config=config,
    )

    assert next_state.energy == state.energy
    assert next_state.stagnation_steps == state.stagnation_steps


def test_accepted_step_updates_energy_and_best_fitness() -> None:
    state = AgentState(energy=1.0, stagnation_steps=0, best_fitness=0.1)
    config = RunConfig(base_survival_cost=0.01)
    evaluation = EvaluationResult(
        train_pass_ratio=1.0,
        hidden_pass_ratio=1.0,
        ast_edit_cost=0.1,
        fitness=0.8,
        train_failures=0,
        hidden_failures=0,
    )

    next_state = apply_step_outcome(
        state=state,
        accepted=True,
        evaluation=evaluation,
        mutation_rejected=False,
        config=config,
    )

    assert next_state.energy == 1.79
    assert next_state.best_fitness == 0.8
