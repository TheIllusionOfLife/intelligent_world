from alife_core.agent.curriculum import should_unlock_next_task
from alife_core.models import EvaluationResult, RunConfig


def test_curriculum_requires_pass_ratio_and_fitness() -> None:
    config = RunConfig(pass_ratio_threshold=0.9, fitness_threshold=0.7)
    result = EvaluationResult(
        train_pass_ratio=1.0,
        hidden_pass_ratio=0.0,
        ast_edit_cost=0.4,
        fitness=0.69,
        train_failures=0,
        hidden_failures=1,
    )

    assert should_unlock_next_task(result=result, config=config) is False

    result2 = EvaluationResult(
        train_pass_ratio=1.0,
        hidden_pass_ratio=0.0,
        ast_edit_cost=0.1,
        fitness=0.9,
        train_failures=0,
        hidden_failures=1,
    )
    assert should_unlock_next_task(result=result2, config=config) is True


def test_curriculum_rejects_when_pass_ratio_below_threshold() -> None:
    config = RunConfig(pass_ratio_threshold=0.9, fitness_threshold=0.7)
    result = EvaluationResult(
        train_pass_ratio=0.5,
        hidden_pass_ratio=1.0,
        ast_edit_cost=0.1,
        fitness=0.95,
        train_failures=1,
        hidden_failures=0,
    )

    assert should_unlock_next_task(result=result, config=config) is False
