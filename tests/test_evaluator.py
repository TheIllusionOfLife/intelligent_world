from alife_core.evaluator.core import evaluate_candidate
from alife_core.models import RunConfig, TaskSpec


def test_evaluator_is_deterministic_for_same_seed_and_inputs() -> None:
    task = TaskSpec(
        name="increment",
        prompt="return x + 1",
        function_name="solve",
        train_cases=[((1,), 2), ((10,), 11)],
        hidden_cases=[((100,), 101)],
    )
    code = "def solve(x):\n    return x + 1\n"
    config = RunConfig(seed=7)

    first = evaluate_candidate(code=code, task=task, edit_cost=0.0, config=config)
    second = evaluate_candidate(code=code, task=task, edit_cost=0.0, config=config)

    assert first == second


def test_hidden_tests_do_not_change_fitness_in_phase1() -> None:
    task = TaskSpec(
        name="only_train_counts",
        prompt="train pass, hidden fail",
        function_name="solve",
        train_cases=[((2,), 3)],
        hidden_cases=[((2,), 99)],
    )
    code = "def solve(x):\n    return x + 1\n"
    config = RunConfig(seed=1, w1_pass_ratio=1.0, w2_ast_edit_cost=0.0)

    result = evaluate_candidate(code=code, task=task, edit_cost=0.0, config=config)

    assert result.train_pass_ratio == 1.0
    assert result.hidden_pass_ratio == 0.0
    assert result.fitness == 1.0
