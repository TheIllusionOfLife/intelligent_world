from alife_core.tasks.builtin import load_builtin_tasks


def test_builtin_tasks_include_full_phase1_case_sets() -> None:
    tasks = load_builtin_tasks()

    assert set(tasks) == {"two_sum_sorted", "run_length_encode", "slugify"}
    assert len(tasks["two_sum_sorted"].train_cases) >= 10
    assert len(tasks["two_sum_sorted"].hidden_cases) >= 5
    assert len(tasks["run_length_encode"].train_cases) >= 10
    assert len(tasks["run_length_encode"].hidden_cases) >= 5
    assert len(tasks["slugify"].train_cases) >= 10
    assert len(tasks["slugify"].hidden_cases) >= 5
