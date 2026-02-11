from alife_core.models import TaskSpec


def load_builtin_tasks() -> dict[str, TaskSpec]:
    return {
        "two_sum_sorted": TaskSpec(
            name="two_sum_sorted",
            prompt="Return indices of two numbers adding to target in sorted list",
            function_name="solve",
            train_cases=[((([2, 7, 11, 15], 9)), (0, 1))],
            hidden_cases=[((([1, 2, 4, 8], 12)), (2, 3))],
        ),
        "run_length_encode": TaskSpec(
            name="run_length_encode",
            prompt="Compress string using run-length encoding",
            function_name="solve",
            train_cases=[(("aabbb",), "a2b3")],
            hidden_cases=[(("abc",), "abc")],
        ),
        "slugify": TaskSpec(
            name="slugify",
            prompt="Normalize text into URL slug",
            function_name="solve",
            train_cases=[(("Hello World",), "hello-world")],
            hidden_cases=[(("under_score_test",), "under-score-test")],
        ),
    }
