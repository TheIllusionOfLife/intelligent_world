from alife_core.models import TaskSpec


def load_builtin_tasks() -> dict[str, TaskSpec]:
    return {
        "two_sum_sorted": TaskSpec(
            name="two_sum_sorted",
            prompt="Return 1-based indices of two numbers adding to target in sorted list",
            function_name="two_sum_sorted",
            train_cases=(
                (([2, 7, 11, 15], 9), (1, 2)),
                (([1, 2, 3, 4, 5], 8), (3, 5)),
                (([-3, -1, 0, 4, 7], 3), (2, 4)),
                (([1, 1, 1, 1, 5], 6), (1, 5)),
                (([1, 2], 3), (1, 2)),
                (([-5, -3, 0, 2, 8], -8), (1, 2)),
                (([1, 3, 5, 7, 9, 11], 12), (1, 6)),
                (([0, 0, 3, 4], 0), (1, 2)),
                (([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 19), (9, 10)),
                (([10, 20, 30, 40, 50], 70), (2, 5)),
            ),
            hidden_cases=(
                (([1, 5, 8, 11, 14], 19), (2, 5)),
                (([-10, -5, 0, 5, 10], 0), (1, 5)),
                (([3, 3], 6), (1, 2)),
                (([1, 4, 6, 8, 10, 12], 14), (2, 5)),
                ((list(range(1, 101)), 101), (1, 100)),
            ),
        ),
        "run_length_encode": TaskSpec(
            name="run_length_encode",
            prompt="Compress string using run-length encoding",
            function_name="run_length_encode",
            train_cases=(
                (("aabbbcccc",), "a2b3c4"),
                (("abc",), "abc"),
                (("",), ""),
                (("aaaa",), "a4"),
                (("a",), "a"),
                (("aabb",), "a2b2"),
                (("aaabba",), "a3b2a"),
                (("zzzzzzzzzz",), "z10"),
                (("abababab",), "abababab"),
                (("aaaaabbbbbccccc",), "a5b5c5"),
            ),
            hidden_cases=(
                (("xxyyxxyyxx",), "x2y2x2y2x2"),
                (("aaaaaaaaaaaa",), "a12"),
                (("abcabcabc",), "abcabcabc"),
                (("aaabbbccc",), "a3b3c3"),
                (("z",), "z"),
            ),
        ),
        "slugify": TaskSpec(
            name="slugify",
            prompt="Convert text to URL-safe slug",
            function_name="slugify",
            train_cases=(
                (("Hello World",), "hello-world"),
                (("  Hello   World  ",), "hello-world"),
                (("Python 3.12 Released!",), "python-312-released"),
                (("",), ""),
                (("already-slugified",), "already-slugified"),
                (("UPPERCASE",), "uppercase"),
                (("foo---bar",), "foo-bar"),
                (("...leading and trailing...",), "leading-and-trailing"),
                (("café résumé",), "caf-rsum"),
                (("one",), "one"),
            ),
            hidden_cases=(
                (("  ",), ""),
                (("---",), ""),
                (("Hello, World! How's it going?",), "hello-world-hows-it-going"),
                (("under_score_test",), "under-score-test"),
                (("MiXeD CaSe 123",), "mixed-case-123"),
            ),
        ),
    }
