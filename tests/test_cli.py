from alife_core.cli import build_parser


def test_cli_supports_run_and_spike_commands() -> None:
    parser = build_parser()

    run_args = parser.parse_args(["run", "--task", "two_sum_sorted", "--seed", "7"])
    assert run_args.command == "run"
    assert run_args.task == "two_sum_sorted"
    assert run_args.seed == 7

    spike_args = parser.parse_args(["spike", "docker-latency"])
    assert spike_args.command == "spike"
    assert spike_args.spike_command == "docker-latency"
