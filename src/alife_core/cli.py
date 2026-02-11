import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alife")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--task", required=True)
    run_parser.add_argument("--config", default="configs/default.yaml")
    run_parser.add_argument("--seed", type=int, default=0)

    spike_parser = subparsers.add_parser("spike")
    spike_subparsers = spike_parser.add_subparsers(dest="spike_command", required=True)
    spike_subparsers.add_parser("docker-latency")
    spike_subparsers.add_parser("ast-feasibility")

    return parser


def _dispatch(_args: argparse.Namespace) -> int:
    # Command handlers are intentionally thin in Phase 1 scaffolding.
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return _dispatch(args)


if __name__ == "__main__":
    raise SystemExit(main())
