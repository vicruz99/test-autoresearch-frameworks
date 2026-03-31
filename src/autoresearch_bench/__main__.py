"""CLI entry point: python -m autoresearch_bench run --config <path>."""

import argparse
import asyncio
import sys


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate command."""
    parser = argparse.ArgumentParser(
        prog="autoresearch_bench",
        description="Benchmark LLM-driven automated research strategies.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment from a YAML config.")
    run_parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the YAML experiment configuration file.",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and list what would run, without executing.",
    )

    args = parser.parse_args()

    if args.command == "run":
        from autoresearch_bench.runner import Runner

        runner = Runner.from_config_file(args.config)
        if args.dry_run:
            runner.dry_run()
        else:
            asyncio.run(runner.run())
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
