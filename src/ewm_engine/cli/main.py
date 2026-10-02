"""Minimal command-line interface for the Enterprise World Model Engine."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import ewm_engine


def main(argv: Sequence[str] | None = None) -> int:
    """Main CLI entrypoint for `ewm` command."""
    parser = argparse.ArgumentParser(
        prog="ewm",
        description="Enterprise World Model Engine (EWM Engine) CLI",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ewm-engine {ewm_engine.__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # `ewm example [minimal|civicflow]`
    example_parser = subparsers.add_parser("example", help="Run a reference simulation example")
    example_parser.add_argument(
        "name",
        choices=["minimal", "civicflow"],
        help="Name of example to run ('minimal' or 'civicflow')",
    )

    args = parser.parse_args(argv)

    if args.command == "example":
        if args.name == "minimal":
            from examples.minimal_world.run import run_minimal_world

            print("Running Minimal Supply World Example...\n")
            run_minimal_world()
            return 0
        elif args.name == "civicflow":
            from examples.civicflow.run import run_civicflow_simulation

            print("Running CivicFlow Flood Response Example...\n")
            run_civicflow_simulation()
            return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
