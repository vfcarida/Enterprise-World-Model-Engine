"""Command-line interface for the Enterprise World Model Engine."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import ewm_engine
from ewm_engine.core.spec import WorldFactory, WorldSpec
from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import TrajectoryStatus


def _run_example(name: str) -> int:
    """Execute a reference demonstration example."""
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    cwd = Path.cwd()
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))

    if name in ("minimal", "minimal_warehouse"):
        from examples.minimal_warehouse.run import run_minimal_warehouse

        sys.stdout.write("Running Minimal Warehouse Example...\n\n")
        sys.stdout.flush()
        run_minimal_warehouse()
        return 0
    elif name == "minimal_world":
        from examples.minimal_world.run import run_minimal_world

        sys.stdout.write("Running Minimal Supply World Example...\n\n")
        sys.stdout.flush()
        run_minimal_world()
        return 0
    elif name == "civicflow":
        from examples.civicflow.run import run_civicflow_simulation

        sys.stdout.write("Running CivicFlow Flood Response Example...\n\n")
        sys.stdout.flush()
        run_civicflow_simulation()
        return 0
    else:
        sys.stderr.write(f"Unknown example: {name}\n")
        return 1


def _validate_spec(path_str: str) -> int:
    """Safely validate a declarative WorldSpec without executing arbitrary code."""
    path = Path(path_str)
    if not path.exists():
        sys.stderr.write(f"Error: Specification file not found: {path_str}\n")
        return 1

    try:
        content = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            spec = WorldSpec.from_json(content)
        else:
            spec = WorldSpec.from_yaml(content)

        # Validate that factory can resolve all registered components
        factory = WorldFactory()
        _ = factory.create_world(spec)

        dynamics_name = spec.dynamics.type if spec.dynamics else "None"
        sys.stdout.write(f"[VALID] World specification is valid: {path.name}\n")
        sys.stdout.write(f"  Name:          {spec.world.name}\n")
        sys.stdout.write(f"  Entities:      {len(spec.entities)}\n")
        sys.stdout.write(f"  Relationships: {len(spec.relationships)}\n")
        sys.stdout.write(f"  Resources:     {len(spec.resources)}\n")
        sys.stdout.write(f"  Constraints:   {len(spec.constraints)}\n")
        sys.stdout.write(f"  Dynamics:      {dynamics_name}\n")
        sys.stdout.write(f"  Event Sources: {len(spec.event_sources)}\n")
        return 0
    except (SimulationConfigurationError, Exception) as err:
        sys.stderr.write(f"[INVALID] Specification error in {path.name}: {err}\n")
        return 1


def _run_spec(
    path_str: str,
    horizon: int = 10,
    samples: int = 1,
    seed: int = 42,
    output_path: str | None = None,
) -> int:
    """Run simulation directly from a declarative specification."""
    path = Path(path_str)
    if not path.exists():
        sys.stderr.write(f"Error: Specification file not found: {path_str}\n")
        return 1

    try:
        content = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            spec = WorldSpec.from_json(content)
        else:
            spec = WorldSpec.from_yaml(content)

        factory = WorldFactory()
        world = factory.create_world(spec)
        scenario = Scenario(
            name=f"{spec.world.name}_Run",
            horizon=horizon,
            samples=samples,
            seed=seed,
        )

        sys.stdout.write(
            f"Simulating '{spec.world.name}' (horizon={horizon}, samples={samples}, seed={seed})...\n"
        )
        result = world.simulate(scenario=scenario)

        completed = sum(1 for t in result.trajectories if t.status == TrajectoryStatus.COMPLETED)
        invalid = sum(1 for t in result.trajectories if t.status == TrajectoryStatus.INVALID)
        failed = sum(1 for t in result.trajectories if t.status == TrajectoryStatus.FAILED)
        duration = result.provenance.runtime_metadata.get("run_duration_seconds", 0.0)

        sys.stdout.write("\nSimulation Run Complete:\n")
        sys.stdout.write(f"  Trajectories Total:     {len(result.trajectories)}\n")
        sys.stdout.write(f"  Status COMPLETED:       {completed}\n")
        if invalid > 0:
            sys.stdout.write(f"  Status INVALID (halted):{invalid}\n")
        if failed > 0:
            sys.stdout.write(f"  Status FAILED:          {failed}\n")
        sys.stdout.write(f"  Runtime Duration:       {duration:.4f}s\n")
        sys.stdout.write(
            f"  Scenario Fingerprint:   {result.provenance.scenario_fingerprint[:16]}...\n"
        )

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            results_data = {
                "scenario": result.scenario.model_dump(),
                "provenance": result.provenance.model_dump(),
                "run_metrics": result.run_metrics.model_dump(),
                "trajectories": [t.model_dump() for t in result.trajectories],
            }
            out_p.write_text(json.dumps(results_data, indent=2, default=str), encoding="utf-8")
            sys.stdout.write(f"\nResults successfully written to: {out_p}\n")

        return 0
    except (SimulationConfigurationError, Exception) as err:
        sys.stderr.write(f"Simulation execution error: {err}\n")
        return 1


def _handle_schema(action: str, name: str | None = None) -> int:
    """Inspect or display committed JSON schemas."""
    repo_schemas = Path(__file__).resolve().parent.parent.parent.parent / "schemas"
    if not repo_schemas.exists():
        sys.stderr.write("Schemas directory not found in repository root.\n")
        return 1

    schemas = sorted(repo_schemas.glob("*.schema.json"))
    schema_map = {s.name.replace(".schema.json", ""): s for s in schemas}

    if action == "list":
        sys.stdout.write("Available JSON Schemas (Draft 2020-12, schema_version 1.0.0):\n")
        for s_name, s_path in schema_map.items():
            sys.stdout.write(f"  - {s_name} ({s_path.name})\n")
        return 0
    elif action == "show":
        if not name:
            sys.stderr.write("Error: Please provide a schema name to show (e.g. 'world-state').\n")
            return 1
        clean_name = name.replace(".schema.json", "")
        if clean_name not in schema_map:
            sys.stderr.write(
                f"Error: Unknown schema '{name}'. Available: {list(schema_map.keys())}\n"
            )
            return 1
        schema_content = schema_map[clean_name].read_text(encoding="utf-8")
        sys.stdout.write(f"{schema_content}\n")
        return 0
    else:
        sys.stderr.write(f"Unknown schema action: {action}. Use 'list' or 'show'.\n")
        return 1


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

    # 1. `ewm example [minimal|civicflow]`
    example_parser = subparsers.add_parser("example", help="Run a reference simulation example")
    example_parser.add_argument(
        "name",
        choices=["minimal", "minimal_warehouse", "minimal_world", "civicflow"],
        help="Name of example to run ('minimal', 'minimal_warehouse', 'civicflow')",
    )

    # 2. `ewm validate <spec_file>`
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a declarative WorldSpec file"
    )
    validate_parser.add_argument("spec_file", help="Path to YAML or JSON world specification file")

    # 3. `ewm run <spec_file> [options]`
    run_parser = subparsers.add_parser(
        "run", help="Simulate a world directly from a declarative spec"
    )
    run_parser.add_argument("spec_file", help="Path to YAML or JSON world specification file")
    run_parser.add_argument(
        "--horizon", type=int, default=10, help="Simulation time horizon (default: 10)"
    )
    run_parser.add_argument(
        "--samples", type=int, default=1, help="Monte Carlo sample count (default: 1)"
    )
    run_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    run_parser.add_argument(
        "--out", type=str, default=None, help="Optional output JSON path for results"
    )

    # 4. `ewm schema [list|show <name>]`
    schema_parser = subparsers.add_parser("schema", help="Inspect committed JSON Schemas")
    schema_parser.add_argument("action", choices=["list", "show"], help="Action: 'list' or 'show'")
    schema_parser.add_argument(
        "name", nargs="?", default=None, help="Schema name to show (when action is 'show')"
    )

    args = parser.parse_args(argv)

    if args.command == "example":
        return _run_example(args.name)
    elif args.command == "validate":
        return _validate_spec(args.spec_file)
    elif args.command == "run":
        return _run_spec(
            args.spec_file,
            horizon=args.horizon,
            samples=args.samples,
            seed=args.seed,
            output_path=args.out,
        )
    elif args.command == "schema":
        return _handle_schema(args.action, args.name)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
