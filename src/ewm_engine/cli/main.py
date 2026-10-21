"""Command-line interface for the Enterprise World Model Engine."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

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
    html_trace_path: str | None = None,
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

        if html_trace_path and result.trajectories:
            html_p = Path(html_trace_path)
            html_p.parent.mkdir(parents=True, exist_ok=True)
            html_p.write_text(
                result.trajectories[0].systemic_trace.to_html(title=f"{spec.world.name} Trace"),
                encoding="utf-8",
            )
            sys.stdout.write(f"Interactive trace visualizer successfully written to: {html_p}\n")

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


def _import_sd(model_path_str: str, out_path_str: str | None = None) -> int:
    """Import a System Dynamics (XMILE / Vensim) model into a WorldSpec YAML template."""
    model_path = Path(model_path_str)
    if not model_path.exists():
        sys.stderr.write(f"Error: System dynamics model file not found: {model_path_str}\n")
        return 1

    try:
        from ewm_engine.cosim.adapters import is_pysd_available

        if not is_pysd_available():
            sys.stderr.write(
                "Error: PySD is required to import System Dynamics models. "
                "Install via `pip install ewm-engine[sd]`.\n"
            )
            return 1

        import yaml

        model_name = model_path.stem
        spec_dict = {
            "schema_version": "1.0.0",
            "world": {
                "name": f"SD_{model_name}",
                "description": f"Imported System Dynamics model from {model_path.name}",
            },
            "entities": [{"id": f"sd_system_{model_name}", "type": "system_dynamics"}],
            "resources": [],
            "constraints": [],
            "dynamics": {
                "type": "pysd_submodel",
                "config": {"model_path": str(model_path.resolve())},
            },
        }

        yaml_content = yaml.dump(spec_dict, sort_keys=False)

        if out_path_str:
            out_p = Path(out_path_str)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(yaml_content, encoding="utf-8")
            sys.stdout.write(f"Successfully generated WorldSpec template at {out_p}\n")
        else:
            sys.stdout.write(yaml_content)
        return 0
    except Exception as err:
        sys.stderr.write(f"Error importing SD model: {err}\n")
        return 1


def _run_decide(
    spec_path_str: str,
    candidates_path_str: str | None = None,
    uncertainty_path_str: str | None = None,
    objective_metric: str = "unserved_demand",
    satisficing_threshold: float = 0.0,
    higher_is_better: bool = False,
    horizon: int = 5,
    samples: int = 40,
    seed: int = 42,
    algorithm: Literal["prim", "cart"] = "prim",
    include_efe: bool = False,
    efe_weight: float = 0.3,
    output_path: str | None = None,
    markdown_path: str | None = None,
) -> int:
    """Execute decision-focused scenario discovery and multi-robustness analysis."""
    import yaml

    from ewm_engine.core.actions import Action, Intervention
    from ewm_engine.decision.engine import DecisionEngine
    from ewm_engine.experimentation.params import ParameterDef, ParameterSpace

    spec_path = Path(spec_path_str)
    if not spec_path.exists():
        sys.stderr.write(f"Error: Specification file not found: {spec_path_str}\n")
        return 1

    try:
        content = spec_path.read_text(encoding="utf-8")
        if spec_path.suffix.lower() == ".json":
            spec = WorldSpec.from_json(content)
        else:
            spec = WorldSpec.from_yaml(content)

        factory = WorldFactory()
        world = factory.create_world(spec)

        # 1. Parse or build Uncertainty Space
        if uncertainty_path_str:
            unc_path = Path(uncertainty_path_str)
            if not unc_path.exists():
                sys.stderr.write(
                    f"Error: Uncertainty space file not found: {uncertainty_path_str}\n"
                )
                return 1
            unc_raw = yaml.safe_load(unc_path.read_text(encoding="utf-8"))
            if isinstance(unc_raw, dict) and "parameters" in unc_raw:
                uncertainty_space = ParameterSpace(**unc_raw)
            else:
                sys.stderr.write(
                    "Error: Invalid uncertainty space format; expected 'parameters' list.\n"
                )
                return 1
        else:
            # Default uncertainty space: +/- 40% variation on numeric resources
            param_defs: list[ParameterDef] = []
            for res in spec.resources:
                curr = float(res.current)
                low = max(float(res.min_value), curr * 0.5)
                high = min(float(res.max_value), max(curr * 1.5, curr + 10.0))
                if high > low:
                    param_defs.append(
                        ParameterDef(
                            name=f"{res.id}_initial",
                            type="continuous",
                            bounds=(low, high),
                            default=curr,
                            target="resource",
                            target_id=res.id,
                            description=f"Uncertain initial level for resource {res.id}",
                        )
                    )
            if not param_defs:
                param_defs.append(
                    ParameterDef(
                        name="dummy_uncertainty",
                        type="continuous",
                        bounds=(0.0, 1.0),
                        default=0.5,
                        target="context",
                        target_id="uncertainty_factor",
                    )
                )
            uncertainty_space = ParameterSpace(
                name=f"{spec.world.name}_UncertaintySpace",
                parameters=tuple(param_defs),
            )

        # 2. Parse or build Candidate Decisions
        candidates: dict[str, Action | Intervention | Sequence[Action] | None] = {}
        if candidates_path_str:
            cand_path = Path(candidates_path_str)
            if not cand_path.exists():
                sys.stderr.write(f"Error: Candidates file not found: {candidates_path_str}\n")
                return 1
            cand_raw = yaml.safe_load(cand_path.read_text(encoding="utf-8"))
            if isinstance(cand_raw, dict):
                for c_id, c_val in cand_raw.items():
                    if c_val is None:
                        candidates[c_id] = None
                    elif isinstance(c_val, dict):
                        if "intervention" in c_val:
                            candidates[c_id] = Intervention(**c_val["intervention"])
                        elif "parameters" in c_val or "type" in c_val:
                            candidates[c_id] = Action(
                                id=c_val.get("id", c_id),
                                type=c_val.get("type", "generic"),
                                parameters=c_val.get("parameters", {}),
                            )
                        else:
                            candidates[c_id] = None
        else:
            # Default candidates: Status Quo + transfer if multiple resources exist
            candidates["status_quo"] = None
            if len(spec.resources) >= 2:
                r1 = spec.resources[0].id
                r2 = spec.resources[1].id
                candidates["policy_transfer_moderate"] = Action(
                    id="act_transfer_mod",
                    type="transfer_resource",
                    parameters={"source_resource": r1, "target_resource": r2, "quantity": 15.0},
                )
                candidates["policy_transfer_aggressive"] = Action(
                    id="act_transfer_agg",
                    type="transfer_resource",
                    parameters={"source_resource": r1, "target_resource": r2, "quantity": 30.0},
                )

        engine = DecisionEngine(discovery_algorithm=algorithm)
        sys.stdout.write(
            f"Running decision-focused scenario discovery ('{spec.world.name}', samples={samples}, seed={seed})...\n"
        )
        sys.stdout.flush()

        report = engine.decide(
            world=world,
            uncertainty_space=uncertainty_space,
            candidates=candidates,
            objective_metric=objective_metric,
            satisficing_threshold=satisficing_threshold,
            higher_is_better=higher_is_better,
            horizon=horizon,
            n_samples=samples,
            seed=seed,
            include_efe=include_efe,
            efe_weight=efe_weight,
            report_title=f"Decision Analysis for {spec.world.name}",
        )

        sys.stdout.write("\n" + report.summary_table() + "\n")

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(report.to_json(), encoding="utf-8")
            sys.stdout.write(f"Report JSON written to: {out_p}\n")

        if markdown_path:
            md_p = Path(markdown_path)
            md_p.parent.mkdir(parents=True, exist_ok=True)
            md_p.write_text(report.to_markdown(), encoding="utf-8")
            sys.stdout.write(f"Report Markdown written to: {md_p}\n")

        return 0
    except Exception as err:
        sys.stderr.write(f"Decision execution error: {err}\n")
        return 1


def _run_calibrate(
    spec_path_str: str,
    data_path_str: str,
    priors_path_str: str | None = None,
    resources_str: str | None = None,
    horizon: int = 10,
    samples: int = 100,
    posterior_samples: int = 500,
    seed: int = 42,
    summary_mode: Literal["trajectory", "moments"] = "trajectory",
    output_path: str | None = None,
    markdown_path: str | None = None,
    allow_uncertified: bool = False,
) -> int:
    """Execute amortized Bayesian calibration with mandatory coverage certification."""
    import csv

    import yaml

    from ewm_engine.calibration.priors import CalibrationPrior, PriorSet
    from ewm_engine.calibration.workflow import CalibrationWorkflow

    spec_path = Path(spec_path_str)
    if not spec_path.exists():
        sys.stderr.write(f"Error: Specification file not found: {spec_path_str}\n")
        return 1

    data_path = Path(data_path_str)
    if not data_path.exists():
        sys.stderr.write(f"Error: Target data file not found: {data_path_str}\n")
        return 1

    try:
        content = spec_path.read_text(encoding="utf-8")
        if spec_path.suffix.lower() == ".json":
            spec = WorldSpec.from_json(content)
        else:
            spec = WorldSpec.from_yaml(content)

        # 1. Parse target data
        target_data: dict[str, list[float]] = {}
        data_text = data_path.read_text(encoding="utf-8")
        if data_path.suffix.lower() == ".csv":
            reader = csv.DictReader(data_text.splitlines())
            for row in reader:
                for k, v in row.items():
                    if k is not None and v is not None:
                        try:
                            val = float(v.strip())
                            target_data.setdefault(k.strip(), []).append(val)
                        except ValueError:
                            pass
        else:
            raw_obj = yaml.safe_load(data_text)
            if isinstance(raw_obj, dict):
                if "observations" in raw_obj and isinstance(raw_obj["observations"], dict):
                    obs_dict = raw_obj["observations"]
                else:
                    obs_dict = raw_obj
                for k, v in obs_dict.items():
                    if isinstance(v, (list, tuple)):
                        target_data[str(k)] = [float(x) for x in v]
            elif isinstance(raw_obj, list):
                for row in raw_obj:
                    if isinstance(row, dict):
                        for k, v in row.items():
                            try:
                                target_data.setdefault(str(k), []).append(float(v))
                            except (ValueError, TypeError):
                                pass

        if not target_data:
            sys.stderr.write(
                f"Error: No valid numeric series found in target data: {data_path_str}\n"
            )
            return 1

        # Determine tracked resources
        if resources_str:
            tracked_resources = [r.strip() for r in resources_str.split(",") if r.strip()]
        else:
            spec_res_ids = {r.id for r in spec.resources}
            matched = [k for k in target_data.keys() if k in spec_res_ids]
            tracked_resources = matched if matched else list(target_data.keys())

        # 2. Build or load priors
        if priors_path_str:
            priors_p = Path(priors_path_str)
            if not priors_p.exists():
                sys.stderr.write(f"Error: Priors file not found: {priors_path_str}\n")
                return 1
            raw_priors = yaml.safe_load(priors_p.read_text(encoding="utf-8"))
            if isinstance(raw_priors, dict) and "priors" in raw_priors:
                prior_list = [CalibrationPrior(**p) for p in raw_priors["priors"]]
            elif isinstance(raw_priors, list):
                prior_list = [CalibrationPrior(**p) for p in raw_priors]
            else:
                sys.stderr.write(
                    "Error: Priors file must contain a list of priors or 'priors' key.\n"
                )
                return 1
            prior_set = PriorSet(priors=tuple(prior_list))
        else:
            # Default priors: uniform on spec resources
            auto_priors: list[CalibrationPrior] = []
            for r in spec.resources:
                curr = float(r.current)
                low = max(float(r.min_value), curr * 0.2 if curr > 0 else 0.0)
                high = min(float(r.max_value), max(curr * 2.0, curr + 50.0))
                if high > low:
                    auto_priors.append(
                        CalibrationPrior(
                            name=f"{r.id}_initial",
                            target="resource",
                            target_id=r.id,
                            distribution="uniform",
                            bounds=(low, high),
                        )
                    )
            if not auto_priors:
                auto_priors.append(
                    CalibrationPrior(
                        name="dummy_param",
                        target="resource",
                        target_id=spec.resources[0].id if spec.resources else "res1",
                        distribution="uniform",
                        bounds=(0.0, 100.0),
                    )
                )
            prior_set = PriorSet(priors=tuple(auto_priors))

        max_target_len = max((len(v) for v in target_data.values()), default=11)
        sim_horizon = horizon if horizon != 10 else max(1, max_target_len - 1)

        scenario = Scenario(
            scenario_id=f"calib_scen_{spec.world.name}",
            horizon=sim_horizon,
            samples=1,
            seed=seed,
        )

        workflow = CalibrationWorkflow(
            spec=spec,
            prior_set=prior_set,
            tracked_resources=tracked_resources,
            scenario=scenario,
            summary_mode=summary_mode,
        )

        sys.stdout.write(
            f"Running amortized calibration for '{spec.world.name}' (simulations={samples}, seed={seed})...\n"
        )
        sys.stdout.flush()

        report = workflow.run_calibration(
            target_data=target_data,
            n_simulations=samples,
            n_posterior_samples=posterior_samples,
            seed=seed,
        )

        sys.stdout.write("\n" + report.to_console_table() + "\n")

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(report.to_json(), encoding="utf-8")
            sys.stdout.write(f"Calibration JSON artifact written to: {out_p}\n")

        if markdown_path:
            md_p = Path(markdown_path)
            md_p.parent.mkdir(parents=True, exist_ok=True)
            md_p.write_text(report.to_markdown(), encoding="utf-8")
            sys.stdout.write(f"Calibration Markdown artifact written to: {md_p}\n")

        if not report.certificate.is_certified:
            sys.stderr.write(
                f"\n[CALIBRATION REFUSED FOR DECISION USE]\n"
                f"Reason: {report.certificate.refusal_reason}\n"
                f"Status: {report.certificate.status}\n"
            )
            if not allow_uncertified:
                return 1

        return 0
    except Exception as err:
        sys.stderr.write(f"Calibration execution error: {err}\n")
        return 1


def _run_mcp(
    provenance_strict: bool = False,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
) -> int:
    """Launch the Model Context Protocol (MCP) server for agentic integrations."""
    from ewm_engine.mcp.server import create_mcp_server, is_mcp_available

    if not is_mcp_available():
        sys.stderr.write(
            "Error: The Model Context Protocol SDK is required to run the MCP server.\n"
            "Install via `pip install ewm-engine[mcp]`.\n"
        )
        return 1

    server = create_mcp_server(provenance_strict=provenance_strict)
    if transport == "stdio":
        server.run_stdio()
        return 0
    elif transport == "sse":
        if hasattr(server.server, "run_sse_async"):
            import asyncio

            asyncio.run(server.server.run_sse_async(host=host, port=port))
            return 0
        elif hasattr(server.server, "run_sse"):
            server.server.run_sse(host=host, port=port)
            return 0
        else:
            sys.stderr.write("Error: SSE transport is not supported by the installed MCP SDK.\n")
            return 1
    else:
        sys.stderr.write(f"Unknown transport: {transport}. Use 'stdio' or 'sse'.\n")
        return 1


def _run_induce(
    target_type: str,
    data_path: str | None = None,
    db_path: str | None = None,
    ddl_str: str | None = None,
    ocel_path: str | None = None,
    trajectories_path: str | None = None,
    interventions_path: str | None = None,
    spec_path: str | None = None,
    held_out_path: str | None = None,
    strict: bool = False,
    engine: str = "native",
    as_of: str | None = None,
    id_col: str | None = None,
    type_col: str | None = None,
    ts_col: str | None = None,
    link_prediction: bool = True,
    threshold: float = 0.7,
    alpha: float = 0.05,
    min_support: float = 0.5,
    degree: int = 2,
    tolerance: float = 1e-3,
    mc_samples: int = 10,
    name: str | None = None,
    seed: int = 42,
    output_path: str | None = None,
    report_path: str | None = None,
    report_out_path: str | None = None,
    markdown_out_path: str | None = None,
) -> int:
    """Execute world-model induction from DataFrame, SQL schema, OCEL 2.0 log, causal discovery, or dynamics/invariants."""
    from ewm_engine.core.spec import (
        ComponentSpec,
        EntitySpec,
        RelationshipSpec,
        ResourceSpec,
        WorldMetadataSpec,
        WorldSpec,
    )
    from ewm_engine.induction.causal import induce_causal
    from ewm_engine.induction.connectors import DataFrameSchemaMapping, induce_from_dataframe
    from ewm_engine.induction.dynamics import induce_dynamics
    from ewm_engine.induction.invariants import induce_invariants
    from ewm_engine.induction.process import induce_process
    from ewm_engine.induction.report import InductionReport
    from ewm_engine.induction.schema import induce_schema
    from ewm_engine.induction.validation import InductionValidationGate, validate_induction
    from ewm_engine.provenance.evidence import EvidenceLevel
    from ewm_engine.serialization.yaml import safe_dump_yaml

    def _load_data_file(p_str: str) -> list[dict[str, Any]]:
        p = Path(p_str)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p_str}")
        if p.suffix.lower() == ".csv":
            import csv

            with open(p, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                return list(reader)
        elif p.suffix.lower() in (".json", ".jsonl"):
            content = p.read_text(encoding="utf-8")
            if p.suffix.lower() == ".jsonl":
                return [json.loads(line) for line in content.splitlines() if line.strip()]
            else:
                loaded = json.loads(content)
                return loaded if isinstance(loaded, list) else [loaded]
        elif p.suffix.lower() == ".parquet":
            try:
                import pyarrow.parquet as pq

                table = pq.read_table(p)
                return list(table.to_pylist())
            except ImportError as err:
                raise ImportError("pyarrow is required to read Parquet files.") from err
        else:
            raise ValueError(
                f"Unsupported file format: {p.suffix}. Use .csv, .json, .jsonl, or .parquet."
            )

    try:
        if target_type == "schema":
            if not db_path and not ddl_str:
                sys.stderr.write("Error: Must provide --db or --ddl for schema induction.\n")
                return 1
            spec, report = induce_schema(
                db=db_path,
                ddl=ddl_str,
                link_prediction=link_prediction,
                link_threshold=threshold,
                name=name or "InducedSchemaWorld",
                seed=seed,
            )
        elif target_type == "process":
            if not ocel_path:
                sys.stderr.write("Error: Must provide --ocel path for process induction.\n")
                return 1
            spec, report = induce_process(
                ocel_data=Path(ocel_path),
                engine="pm4py" if engine == "pm4py" else "native",
                name=name or "InducedProcessWorld",
                seed=seed,
            )
        elif target_type == "dataframe":
            if not data_path:
                sys.stderr.write("Error: Must provide --data path for dataframe induction.\n")
                return 1
            df_data = _load_data_file(data_path)
            mapping = DataFrameSchemaMapping(
                entity_id_col=id_col,
                entity_type_col=type_col,
                timestamp_col=ts_col,
            )
            spec, report = induce_from_dataframe(
                df=df_data,
                schema=mapping,
                as_of=as_of,
                name=name or "InducedDataFrameWorld",
                seed=seed,
            )
        elif target_type == "causal":
            if not data_path:
                sys.stderr.write("Error: Must provide --data path for causal induction.\n")
                return 1
            c_data = _load_data_file(data_path)
            interv_data = None
            if interventions_path:
                interv_p = Path(interventions_path)
                if interv_p.exists():
                    interv_data = json.loads(interv_p.read_text(encoding="utf-8"))

            equiv_class, report = induce_causal(
                data=c_data,
                interventional_data=interv_data,
                alpha=alpha,
                min_support=min_support,
                asserted_threshold=threshold,
                seed=seed,
            )
            gate = InductionValidationGate(min_causal_support=threshold, fail_closed=True)
            admitted_edges, refused_edges, open_qs, _ = gate.validate_causal(
                equiv_class=equiv_class,
                data=c_data,
                seed=seed,
            )
            entities = [
                EntitySpec(id=v, type="variable", attributes={"domain": "continuous"})
                for v in equiv_class.variables
            ]
            relationships = [
                RelationshipSpec(
                    source=e.source,
                    target=e.target,
                    type="directed_dependency",
                    attributes={
                        "posterior_support": e.posterior_support,
                        "evidence_level": EvidenceLevel.ASSERTED.value,
                        "edge_type": e.edge_type.value,
                    },
                )
                for e in admitted_edges
            ]
            spec = WorldSpec(
                world=WorldMetadataSpec(name=name or "InducedCausalWorld"),
                entities=entities,
                relationships=relationships,
            )
            report.validation_passed = len(refused_edges) == 0
            report.refused_elements_count = len(refused_edges)
            report.open_questions_count = len(open_qs)
            report.refused_elements = [r.model_dump(mode="json") for r in refused_edges]
            report.open_questions = [q.model_dump(mode="json") for q in open_qs]
            report.sha256_fingerprint = report.compute_fingerprint()
        elif target_type == "dynamics":
            traj_p = trajectories_path or data_path
            if not traj_p:
                sys.stderr.write(
                    "Error: Must provide --trajectories or --data path for dynamics induction.\n"
                )
                return 1
            traj_data = _load_data_file(traj_p)
            rate_laws, report = induce_dynamics(
                trajectories_or_dataset=traj_data,
                degree=degree,
                threshold=threshold,
                seed=seed,
            )
            entities = [
                EntitySpec(
                    id=r.target_variable,
                    type="resource_entity",
                    attributes={"domain": "continuous"},
                )
                for r in rate_laws
            ]
            resources = [
                ResourceSpec(
                    id=r.target_variable,
                    current=100.0,
                    min_value=0.0,
                    max_value=1e9,
                    unit="units",
                )
                for r in rate_laws
            ]
            spec = WorldSpec(
                world=WorldMetadataSpec(name=name or "InducedDynamicsWorld"),
                entities=entities,
                resources=resources,
                dynamics=ComponentSpec(
                    type="DeterministicDynamics",
                    id="InducedDeterministicDynamics",
                ),
            )
        elif target_type == "invariants":
            traj_p = trajectories_path or data_path
            if not traj_p:
                sys.stderr.write(
                    "Error: Must provide --trajectories or --data path for invariant induction.\n"
                )
                return 1
            traj_data = _load_data_file(traj_p)
            hypotheses, report = induce_invariants(
                trajectories_or_dataset=traj_data,
                tolerance=tolerance,
                mc_samples=mc_samples,
                seed=seed,
            )
            var_names = list(traj_data[0].keys()) if traj_data else []
            entities = [
                EntitySpec(id=v, type="resource_entity", attributes={"domain": "continuous"})
                for v in var_names
            ]
            resources = [
                ResourceSpec(
                    id=v,
                    current=float(traj_data[0].get(v, 100.0)) if traj_data else 100.0,
                    min_value=0.0,
                    max_value=1e9,
                    unit="units",
                )
                for v in var_names
            ]
            gate = InductionValidationGate(invariant_tolerance=tolerance, fail_closed=True)
            admitted_hypo, refused_invs, open_qs, _ = gate.validate_invariants(hypotheses)
            constraints = [h.to_constraint_spec() for h in admitted_hypo]
            spec = WorldSpec(
                world=WorldMetadataSpec(name=name or "InducedInvariantWorld"),
                entities=entities,
                resources=resources,
                constraints=constraints,
            )
            report.validation_passed = len(refused_invs) == 0
            report.refused_elements_count = len(refused_invs)
            report.open_questions_count = len(open_qs)
            report.refused_elements = [r.model_dump(mode="json") for r in refused_invs]
            report.open_questions = [q.model_dump(mode="json") for q in open_qs]
            report.sha256_fingerprint = report.compute_fingerprint()
        elif target_type == "validate":
            rep_file = report_path
            spec_file = spec_path
            if not rep_file and not spec_file:
                sys.stderr.write("Error: Must provide --report or --spec for validation.\n")
                return 1

            loaded_report = None
            if rep_file:
                rp = Path(rep_file)
                if not rp.exists():
                    sys.stderr.write(f"Error: Report file not found: {rep_file}\n")
                    return 1
                loaded_report = InductionReport.model_validate_json(rp.read_text(encoding="utf-8"))

            loaded_spec = None
            if spec_file:
                loaded_spec = WorldSpec.from_yaml(spec_file)

            c_data = None
            if data_path:
                c_data = _load_data_file(data_path)

            held_out = None
            if held_out_path:
                held_out = _load_data_file(held_out_path)

            target_spec_or_report: WorldSpec | InductionReport | None = (
                loaded_report if loaded_report is not None else loaded_spec
            )
            assert target_spec_or_report is not None
            spec, val_report = validate_induction(
                spec_or_report=target_spec_or_report,
                data=c_data,
                held_out_trajectories=held_out,
                report=loaded_report,
                strict=strict,
                seed=seed,
            )
            report = loaded_report or InductionReport(
                source=spec.world.name,
                source_type="causal",
                seed=seed,
                validation_passed=val_report.passed,
                refused_elements_count=val_report.refused_elements_count,
                open_questions_count=len(val_report.open_questions),
                refused_elements=[r.model_dump(mode="json") for r in val_report.refused_elements],
                open_questions=[q.model_dump(mode="json") for q in val_report.open_questions],
            )

            if markdown_out_path:
                md_p = Path(markdown_out_path)
                md_p.parent.mkdir(parents=True, exist_ok=True)
                md_p.write_text(val_report.to_markdown(), encoding="utf-8")
                sys.stdout.write(f"Validation Markdown audit written to: {markdown_out_path}\n")

            if report_out_path:
                ro_p = Path(report_out_path)
                ro_p.parent.mkdir(parents=True, exist_ok=True)
                ro_p.write_text(val_report.model_dump_json(indent=2), encoding="utf-8")
                sys.stdout.write(f"Validation report JSON written to: {report_out_path}\n")
        else:
            sys.stderr.write(
                f"Unknown induction target: {target_type}. Use 'schema', 'process', 'dataframe', 'causal', 'dynamics', 'invariants', or 'validate'.\n"
            )
            return 1

        # Format spec as YAML
        yaml_content = safe_dump_yaml(spec.model_dump(mode="json"))

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(yaml_content, encoding="utf-8")
            sys.stdout.write(f"World specification written to: {output_path}\n")
        else:
            sys.stdout.write(yaml_content)

        if report_path:
            rep_p = Path(report_path)
            rep_p.parent.mkdir(parents=True, exist_ok=True)
            rep_p.write_text(report.to_json(), encoding="utf-8")
            sys.stdout.write(f"Induction report written to: {report_path}\n")

        sys.stdout.write(
            f"\nInduction Successful:\n"
            f"  Entities: {report.entities_count}\n"
            f"  Relationships: {report.relationships_count}\n"
            f"  Resources: {report.resources_count}\n"
            f"  Fingerprint: {report.sha256_fingerprint}\n"
        )
        return 0
    except Exception as exc:
        sys.stderr.write(f"Induction failed: {exc}\n")
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
    run_parser.add_argument(
        "--html-trace",
        type=str,
        default=None,
        help="Optional output HTML path for interactive systemic trace visualizer",
    )

    # 4. `ewm decide <spec_file> [options]`
    decide_parser = subparsers.add_parser(
        "decide", help="Decision-focused scenario discovery and multi-robustness analysis"
    )
    decide_parser.add_argument("spec_file", help="Path to YAML or JSON world specification file")
    decide_parser.add_argument(
        "--candidates", type=str, default=None, help="Optional YAML path for candidate strategies"
    )
    decide_parser.add_argument(
        "--uncertainty",
        type=str,
        default=None,
        help="Optional YAML path for parameter uncertainty space",
    )
    decide_parser.add_argument(
        "--objective",
        type=str,
        default="unserved_demand",
        help="Target performance metric (default: unserved_demand)",
    )
    decide_parser.add_argument(
        "--threshold", type=float, default=0.0, help="Satisficing threshold value (default: 0.0)"
    )
    decide_parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Set if higher metric represents better outcome",
    )
    decide_parser.add_argument(
        "--horizon", type=int, default=5, help="Simulation time horizon (default: 5)"
    )
    decide_parser.add_argument(
        "--samples",
        type=int,
        default=40,
        help="Number of uncertainty scenarios to sample (default: 40)",
    )
    decide_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    decide_parser.add_argument(
        "--algorithm",
        choices=["prim", "cart"],
        default="prim",
        help="Discovery algorithm (prim or cart)",
    )
    decide_parser.add_argument(
        "--efe",
        action="store_true",
        help="Include active inference Expected Free Energy (EFE) scoring",
    )
    decide_parser.add_argument(
        "--efe-weight",
        type=float,
        default=0.3,
        help="Weight for epistemic information gain in EFE (default: 0.3)",
    )
    decide_parser.add_argument(
        "--out", type=str, default=None, help="Optional output JSON path for decision report"
    )
    decide_parser.add_argument(
        "--markdown-out",
        type=str,
        default=None,
        help="Optional output Markdown path for decision report",
    )

    # 5. `ewm schema [list|show <name>]`
    schema_parser = subparsers.add_parser("schema", help="Inspect committed JSON Schemas")
    schema_parser.add_argument("action", choices=["list", "show"], help="Action: 'list' or 'show'")
    schema_parser.add_argument(
        "name", nargs="?", default=None, help="Schema name to show (when action is 'show')"
    )

    # 6. `ewm import-sd <model_file> [--out <out_file>]`
    import_sd_parser = subparsers.add_parser(
        "import-sd", help="Import a System Dynamics XMILE model into WorldSpec format"
    )
    import_sd_parser.add_argument(
        "model_file", help="Path to .xmile or .mdl system dynamics model file"
    )
    import_sd_parser.add_argument(
        "--out", type=str, default=None, help="Optional output YAML path for generated spec"
    )

    # 7. `ewm calibrate <spec_file> --data <data_file> [options]`
    calib_parser = subparsers.add_parser(
        "calibrate", help="Amortized Bayesian calibration with certified coverage (SBI)"
    )
    calib_parser.add_argument("spec_file", help="Path to YAML or JSON world specification file")
    calib_parser.add_argument(
        "--data",
        required=True,
        type=str,
        help="Path to target observation data (YAML, JSON, or CSV)",
    )
    calib_parser.add_argument(
        "--priors",
        type=str,
        default=None,
        help="Optional YAML path for parameter prior distributions",
    )
    calib_parser.add_argument(
        "--resources",
        type=str,
        default=None,
        help="Optional comma-separated list of tracked resources",
    )
    calib_parser.add_argument(
        "--horizon", type=int, default=10, help="Simulation horizon (default: 10)"
    )
    calib_parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of forward simulation samples for amortizer (default: 100)",
    )
    calib_parser.add_argument(
        "--posterior-samples",
        type=int,
        default=500,
        help="Number of posterior draws (default: 500)",
    )
    calib_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    calib_parser.add_argument(
        "--summary-mode",
        choices=["trajectory", "moments"],
        default="trajectory",
        help="Summary statistic mode",
    )
    calib_parser.add_argument(
        "--out", type=str, default=None, help="Optional output JSON path for calibration report"
    )
    calib_parser.add_argument(
        "--markdown-out",
        type=str,
        default=None,
        help="Optional output Markdown path for calibration report",
    )
    calib_parser.add_argument(
        "--allow-uncertified",
        action="store_true",
        help="Do not exit with error code if coverage test fails",
    )

    # 8. `ewm mcp [--provenance-strict] [--transport stdio|sse] [--host HOST] [--port PORT]`
    mcp_parser = subparsers.add_parser(
        "mcp", help="Run the Model Context Protocol (MCP) server for agentic hosts"
    )
    mcp_parser.add_argument(
        "--provenance-strict",
        action="store_true",
        help="Enforce strict causal provenance and default LLM specs to low evidence level",
    )
    mcp_parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol to use (default: stdio)",
    )
    mcp_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for SSE transport (default: 127.0.0.1)",
    )
    mcp_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for SSE transport (default: 8000)",
    )

    # 9. `ewm induce <schema|process|dataframe> [options]`
    induce_parser = subparsers.add_parser(
        "induce",
        help="Propose a validated WorldSpec from relational schema, event log, or tabular data",
    )
    induce_subparsers = induce_parser.add_subparsers(dest="target", help="Induction target type")

    # 9a. Schema
    ind_schema = induce_subparsers.add_parser(
        "schema", help="Induce typed-graph skeleton from relational database or DDL"
    )
    ind_schema.add_argument(
        "--db", type=str, default=None, help="SQLite database path or connection string"
    )
    ind_schema.add_argument(
        "--ddl", type=str, default=None, help="Path to SQL DDL file or inline DDL string"
    )
    ind_schema.add_argument(
        "--link-prediction",
        action="store_true",
        default=True,
        help="Propose latent edges via relational link discovery",
    )
    ind_schema.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for provisional links (default: 0.7)",
    )
    ind_schema.add_argument(
        "--name", type=str, default="InducedSchemaWorld", help="World specification name"
    )
    ind_schema.add_argument(
        "--seed", type=int, default=42, help="Deterministic random seed (default: 42)"
    )
    ind_schema.add_argument("--out", type=str, default=None, help="Output YAML file for WorldSpec")
    ind_schema.add_argument(
        "--report", type=str, default=None, help="Output JSON file for InductionReport"
    )

    # 9b. Process
    ind_proc = induce_subparsers.add_parser(
        "process", help="Induce process model and invariants from OCEL 2.0 event log"
    )
    ind_proc.add_argument(
        "--ocel", type=str, required=True, help="Path to OCEL 2.0 JSON or JSONL file"
    )
    ind_proc.add_argument(
        "--engine",
        choices=["native", "pm4py"],
        default="native",
        help="Induction miner engine (default: native)",
    )
    ind_proc.add_argument(
        "--name", type=str, default="InducedProcessWorld", help="World specification name"
    )
    ind_proc.add_argument(
        "--seed", type=int, default=42, help="Deterministic random seed (default: 42)"
    )
    ind_proc.add_argument("--out", type=str, default=None, help="Output YAML file for WorldSpec")
    ind_proc.add_argument(
        "--report", type=str, default=None, help="Output JSON file for InductionReport"
    )

    # 9c. Dataframe
    ind_df = induce_subparsers.add_parser(
        "dataframe", help="Induce WorldSpec from tabular CSV, Parquet, or JSON"
    )
    ind_df.add_argument(
        "--data", type=str, required=True, help="Path to CSV, JSON, JSONL, or Parquet file"
    )
    ind_df.add_argument(
        "--as-of", type=str, default=None, help="Point-in-time timestamp/step cutoff"
    )
    ind_df.add_argument(
        "--id-col", type=str, default=None, help="Column identifying entity instances"
    )
    ind_df.add_argument(
        "--type-col", type=str, default=None, help="Column identifying entity types"
    )
    ind_df.add_argument("--ts-col", type=str, default=None, help="Column identifying timestamps")
    ind_df.add_argument(
        "--name", type=str, default="InducedDataFrameWorld", help="World specification name"
    )
    ind_df.add_argument(
        "--seed", type=int, default=42, help="Deterministic random seed (default: 42)"
    )
    ind_df.add_argument("--out", type=str, default=None, help="Output YAML file for WorldSpec")
    ind_df.add_argument(
        "--report", type=str, default=None, help="Output JSON file for InductionReport"
    )

    # 9d. Causal structure induction
    ind_causal = induce_subparsers.add_parser(
        "causal",
        help="Induce causal Markov equivalence class (CPDAG/PAG) from observational data",
    )
    ind_causal.add_argument(
        "--data", type=str, required=True, help="Path to CSV, JSON, JSONL, or Parquet data file"
    )
    ind_causal.add_argument(
        "--interventions",
        type=str,
        default=None,
        help="Optional path to JSON file with experimental A/B test / intervention data",
    )
    ind_causal.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for independence tests (default: 0.05)",
    )
    ind_causal.add_argument(
        "--min-support",
        type=float,
        default=0.5,
        help="Minimum ensemble posterior support (default: 0.5)",
    )
    ind_causal.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Posterior threshold for ASSERTED status (default: 0.7)",
    )
    ind_causal.add_argument(
        "--name", type=str, default="InducedCausalWorld", help="World specification name"
    )
    ind_causal.add_argument(
        "--seed", type=int, default=42, help="Deterministic random seed (default: 42)"
    )
    ind_causal.add_argument("--out", type=str, default=None, help="Output YAML file for WorldSpec")
    ind_causal.add_argument(
        "--report", type=str, default=None, help="Output JSON file for InductionReport"
    )

    # 9e. Dynamics induction
    ind_dyn = induce_subparsers.add_parser(
        "dynamics",
        help="Induce governing rate laws via SINDy and symbolic regression",
    )
    ind_dyn.add_argument(
        "--trajectories",
        "--data",
        dest="trajectories",
        type=str,
        required=True,
        help="Path to CSV, JSON, JSONL, or Parquet state transition trajectories file",
    )
    ind_dyn.add_argument(
        "--degree",
        type=int,
        default=2,
        help="Maximum polynomial degree for SINDy feature library (default: 2)",
    )
    ind_dyn.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="SINDy sparse regression threshold (default: 0.05)",
    )
    ind_dyn.add_argument(
        "--name", type=str, default="InducedDynamicsWorld", help="World specification name"
    )
    ind_dyn.add_argument(
        "--seed", type=int, default=42, help="Deterministic random seed (default: 42)"
    )
    ind_dyn.add_argument("--out", type=str, default=None, help="Output YAML file for WorldSpec")
    ind_dyn.add_argument(
        "--report", type=str, default=None, help="Output JSON file for InductionReport"
    )

    # 9f. Invariant / conservation induction
    ind_inv = induce_subparsers.add_parser(
        "invariants",
        help="Induce conservation laws via AI-Poincaré nullspace and Monte Carlo stress testing",
    )
    ind_inv.add_argument(
        "--trajectories",
        "--data",
        dest="trajectories",
        type=str,
        required=True,
        help="Path to CSV, JSON, JSONL, or Parquet state transition trajectories file",
    )
    ind_inv.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Numerical tolerance for hard constraint promotion (default: 1e-3)",
    )
    ind_inv.add_argument(
        "--mc-samples",
        type=int,
        default=10,
        help="Number of Monte Carlo stress testing perturbation rollouts (default: 10)",
    )
    ind_inv.add_argument(
        "--name", type=str, default="InducedInvariantWorld", help="World specification name"
    )
    ind_inv.add_argument(
        "--seed", type=int, default=42, help="Deterministic random seed (default: 42)"
    )
    ind_inv.add_argument("--out", type=str, default=None, help="Output YAML file for WorldSpec")
    ind_inv.add_argument(
        "--report", type=str, default=None, help="Output JSON file for InductionReport"
    )

    # 9g. Honesty and validation gate
    ind_val = induce_subparsers.add_parser(
        "validate",
        help="Validate induced world specifications and reports (the honesty gate)",
    )
    ind_val.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to JSON InductionReport to validate",
    )
    ind_val.add_argument(
        "--spec",
        type=str,
        default=None,
        help="Path to YAML WorldSpec to validate",
    )
    ind_val.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional path to observational/interventional dataset for DoWhy refutations",
    )
    ind_val.add_argument(
        "--held-out",
        type=str,
        default=None,
        help="Optional path to held-out trajectory dataset for dynamics validation",
    )
    ind_val.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Fail closed: raise an error if any proposed structure is refused",
    )
    ind_val.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42)",
    )
    ind_val.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output YAML file for validated WorldSpec",
    )
    ind_val.add_argument(
        "--report-out",
        type=str,
        default=None,
        help="Output JSON file for InductionValidationReport",
    )
    ind_val.add_argument(
        "--markdown-out",
        type=str,
        default=None,
        help="Output Markdown file for human-readable audit",
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
            html_trace_path=args.html_trace,
        )
    elif args.command == "decide":
        return _run_decide(
            args.spec_file,
            candidates_path_str=args.candidates,
            uncertainty_path_str=args.uncertainty,
            objective_metric=args.objective,
            satisficing_threshold=args.threshold,
            higher_is_better=args.higher_is_better,
            horizon=args.horizon,
            samples=args.samples,
            seed=args.seed,
            algorithm=args.algorithm,
            include_efe=args.efe,
            efe_weight=args.efe_weight,
            output_path=args.out,
            markdown_path=args.markdown_out,
        )
    elif args.command == "calibrate":
        return _run_calibrate(
            args.spec_file,
            data_path_str=args.data,
            priors_path_str=args.priors,
            resources_str=args.resources,
            horizon=args.horizon,
            samples=args.samples,
            posterior_samples=args.posterior_samples,
            seed=args.seed,
            summary_mode=args.summary_mode,
            output_path=args.out,
            markdown_path=args.markdown_out,
            allow_uncertified=args.allow_uncertified,
        )
    elif args.command == "schema":
        return _handle_schema(args.action, args.name)
    elif args.command == "import-sd":
        return _import_sd(args.model_file, out_path_str=args.out)
    elif args.command == "mcp":
        return _run_mcp(
            provenance_strict=args.provenance_strict,
            transport=args.transport,
            host=args.host,
            port=args.port,
        )
    elif args.command == "induce":
        target = getattr(args, "target", None)
        if not target:
            induce_parser.print_help()
            return 1
        return _run_induce(
            target_type=target,
            data_path=getattr(args, "data", None),
            db_path=getattr(args, "db", None),
            ddl_str=getattr(args, "ddl", None),
            ocel_path=getattr(args, "ocel", None),
            trajectories_path=getattr(args, "trajectories", None),
            interventions_path=getattr(args, "interventions", None),
            spec_path=getattr(args, "spec", None),
            held_out_path=getattr(args, "held_out", None),
            strict=getattr(args, "strict", False),
            engine=getattr(args, "engine", "native"),
            as_of=getattr(args, "as_of", None),
            id_col=getattr(args, "id_col", None),
            type_col=getattr(args, "type_col", None),
            ts_col=getattr(args, "ts_col", None),
            link_prediction=getattr(args, "link_prediction", True),
            threshold=getattr(args, "threshold", 0.7),
            alpha=getattr(args, "alpha", 0.05),
            min_support=getattr(args, "min_support", 0.5),
            degree=getattr(args, "degree", 2),
            tolerance=getattr(args, "tolerance", 1e-3),
            mc_samples=getattr(args, "mc_samples", 10),
            name=getattr(args, "name", None),
            seed=getattr(args, "seed", 42),
            output_path=getattr(args, "out", None),
            report_path=getattr(args, "report", None),
            report_out_path=getattr(args, "report_out", None),
            markdown_out_path=getattr(args, "markdown_out", None),
        )

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
