# Command-Line Interface (CLI) Reference

The `ewm` command-line interface provides deterministic utilities for running reference examples, validating declarative `.yaml`/`.json`/`.wsl` specifications, executing Monte Carlo simulations, discovering decision vulnerabilities (`ewm decide`), calibrating posterior parameters with TARP coverage diagnostics (`ewm calibrate`), inspecting Draft 2020-12 JSON Schemas (`ewm schema`), importing System Dynamics models (`ewm import-sd`), launching the Model Context Protocol server (`ewm mcp`), and inducing world specifications from data (`ewm induce`).

---

## Command Invocation

```bash
# Invocation via the installed console script
ewm <subcommand> [options]

# Or invocation via the Python module
python -m ewm_engine.cli.main <subcommand> [options]
```

---

## Subcommands

### 1. `ewm example`

Run a built-in reference simulation example directly from the command line:

```bash
ewm example [civicflow|minimal]
```

Available examples:
- `civicflow` *(default)*: Flood disaster emergency response and capacity-aware resource allocation simulation (`examples/civicflow_disaster_response.py`).
- `minimal`: Two-warehouse deterministic inventory transfer with pre-action and post-transition constraints (`examples/minimal_warehouse.py`).

### 2. `ewm validate`

Validate a declarative `WorldSpec` file (`.yaml`, `.yml`, `.json`, or `.wsl`) against Pydantic contracts, JSON Schema Draft 2020-12, and the closed `WorldFactory` component registry:

```bash
ewm validate <spec_file>
```

### 3. `ewm run`

Execute a deterministic Monte Carlo simulation from a declarative `WorldSpec` file:

```bash
ewm run <spec_file> \
  [--horizon <int>] \
  [--samples <int>] \
  [--seed <int>] \
  [--out <output.json>] \
  [--html-trace <trace.html>]
```

- `--horizon`: Override simulation step horizon.
- `--samples`: Override number of Monte Carlo trajectory replicas.
- `--seed`: Master integer seed spawned via `np.random.SeedSequence`.
- `--out`: Path to write canonical JSON simulation results (`scenario`, `provenance`, `run_metrics`, `trajectories`).
- `--html-trace`: Path to write a standalone interactive HTML causal trace visualizer.

### 4. `ewm decide`

Run decision-focused scenario discovery (PRIM bump-hunting or CART) and multi-metric robustness analysis (`minimax_regret`, `cvar_regret`, `satisficing_rate`, `domain_criterion`, and optional Active Inference Expected Free Energy):

```bash
ewm decide <spec_file> \
  [--candidates <candidates.yaml>] \
  [--uncertainty <parameter_space.yaml>] \
  [--objective <metric_name>] \
  [--threshold <float>] \
  [--higher-is-better] \
  [--horizon <int>] \
  [--samples <int>] \
  [--seed <int>] \
  [--algorithm prim|cart] \
  [--efe] \
  [--efe-weight <float>] \
  [--out <report.json>] \
  [--markdown-out <report.md>]
```

### 5. `ewm calibrate`

Run amortized Simulation-Based Inference (SBI) parameter calibration and verify posterior credibility via Tests of Accuracy with Random Points (TARP) and Simulation-Based Calibration (SBC):

```bash
ewm calibrate <spec_file> \
  --data <observed_timeseries.csv> \
  [--priors <priors.yaml>] \
  [--resources <res_a,res_b>] \
  [--horizon <int>] \
  [--samples <int>] \
  [--posterior-samples <int>] \
  [--seed <int>] \
  [--summary-mode standard|moments] \
  [--allow-uncertified] \
  [--out <calibration_report.json>] \
  [--markdown-out <calibration_report.md>]
```

By default, `ewm calibrate` fails closed (exit code `1`) if the posterior fails TARP coverage or misspecification gates unless `--allow-uncertified` is explicitly passed.

### 6. `ewm schema`

List or print the 14 canonical Draft 2020-12 JSON Schemas (`schema_version = "1.0.0"`), working both from a repository checkout and from an installed wheel distribution:

```bash
# List all 14 registered JSON Schemas
ewm schema list

# Print a specific JSON Schema to stdout
ewm schema show <schema_name>
```

Available schema names: `action`, `dataset-card`, `model-card`, `parameter-space`, `property-spec`, `provenance`, `report-model`, `scenario-card`, `scenario`, `trajectory`, `transition-event`, `wmif-manifest`, `world-state`, `world`.

### 7. `ewm import-sd`

Import an external System Dynamics model (`.xmile` / `.mdl`) into a declarative `WorldSpec` YAML template (requires `pip install ewm-engine[sd]`):

```bash
ewm import-sd <model_file> [--out <output_spec.yaml>]
```

### 8. `ewm mcp`

Launch the Model Context Protocol (MCP) server exposing deterministic world loading, counterfactual simulation, and epistemic checks to AI agent hosts (requires `pip install ewm-engine[mcp]`):

```bash
ewm mcp [--provenance-strict] [--transport stdio|sse] [--host 127.0.0.1] [--port 8000]
```

### 9. `ewm induce`

Induce candidate `WorldSpec` structures, causal equivalence classes, rate laws, or conservation invariants from empirical telemetry, and audit them through the fail-closed honesty gate (`InductionValidationGate`):

```bash
# 9a. Induce typed-graph skeleton from SQLite DB or SQL DDL
ewm induce schema [--db <path.db> | --ddl <schema.sql>] [--threshold 0.7] [--seed 42] [--out <spec.yaml>] [--report <report.json>]

# 9b. Induce process model & invariants from OCEL 2.0 event log
ewm induce process --ocel <events.json> [--engine native|pm4py] [--seed 42] [--out <spec.yaml>] [--report <report.json>]

# 9c. Induce WorldSpec from point-in-time tabular DataFrame (CSV/JSON/JSONL/Parquet)
ewm induce dataframe --data <table.csv> [--as-of <timestamp>] [--id-col <col>] [--type-col <col>] [--ts-col <col>] [--out <spec.yaml>]

# 9d. Induce Markov equivalence class (CPDAG/PAG) from observational/interventional data
ewm induce causal --data <obs.csv> [--interventions <ab.json>] [--alpha 0.05] [--min-support 0.5] [--threshold 0.7] [--out <spec.yaml>]

# 9e. Induce governing rate laws via SINDy sparse regression
ewm induce dynamics --trajectories <traj.csv> [--degree 2] [--threshold 0.05] [--out <spec.yaml>] [--report <report.json>]

# 9f. Induce linear conservation invariants via nullspace SVD & Monte Carlo stress testing
ewm induce invariants --trajectories <traj.csv> [--tolerance 1e-3] [--mc-samples 10] [--out <spec.yaml>] [--report <report.json>]

# 9g. Validate an induced WorldSpec / InductionReport through the fail-closed honesty gate
ewm induce validate [--report <report.json>] [--spec <spec.yaml>] [--data <obs.csv>] [--held-out <test.csv>] [--strict] [--report-out <val.json>] [--markdown-out <val.md>]
```
