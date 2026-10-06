# Command-Line Interface (CLI) Reference

The `ewm` CLI provides command-line utilities for running reference simulation examples, validating declarative specification files, executing simulations from YAML/JSON specs, inspecting schemas, and importing external models.

---

## Command Invocation

```bash
# Invocation via the installed console script
ewm [subcommand] [options]

# Or invocation via python module
python -m ewm_engine.cli.main [subcommand] [options]
```

---

## Subcommands

### `ewm example`

Run a reference simulation example directly from the command line:

```bash
ewm example <example_name>
```

Available examples:
- `civicflow`: Flagship flood disaster emergency response and resource allocation simulation.
- `minimal` / `minimal_warehouse`: Two-warehouse inventory transfer with capacity constraints.
- `minimal_world`: Minimal multi-echelon supply network simulation.
- `agent_eval`: Multi-actor decision policy evaluation.

### `ewm validate`

Validate a declarative WorldSpec or parameter specification file against JSON Schemas:

```bash
ewm validate <spec_file> [--schema <schema_name>]
```

### `ewm run`

Simulate an enterprise world directly from a declarative specification:

```bash
ewm run <spec_file> [--horizon <int>] [--samples <int>] [--seed <int>] [--out <output_json>]
```

### `ewm schema`

Inspect committed Draft 2020-12 JSON Schemas:

```bash
ewm schema [schema_name]
```

Available schemas: `world-spec`, `parameter-space`, `report-model`.

### `ewm import-sd`

Import an external System Dynamics model in XMILE format into an EWM declarative specification:

```bash
ewm import-sd <model.xmile> [--out <output_spec.yaml>]
```
