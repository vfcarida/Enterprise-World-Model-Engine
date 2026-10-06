# Committed JSON Schemas Reference

EWM Engine validates declarative configurations and reporting models against formal **Draft 2020-12 JSON Schemas** committed directly in the repository under [`schemas/`](https://github.com/vfcarida/Enterprise-World-Model-Engine/tree/main/schemas).

---

## 1. World Specification Schema (`world-spec.schema.json`)

Governs declarative world definitions in YAML or JSON format, specifying entities, initial resources, relational graphs, constraints, and pluggable dynamics.

- **Schema URI**: `https://raw.githubusercontent.com/vfcarida/Enterprise-World-Model-Engine/main/schemas/world-spec.schema.json`
- **Specification Standard**: JSON Schema Draft 2020-12
- **Key Fields**:
  - `version`: Version identifier string (e.g., `"1.0.0"`).
  - `state`: Initial state definitions containing `entities`, `resources`, `relationships`, `memory`, and `context`.
  - `dynamics`: Declared transition model name or composite model sequence.
  - `constraints`: Declared operational constraints partitioned into `pre_action` and `post_transition`.

---

## 2. Parameter Space Schema (`parameter-space.schema.json`)

Governs parameter space declarations for Design of Experiments (DoE), parameter sweeps, sensitivity analysis, and optimization.

- **Schema URI**: `https://raw.githubusercontent.com/vfcarida/Enterprise-World-Model-Engine/main/schemas/parameter-space.schema.json`
- **Key Fields**:
  - `parameters`: Array of `ParameterDef` objects specifying `name`, `type` (`continuous`, `integer`, `categorical`), `bounds`, and optional `distribution`.

---

## 3. Report Model Schema (`report-model.schema.json`)

Governs renderer-neutral serialized simulation summaries, quantile distributions, and branch DAGs.

- **Schema URI**: `https://raw.githubusercontent.com/vfcarida/Enterprise-World-Model-Engine/main/schemas/report-model.schema.json`
- **Key Fields**:
  - `run_id`: Unique simulation run UUID or identifier.
  - `fingerprint`: Cryptographic SHA-256 fingerprint of the simulation.
  - `quantiles`: Empirical signal quantiles ($p_{10}, p_{25}, p_{50}, p_{75}, p_{90}, \mu$).
  - `violations`: Summary counts and audit records of constraint breaches.
