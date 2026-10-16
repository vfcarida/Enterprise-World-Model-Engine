# World Model Interchange Format (WMIF) Specification Version 1.0

> **Status**: Proposed / Experimental (`v1.0.0-proposed`).
> **Inspiration**: FMI 3.0 / Reference-FMUs self-describing ZIP containers + SBML Levels and Packages.
> **Conformance Bar**: Bit-exact canonical SHA-256 fingerprint verification against embedded golden simulation results.

---

## 1. Motivation & Architecture

Complex organizational simulations are fragmented across disparate modeling formats (AnyLogic, Simio, NetLogo, Stella, custom Python). Even when models share common assumptions, numerical divergence across simulation engines prevents trustworthy cross-validation.

The **World Model Interchange Format (WMIF)** provides an open, vendor-neutral, self-describing container format for sharing and executing enterprise world models.

### Key Innovations
1. **Self-Describing Container**: A standard PKZIP archive (`.wmif`) bundling model topologies, resources, constraints, and **reference simulation results**.
2. **Bit-Exact Conformance Oracle**: Rather than relying on fuzzy tolerance bands ($\epsilon \le 10^{-4}$), WMIF embeds golden reference trajectories with **exact SHA-256 fingerprints**. An engine conforms to a WMIF model if and only if it produces identical fingerprints for identical seeds.
3. **SBML-Style Levels & Packages**: Modular, independently versioned growth where core Level 1 simulations remain backward-compatible, while specialized extensions (`PkgCausal`, `PkgCosim`) declare explicit requirement gates.

---

## 2. Container Layout

A `.wmif` file is a ZIP archive containing:

```text
my_enterprise_model.wmif
├── modelDescription.json             # Root manifest conforming to Draft 2020-12 schema
├── model.wsl                         # Deterministic model definition payload (or model.yaml)
├── provenance.json                   # Machine, author, and toolchain provenance block
└── reference_results/                # Golden simulation results keyed by seed & fingerprint
    └── scen_baseline_7a811e8a2a76.json
```

---

## 3. Manifest (`modelDescription.json`)

The manifest defines:
- **`schema_version`**: `"1.0.0"` (Draft 2020-12 JSON Schema).
- **`format`**: `"WMIF/1.0"`.
- **`level`**: Integer specification level (default `1`).
- **`status`**: `"Proposed/Experimental"`.
- **`model_id`**, **`model_name`**, **`description`**, **`author`**.
- **`model_file`**: Relative path to payload (e.g., `"model.wsl"`).
- **`packages`**: Array of declared SBML-style extension packages.
- **`reference_results`**: Array of golden simulation runs with golden SHA-256 fingerprints.

---

## 4. SBML Levels & Extension Packages

| Package | Name | Status | Purpose |
|:---|:---|:---:|:---|
| **Core** | `Level 1` | Standard | Entities, resources, relationships, constraints, and deterministic dynamics |
| **Causal** | `PkgCausal` | Optional | Causal DAG structures, do-operator interventions, and structural equations |
| **Co-Sim** | `PkgCosim` | Optional | Co-simulation ports, continuous clock coupling, and FMI FMU bridges |
| **Decisions** | `PkgDecisions` | Optional | Multi-objective Pareto optimization spaces and scenario discovery bounds |
| **Stochastic** | `PkgStochastic` | Optional | Seed sequence PRNG trees and parametric noise distributions |

### Fail-Closed Requirement Rule
If a model declares a package with `required: true` and the executing runtime does not implement that package, the engine **must fail closed** and refuse execution with a descriptive configuration error.

---

## 5. Python API Reference

```python
from ewm_engine.serialization.wmif import export_wmif, import_wmif, verify_wmif_conformance

# 1. Export world into .wmif container with golden reference scenarios
export_wmif(
    world=my_world,
    output_path="dist/logistics_v1.wmif",
    scenarios=[baseline_scenario, stress_scenario],
    model_name="Global Logistics Network",
)

# 2. Import and reconstitute world
imported_world, manifest = import_wmif("dist/logistics_v1.wmif")

# 3. Verify conformance
conformance_report = verify_wmif_conformance("dist/logistics_v1.wmif")
assert conformance_report["passed"] is True
```
