# ADR-051: World Model Interchange Format (WMIF), Conformance Suite, and DSL Authoring

## Context and Problem Statement

Interoperability standards in scientific simulation and enterprise modeling frequently suffer from two fatal pitfalls:
1. **Ambiguous Prose Specifications**: Standards defined purely by narrative text lead to diverging interpretations across vendors. Without an automated, executable test suite, implementers diverge, as observed in early FMI cross-check efforts which ossified through manual, non-continuous testing.
2. **Subjective Numerical Tolerances**: Traditional continuous simulators use floating-point tolerance bands ($\epsilon \le 10^{-4}$), making it impossible to definitively declare whether an implementation is conformant or subtly broken.

Enterprise World Model Engine (EWM) holds a unique architectural advantage: **deterministic logical transitions and canonical SHA-256 fingerprints (Invariant #5)**. This enables what no prior simulation standard could offer:
$$\text{Conformance} \iff \text{actual\_fingerprint} == \text{expected\_fingerprint}$$

To establish EWM as the definitive standard for enterprise digital twins and world models, we must operationalize the principle: **"Own the suite = own the standard."**

## Considered Options

1. **Adopt Existing FMI / PMML / SBML Directly**:
   - Force enterprise socio-technical world models into FMI (ODEs/C-binaries) or SBML (systems biology biochemical networks).
   - *Rejected*: Structural enterprise models require discrete graph states, heterogeneous entities, typed resources, relational networks, and categorical constraints that do not map naturally onto continuous ODE systems without extreme distortion.

2. **Proprietary Single-Vendor Format**:
   - Maintain EWM as a standalone closed specification without containerization or independent test suites.
   - *Rejected*: Fails to drive industry adoption or scientific reproducibility across third-party tools, academic research, and regulatory audits.

3. **Self-Describing Container (WMIF) + Standalone Conformance Suite + Authored DSL**:
   - **World Model Interchange Format (WMIF)**: Model on FMI's self-describing PKZIP archive (`.wmif`) bundling `modelDescription.json`, the deterministic model payload, and embedded reference simulation results.
   - **SBML Levels & Packages**: Modular, versioned growth where Level 1 defines the deterministic core, while optional packages (`PkgCausal`, `PkgCosim`, `PkgDecisions`, `PkgStochastic`) enforce fail-closed requirement gates.
   - **Draft 2020-12 & OpenAPI 3.1 Unification**: Canonical `WorldSpec` schema validates HTTP REST payloads, MCP agent requests, and standalone config files.
   - **MLCommons Croissant 1.0**: Emit JSON-LD dataset manifests per simulation output for direct discovery on HuggingFace and Kaggle.
   - **AsyncAPI 3.0**: Formal streaming specification for long-running Monte Carlo progress, step transitions, and completion notifications.
   - **Standalone Conformance Suite**: Language-agnostic test suite modeled on the JSON Schema Test Suite with independent MIT licensing, CI automation, and negative mutation testing.
   - **WSL DSL Authoring Tooling**: Bounded-time Lark LALR(1) parser + pygls Language Server + VS Code extension client.

## Decision Outcome

Chosen option: **Option 3 (WMIF Container + Standalone Conformance Suite + DSL Authoring)**.

### Architectural Rules & Invariants

1. **WMIF Container Structure**:
   - Standard PKZIP archive with `.wmif` extension.
   - `modelDescription.json`: Root manifest declaring schema version, Level, active packages, model file path, and reference golden runs.
   - Safe unzipping: Verifies absence of path traversal markers (`..`, leading `/`).
   - Round-trip fidelity: `export_wmif` → `import_wmif` → simulation reproduces bit-identical per-branch fingerprints.

2. **SBML-Style Package Requirement Gate**:
   - Models declaring packages with `required: true` must be supported by the runtime engine.
   - If an engine encounters an unrecognized required package, it must fail closed and refuse execution with a descriptive error.

3. **Bit-Exact Conformance Oracle**:
   - The conformance suite asserts exact SHA-256 state and rollout fingerprints.
   - CI executes the suite continuously against the reference engine.
   - Negative mutation testing verifies that perturbed fingerprints fail loudly.

4. **Safe DSL Authoring**:
   - The authored WSL DSL parser utilizes a strict LALR(1) grammar implemented via Lark.
   - Strictly zero code execution (`eval`/`exec` rejected); syntax trees map directly into Pydantic models with `extra="forbid"`.
   - pygls Language Server exposes real-time diagnostics, hover info, and autocompletion using the identical validator deployed in CLI and CI.

5. **Licensing & Standardization Labeling**:
   - WMIF is formally labeled **Proposed / Experimental (`v1.0.0-proposed`)**.
   - The Conformance Suite (`conformance/`) is independently licensed under the MIT License to encourage cross-engine adoption.

## Consequences

### Positive
- **Vendor-Neutral Interoperability**: Any simulation engine can implement WMIF and objectively certify compliance via the Conformance Suite.
- **Unambiguous Correctness**: Binary pass/fail determination replaces vague numerical tolerance thresholds.
- **Developer Experience**: Authoring `.wsl` models in VS Code provides immediate diagnostic feedback and syntax highlighting without requiring Python knowledge.

### Negative
- **Standard Governance Overhead**: Versioning WMIF Levels and maintaining backward compatibility across package releases requires rigorous change-control procedures.
