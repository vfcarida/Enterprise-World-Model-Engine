# ADR-015: Semantic Versioning Stability Policy, Public API Gate, and v1.0.0 Release

## Status
Accepted

## Date
2026-10-02

## Context
With the completion of Milestones M1 through M10, the Enterprise World Model Engine (EWM Engine) has implemented every component required by the authoritative specification (`docs/specs/spec-driven-development.md`): domain-independent core, deeply immutable state snapshots, pluggable dynamics, first-class phase-aware constraints, deterministic Monte Carlo uncertainty quantification, complete provenance and audit fingerprints, inspectable non-causal systemic traces, versioned JSON Schemas, normative acceptance fixtures (Minimal Warehouse and CivicFlow), lifecycle observability hooks, and a 12-gate CI/CD automation pipeline.

Cutting release `v1.0.0` marks a permanent boundary: moving from pre-1.0 rapid refactoring to a strict, long-term stability and backward-compatibility contract for enterprise consumers and downstream researchers.

## Decision

### 1. Semantic Versioning Framework
We strictly adopt [Semantic Versioning (SemVer 2.0.0)](https://semver.org/):
- **Patch Releases (`1.0.x`)**: Bug fixes, performance optimizations, internal refactorings, and security patches that do not alter public API signatures or behavior.
- **Minor Releases (`1.x.0`)**: Backwards-compatible additions to the public API (new optional dynamics, constraints, evaluation metrics, or export formats).
- **Major Releases (`2.0.0+`)**: Reserved strictly for intentional, breaking architectural changes that have passed through the full deprecation lifecycle.

### 2. Normative Stable Surface
The Stable Public API surface for `1.x` is defined exclusively by the 22 canonical symbols exported in `ewm_engine.__all__`:
```
Action, Constraint, ConstraintPhase, ConstraintResult, ConstraintSeverity,
DynamicsModel, Entity, EvidenceLevel, ExogenousEvent, Provenance, Relationship,
Resource, Scenario, SimulationEngine, SimulationResult, TraceEdge, Trajectory,
TrajectoryStatus, TransitionResult, World, WorldState, compare_scenarios,
__version__
```
Any addition, deletion, or renaming requires an approved API Change Proposal.

### 3. Experimental Namespace Isolation
All rapidly evolving research components (such as `Intervention`, `LearnedDynamics`, `LinearResidualDynamics`, and `RecedingHorizonSimulator`) are quarantined in `ewm_engine.experimental.*`. They may iterate across minor releases and graduate to Stable only after formal ADR and schema approval.

### 4. Deprecation Lifecycle
Any planned removal of a Stable capability requires:
1. Emitting a runtime `DeprecationWarning`.
2. Documentation in `CHANGELOG.md` under `### Deprecated`.
3. A mandatory waiting window of at least **one minor release cycle** AND **at least 90 calendar days** (whichever is longer) before removal in `2.0.0+`.

### 5. Automated API Contract Gate (AC-024)
We institute `tests/contract/test_api_compatibility.py` as an automated CI quality gate asserting that:
- `ewm_engine.__all__` matches the committed v1.0.0 surface.
- Core entrypoint parameter signatures do not introduce required parameters without defaults.
- Committed JSON Schemas maintain const `schema_version = "1.0.0"`.

### 6. Formal v1.0.0 Release
Having fulfilled all 24 Acceptance Criteria (AC-001 through AC-024) and verified the complete Definition of Done, the repository version is bumped to `1.0.0`.
