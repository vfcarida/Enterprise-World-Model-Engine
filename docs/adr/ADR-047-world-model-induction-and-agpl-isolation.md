# ADR-047: World-Model Induction, Enterprise Schema Mining, and AGPL-3.0 License Isolation

## Context and Problem Statement

To transform Enterprise World Model Engine from a reference toolkit requiring manual specification authoring into an enterprise-adopted standard, users must be able to **point the engine at existing organizational data (DataFrames, SQL databases, event logs) and receive an auditable, replay-validated simulation twin**.

However, automating world model induction introduces two critical technical and governance challenges:
1. **License Contamination (The AGPL-3.0 Trap)**: The leading Python process-mining library, PM4Py ([arXiv:1905.06169](https://arxiv.org/abs/1905.06169)), is licensed under the **GNU Affero General Public License (AGPL-3.0)**. Static-linking or importing PM4Py directly into EWM Engine core would contaminate the permissively licensed Apache-2.0 / MIT codebase and prevent enterprise deployment.
2. **Epistemic Overclaiming**: Empirical data mining often yields spurious correlations or ungrounded transitions. How can EWM induce models while maintaining strict epistemic humility and determinism?

## Considered Options

1. **Mandatory In-Process PM4Py Dependency**: Directly depend on PM4Py in core.
   - *Rejected*: Incompatible with EWM's Apache-2.0/MIT licensing; violates the core invariant of zero mandatory heavy dependencies.
2. **Pure Manual Modeling (Status Quo)**: Require users to hand-write all `WorldSpec` YAML declarations.
   - *Rejected*: Limits adoption; enterprise practitioners require automated ingestion of existing databases and logs.
3. **Native Inductive Miner with Subprocess AGPL Quarantine**:
   - Provide native pure-Python/NumPy Inductive Miner and trace replay conformance checking with zero external dependencies.
   - When PM4Py is optionally requested (`[pm4py]` extra), isolate it strictly out-of-process via an isolated subprocess worker communicating exclusively through JSON pipes.
   - Enforce point-in-time correctness (RelBench discipline) on all tabular/SQL connectors.
   - Distinguish `ASSERTED` schema constraints from `INFERRED-LINK` provisional proposals.

## Decision Outcome

Chosen option: **Option 3 (Native Inductive Miner with Subprocess AGPL Quarantine)**.

### Architectural Rules & Invariants

1. **Strict AGPL Subprocess Boundary**:
   - EWM Engine core NEVER imports `pm4py` in-process.
   - Subprocess bridge (`ewm_engine.induction.pm4py_bridge`) invokes `pm4py_worker.py` in a separate OS process via `subprocess.run()`, exchanging data strictly via JSON stdin/stdout.
   - The default engine is `engine="native"`, requiring only standard Python and NumPy.
2. **Induction is Proposal, Never Assertion**:
   - Relational schemas and declared Foreign Keys are tagged `ASSERTED` (physical ground truth).
   - Mined process transitions carry their alignment fitness and precision scores mapped directly to `EvidenceLevel` (`STRUCTURAL`, `QUASI_CAUSAL`, or `PREDICTIVE`).
   - Latent edges discovered via link prediction are tagged `INFERRED-LINK` and marked `provisional: True` until vetted by validation gates.
3. **Point-in-Time Correctness (No Temporal Leakage)**:
   - Connectors accept an `as_of` epoch timestamp. All records strictly after `as_of` are purged prior to world compilation.
4. **Deterministic Fingerprinting**:
   - Every induction run produces an `InductionReport` with a canonical SHA-256 fingerprint, guaranteeing byte-identical proposals for identical data and seeds.

## Consequences

### Positive
- **Clean License Hygiene**: EWM Engine core remains completely untainted by copyleft AGPL requirements.
- **Enterprise Adoption**: Organizations can generate WorldSpecs directly from OCEL 2.0 logs and SQL databases with a single CLI command (`ewm induce`).
- **Soundness & Conformance**: Process discovery guarantees deadlock-free models with quantitative fitness/precision audits.

### Negative / Trade-offs
- Out-of-process subprocess execution introduces minor IPC serialization overhead (~10–50ms per run) when PM4Py is invoked. Native engine has zero IPC overhead.
