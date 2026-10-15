# ADR-040: Twin-Rollout Counterfactual SCM Kernel

<!-- This ADR is written according to the MADR 4.x standard -->

* Status: accepted
* Deciders: Vinicius Caridá (@vfcarida)
* Date: 2026-10-15
* Technical Story: S01 — Twin-Rollout Counterfactual SCM Kernel
* Supersedes: None

## Context and Problem Statement

Enterprise decision systems require simulating complex socio-technical and operational dynamics to answer *"what if?"* questions. Under Judea Pearl's Causal Hierarchy:
1. **Rung 1 (Association / Seeing):** Observational predictions $P(Y \mid X)$.
2. **Rung 2 (Intervention / Doing):** Interventional policy rollouts $P(Y \mid do(X))$.
3. **Rung 3 (Counterfactual / Imagining):** Retrospective entity-specific queries $P(Y_{X=x} \mid X=x', Y=y')$.

While standard world models and agent environments support forward interventional branching ($do(X)$ starting from an initial state), they are incapable of answering entity-specific Rung-3 counterfactuals: *"Given that this specific warehouse experienced a stockout at step $t$ under an unexpected demand spike, what would have happened had we rerouted inventory 30 minutes earlier?"*

Standard Monte Carlo branching fails because it samples *new, independent* exogenous noise in the branched world, conflating the effect of the intervention with random noise divergence. Answering true counterfactuals requires:
1. **Abduction:** Inferring the realized exogenous noise variables $U$ that generated the observed factual trajectory.
2. **Action:** Mutilating the structural assignments via causal graph surgery $do(X = x)$.
3. **Prediction:** Re-simulating under the modified assignments with the **exact same coupled noise $U$** (Twin-Rollout coupling).

Furthermore, the scientific license for EWM Engine's architecture is anchored by **Richens & Everitt ("Robust agents learn causal world models", ICLR 2024 oral, arXiv:2402.10877)**, which proves that any agent capable of robustly generalizing across arbitrary distribution shifts must have learned an internal model approximating the true structural causal model of the environment.

How should EWM Engine compile its typed world graph into a first-class Structural Causal Model and deliver honest, seeded, bit-reproducible Pearl Rung-3 counterfactual rollouts?

## Decision Drivers

* **First-Class SCM View Without Mutable Duplication:** The SCM must be a compiled, immutable view over the existing `WorldSpec`, `WorldState`, and `World`, preserving canonical SHA-256 fingerprinting and state immutability.
* **Deterministic Noise Coupling (Twin-Rollout):** Exogenous noise draws must be explicitly recorded, addressable, and coupled bit-for-bit across factual and counterfactual rollouts using the engine's hierarchical `SeedSequence` tree.
* **Invariant 5 (Bit-Level Reproducibility):** Identical counterfactual interventions applied to identical factual trajectories must produce bit-identical final states and traces.
* **Epistemic Honesty and Anti-Overclaiming:** Rung-3 outputs must be strictly scoped to `EvidenceLevel.COUNTERFACTUAL`, carry a mandatory model-counterfactual disclaimer, refuse to auto-upgrade to real-world causal truth without empirical SCM validation (S04), degrade to posterior distributions when abduction is non-unique, and never emit synthetic `"causes"` graph relationships.
* **Zero Heavy Dependencies in Core:** The SCM view, trace provenance, and counterfactual engine must run purely on Python standard library, NumPy, and Pydantic.

## Considered Options

* **Option 1: Status-Quo Scenario Branching (`world.branch()`)** — Clone the world state and run forward rollouts. Insufficient: re-seeds or draws new exogenous noise independently, violating Pearl's abduction requirement and failing Rung-3 causal semantics.
* **Option 2: Heavy Third-Party Causal Framework (DoWhy / Causal-Learn / Pyro)** — Bind the simulator to an external probabilistic programming or causal DAG library. Fails because external libraries do not support EWM's typed graph semantics, temporal conservation constraints, or byte-for-byte `SeedSequence` noise coupling, while introducing heavy PyTorch/scipy dependencies.
* **Option 3: Native SCM View, Addressable Noise Trace, and Pearl Three-Step Twin-Rollout Engine (Chosen)** — Compile `World` into an immutable `StructuralCausalModel` view with explicit structural assignments $V_i := f_i(\text{PA}_i, U_i)$, record addressable noise in `SystemicTrace.exogenous_noise`, and execute native Abduction–Action–Prediction with Twin-Rollout noise coupling.

## Decision Outcome

Chosen option: **Option 3: Native SCM View, Addressable Noise Trace, and Pearl Three-Step Twin-Rollout Engine**.

### Consequences

* **Good, because**:
  * **Native Rung-3 Counterfactuals:** `world.counterfactual(observed, intervention=...)` natively implements Pearl's Abduction–Action–Prediction.
  * **Twin-Rollout Noise Coupling:** Factual and counterfactual trajectories share exogenous noise bit-for-bit, isolating the pure causal effect of the intervention from stochastic variation.
  * **Bit-Reproducible & Auditable:** Preserves canonical SHA-256 fingerprinting for both states and abduced exogenous noise trees.
  * **Epistemic Humility:** Every result carries `EvidenceLevel.COUNTERFACTUAL` and an explicit disclaimer that this is a counterfactual of the model. Non-invertible mechanisms warn and yield posterior distributions rather than false point estimates.
  * **Structural Locality:** Variables outside the causal descendants of the intervened node remain bit-identical between factual and counterfactual worlds.
  * **Systemic Trace Divergence:** `TraceDiff` and `TraceDiff.to_mermaid()` export structured visual flowcharts showing exact divergence and noise invariance.
  * **Zero Dependency Bloat:** Fully implemented within `core` and `simulation` using only NumPy and Pydantic.
* **Bad, because**:
  * Mechanisms with complex non-linear or multi-to-one dynamics cannot be algebraically inverted and must rely on posterior sampling.
  * Callers must explicitly handle `CounterfactualResult` objects and epistemic disclaimers.

### Confirmation

* Unit tests in `tests/unit/test_scm.py` verify:
  1. SCM compilation over `WorldSpec` and `WorldState`.
  2. Structural assignment evaluation and inversion.
  3. Graph surgery $do(X = x)$ preserving non-intervened mechanisms.
  4. Population-level identifiability checks rejecting non-identifiable queries.
* Unit tests in `tests/unit/test_counterfactual.py` verify:
  1. Pearl three-step execution and noise coupling.
  2. Bit-reproducibility of counterfactual trajectories across independent runs.
  3. Structural locality preservation on off-target entities.
  4. Non-unique abduction warnings and posterior distribution sampling.
* Contract tests in `tests/contract/test_scm_counterfactual_contract.py` verify:
  1. Invariant 5 (fingerprint equivalence under coupled noise).
  2. Mandatory model-counterfactual disclaimer presence.
  3. Prohibition of synthetic `"causes"` relationships.
* Worked example in `examples/counterfactual_warehouse_reroute.py` executes end-to-end stockout rerouting counterfactual with Mermaid trace diff export.

## Pros and Cons of the Options

### Option 1: Status-Quo Scenario Branching
* Good: No new code needed.
* Bad: Conflates causal effects with random noise; cannot answer retrospective entity counterfactuals (Rung 3).

### Option 2: Heavy Third-Party Causal Framework
* Good: Leverages off-the-shelf DAG libraries.
* Bad: Incompatible with EWM's typed graph state, breaks bit-reproducibility, introduces bloat, and cannot trace discrete simulation events.

### Option 3: Native SCM View & Twin-Rollout Engine
* Good: Native performance, zero bloat, bit-level reproducibility, strict epistemic honesty, and direct integration with EWM's systemic traces.
* Bad: Requires explicit invertibility modeling for custom dynamics.

## More Information

* Richens, J., & Everitt, T. (2024). "Robust agents learn causal world models." *ICLR 2024 oral*, [arXiv:2402.10877](https://arxiv.org/abs/2402.10877).
* Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
* Feature Specification: [FEAT-008](file:///docs/specs/features/FEAT-008-twin-rollout-counterfactual-scm-kernel.md)
* Related ADRs: [ADR-005](ADR-005-deterministic-reproducibility.md), [ADR-009](ADR-009-state-immutability-and-canonical-fingerprint.md), [ADR-010](ADR-010-systemic-trace-epistemic-honesty.md), [ADR-021](ADR-021-ood-detection-and-honest-causal-diagnostics.md), [ADR-033](ADR-033-scientific-ml-provenance-uq-and-causal-gates.md), [ADR-039](ADR-039-responsible-ai-trust-and-anti-overclaiming-enforcement.md).
