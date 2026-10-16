# ADR-048: Causal Structure, Dynamics, and Invariant Induction with GPL-3.0 License Isolation

## Context and Problem Statement

To simulate organizational interventions and counterfactual scenarios, Enterprise World Model Engine requires:
1. A **causal dependency graph** for its Structural Causal Model (SCM) kernel (`S01`, ADR-040).
2. **Dynamic rate laws** for its grey-box simulation kernel (ADR-042).
3. **Conservation laws and hard state invariants** to constrain state transitions and bound epistemic drift.

Automating the induction of causal structures and physical dynamics from observational and interventional enterprise data introduces three critical technical and governance challenges:
1. **License Contamination (The GPL-3.0 Trap)**: The leading time-series causal discovery library, Tigramite ([Science Adv. DOI:10.1126/sciadv.aau4996](https://doi.org/10.1126/sciadv.aau4996)), is licensed under the **GNU General Public License (GPL-3.0)**. Static-linking or importing Tigramite in-process would contaminate EWM Engine's permissive Apache-2.0 / MIT codebase.
2. **Identifiability & Epistemic Honesty**: Observational data in general identifies only a **Markov-equivalence class** (a CPDAG or PAG under latent confounding), never a unique DAG. Furthermore, continuous score-based methods (NOTEARS, DAGMA) are vulnerable to variable scaling and adversarial variance manipulation. Emitting a single asserted DAG from observational data alone is scientifically dishonest and violates EWM's anti-overclaiming invariants (ADR-039).
3. **Zero LLM on Core Simulation Path**: All induced rate laws and dynamics must be closed-form, deterministic, and verifiable without reliance on probabilistic generative AI in the critical simulation loop.

## Considered Options

1. **Single-DAG Observational Assertion with In-Process Tigramite**:
   - Run NOTEARS or DAGMA and emit a single DAG directly into `WorldSpec`. Direct import of Tigramite.
   - *Rejected*: Violates GPL-3.0 license isolation; scientifically dishonest regarding Markov equivalence and latent confounders; fragile to variance scaling.
2. **Pure Manual Specification**:
   - Require users to hand-author causal DAGs and rate laws.
   - *Rejected*: Incompatible with automated enterprise twin workflows.
3. **Ensemble Causal Discovery (CPDAG/PAG) + SINDy Dynamics + AI-Poincaré Invariants with Subprocess Quarantine**:
   - Run an ensemble of complementary discovery paradigms: constraint-based (PC/FCI for latent confounding), score-based (DAGMA/NOTEARS), functional (DirectLiNGAM for non-Gaussian asymmetry), and temporal lag discovery.
   - Reconcile ensemble outputs into an equivalence class annotated with per-edge `EvidenceLevel` (`ASSERTED`, `DIRECTIONAL_UNCERTAIN`, `CONFOUNDED`). Never emit an unearned DAG.
   - Disambiguate edge orientations via interventional data (A/B testing, policy shifts).
   - Isolate GPL-3.0 Tigramite strictly out-of-process via a subprocess bridge.
   - Induce closed-form rate laws via SINDy (STLSQ sparse regression) with goodness-of-fit validation.
   - Induce candidate conservation laws via AI-Poincaré SVD nullspace estimation, promoting them to hard constraints only after passing seeded Monte Carlo stress testing across all trajectories.

## Decision Outcome

Chosen option: **Option 3 (Ensemble Causal Discovery + SINDy Dynamics + AI-Poincaré Invariants with Subprocess Quarantine)**.

### Architectural Rules & Invariants

1. **Strict GPL-3.0 Subprocess Quarantine**:
   - EWM Engine core NEVER imports `tigramite` in-process.
   - Subprocess bridge (`ewm_engine.induction.tigramite_bridge`) invokes `tigramite_worker.py` in a separate OS process via `subprocess.run()`, exchanging data strictly via JSON stdin/stdout.
   - Permissive engines (`causal-learn`, `gcastle`, `pysindy`, `pysr`) are lazy-imported optional extras; pure Python/NumPy fallbacks exist for all core routines.
2. **Equivalence Class Representation (CPDAG/PAG)**:
   - Causal discovery returns a `CausalEquivalenceClass`.
   - Edges are annotated with epistemic standings:
     - `ASSERTED`: Edge is identifiable, ensemble-stable, and oriented (via functional asymmetry, temporal lag, or interventional validation).
     - `DIRECTIONAL_UNCERTAIN`: Edge existence is confirmed within the Markov class, but direction is observationally equivalent ($A \text{ --- } B$).
     - `CONFOUNDED`: Latent common cause indicated by FCI bidirected edge ($A \leftrightarrow B$).
   - In accordance with ADR-039, no raw `"causes"` string is ever emitted into `WorldSpec`; edges use `"directed_dependency"`, `"structural_dependency"`, or `"associated_with"`.
3. **Interventional Disambiguation**:
   - When interventional data (known targets $\text{do}(X=x)$) is provided, edges directed into the intervention target that fail to hold invariant are pruned or re-oriented, upgrading `DIRECTIONAL_UNCERTAIN` edges to `ASSERTED`.
4. **Interpretable Dynamics Induction**:
   - SINDy/STLSQ fits sparse rate laws $\dot{x}_i = \sum w_j \theta_j(x)$ over candidate polynomial and interaction libraries.
   - Each rate law receives an $R^2$ fit score mapped to `EvidenceLevel` (`CALIBRATED`, `PREDICTIVE`, `HYPOTHETICAL`).
   - Default discovery is strictly LLM-free. Any LLM-assisted proposer is advisory-only and strictly isolated from the core simulation path.
5. **Conservation-Law Stress Testing & Promotion Gate**:
   - AI-Poincaré finds conserved quantities $\sum_i w_i x_i = C$ spanning the empirical trajectory nullspace.
   - A candidate is promoted to a `WorldSpec` hard constraint (`HardConstraintSpec`) **only if it passes seeded Monte Carlo stress testing** across all sampled rollouts within numerical tolerance $\epsilon$.
   - Failing candidates are retained as soft constraints or logged in the `InductionReport` with empirical violation rates.
6. **Direct Compilation to SCM Kernel**:
   - `CausalEquivalenceClass.to_scm()` compiles `ASSERTED` directed edges directly into an executable `StructuralCausalModel` compatible with Pearl Rung-3 twin rollout counterfactuals (`S01`, ADR-040).
7. **Deterministic Fingerprinting**:
   - Every causal, dynamics, and invariant induction run generates a canonical SHA-256 fingerprint on `InductionReport`, guaranteeing reproducible, auditable governance.

## Consequences

### Positive
- **Epistemic Rigor**: Zero overclaiming of causal directions; users receive clear visual and quantitative distinctions between directed, undirected, and confounded relations.
- **Enterprise-Safe Licensing**: Tigramite is cleanly isolated; no copyleft contagion affects the Apache-2.0 / MIT core.
- **End-to-End Simulation**: Proposes ready-to-run SCM and grey-box models directly from raw organizational trajectories.
- **Verification-Gated Invariants**: Conservation laws are promoted to hard constraints only when verified under stochastic stress.

### Negative / Trade-offs
- Out-of-process subprocess execution for Tigramite adds IPC overhead (~20–80ms), which is negligible compared to the underlying optimization runtimes.
- True causal directionality remains unasserted when non-Gaussianity or interventional data is absent; this is an inherent mathematical property of observational data that EWM embraces rather than conceals.
