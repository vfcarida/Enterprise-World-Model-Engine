# ADR-021: Out-of-Distribution Detection, Grounded Regimes, and Honest Causal Diagnostics

## Status
Accepted (v2.0.0-experimental)

## Context
A major systemic risk in world-model simulation is the temptation to treat simulated outcomes as verified empirical forecasts or real-world causal proofs. Recent research in causal machine learning and world models highlights two foundational findings:
1. **Unavoidable Interventional Error in Observational World Models** (Song & Cai, arXiv:2610.00012; arXiv:2608.11601):
   Action-conditioned world models ($P(S_{t+1} \mid S_t, A_t)$) trained on observational logs incur unavoidable bias when unobserved confounding or open backdoor paths exist. Action conditioning is not equivalent to causal intervention:
   $$P(Y \mid X) \neq P(Y \mid \text{do}(X))$$
2. **Unbounded Autoregressive Extrapolation Outside Grounded Support**:
   When simulation trajectories enter state spaces outside the empirical support of training data, learned dynamics diverge rapidly due to compounding drift (demonstrated in the `LongHorizon` and `InterventionShift` benchmark families).

Previous versions of EWM Engine established the `EvidenceLevel` hierarchy (`STRUCTURAL`, `INTERVENTIONAL`, `QUASI_CAUSAL`, `PREDICTIVE`, `ASSUMED`) and forbade trace relations labeled as `"causes"`. However, the engine lacked:
- An operational mechanism to detect when a simulation rollout leaves its **grounded regime** (Out-Of-Distribution detection).
- Typed diagnostics for evaluating whether an interventional query is **identifiable** under stated structural assumptions, and measuring sensitivity to unobserved confounding.

## Decision

1. **Out-of-Distribution / Regime-Shift Detection (`ewm_engine.experimental.ood`)**:
   - Provide an operational novelty and support detector interface (`OODDetector`) over `WorldState` and `Trajectory`.
   - Implement `SupportBoundaryOODDetector` (bounding-box / quantile support) and `MahalanobisOODDetector` (multivariate correlation structure).
   - Report a typed `OODTrajectoryReport` including:
     - `grounded_fraction` ($\in [0.0, 1.0]$): The proportion of rollout steps that remained within verified grounded support.
     - `first_ood_step`: The earliest time step at which the trajectory entered an ungrounded extrapolation regime.
     - Per-step anomaly scores and offending resource attributes.
   - **Provenance Guarantee**: OOD detection attaches metadata to `SystemicTrace` nodes. It does **not** count as a constraint violation (invariants remain distinct from epistemic extrapolation) and **never modifies** declared `EvidenceLevel`s.

2. **Ladder-of-Causation Surface & Identifiability Gate (`ewm_engine.experimental.causal`)**:
   - Introduce a typed query interface reflecting Pearl's Causal Hierarchy:
     - `query_observational(X, Y)`: Returns observational association $P(Y \mid X)$ with `EvidenceLevel.PREDICTIVE`.
     - `query_interventional(X, Y, graph, conditioning_set)`: Evaluates whether the effect of $\text{do}(X)$ on $Y$ is identifiable via the Backdoor Criterion on the declared mechanism graph.
     - If unblocked backdoor paths exist, it returns `NotIdentifiable(open_backdoor_paths=[...])` rather than an ungrounded number.
   - **Immutable Epistemic Rule**: The `causal_level` tag is determined strictly by the declared structural graph and the mathematical check—it is **never inferred from data** and **never auto-upgraded**.

3. **Positivity & Confounding Sensitivity Diagnostics**:
   - Implement `check_positivity_overlap`: Evaluates propensity score distributions $e(X)$ across logged transitions and flags positivity violations where treatment was never observed.
   - Implement `report_confounding_sensitivity`: Implements sensitivity analysis (e.g. Rosenbaum bounds $\Gamma$) to report how strong an unobserved confounder would have to be to nullify an observed treatment delta.

4. **Noise-Coupled Counterfactual Branching (Twin Rollouts, arXiv:2608.08982)**:
   - Provide `twin_rollout_counterfactual`: Given a factual trajectory, couples the exact same deterministic pseudorandom noise stream across factual and counterfactual branches, making Judea Pearl's counterfactual abduction step exact by construction.
   - Compute and report **Off-Target Divergence (Locality)**: Measures whether non-targeted variables diverge between factual and counterfactual branches, surfacing unmodeled systemic coupling or confounding leakage.

5. **Strict Epistemic Stance on Causal Claims**:
   - The engine simulates interventions *under a specified world model*. It does **not** claim to discover real-world causality from observational logs.
   - All diagnostic reports must use conditional epistemic language: *"Under the declared structural assumptions $\mathcal{A}$, the effect is identifiable / not identifiable"*.

## Consequences

### Positive
- Users and governance auditors can immediately see what fraction of a simulation was grounded vs. extrapolated (`grounded_fraction`).
- Interventional queries fail gracefully with explicit backdoor paths rather than emitting confounded numbers.
- Twin Rollouts provide exact noise-coupled counterfactuals and measure structural locality.
- Epistemic integrity is preserved: `EvidenceLevel` cannot be upgraded automatically.

### Negative / Trade-offs
- OOD detection requires a calibration/training phase over reference states.
- Causal identifiability requires domain engineers to explicitly declare mechanism DAGs.
