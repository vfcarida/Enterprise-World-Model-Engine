# ADR-044: Causal-Correctness Gates, Metamorphic Oracles, and Epistemic Abstention

- **Status:** Accepted
- **Date:** 2026-10-15
- **Deciders:** Simulation Architecture, Evaluation & Verification Teams
- **Consulted:** S01 Causal SCM Team, S02 Decision Science Team, S04 Calibration Team
- **Informed:** Core Framework Engineers, Release Engineers

---

## Context

Enterprise world models simulate complex dynamic systems and guide strategic capital allocations. Traditional ML validation relies on observational point metrics (RMSE, CRPS, predictive likelihood, conformal coverage). However:
1. **Predictive Accuracy $\neq$ Causal Correctness:** Frontier research (Vafa et al., NeurIPS 2024 / ICML 2025) proves that models can achieve near-zero observational rollout error while maintaining an incoherent internal state. When subjected to interventions $do(X = x)$, such models perturb non-descendants or propagate changes in inverted directions.
2. **Perceptual Realism $\neq$ Physical Correctness:** The Physics-IQ paradox (Motamed et al. 2025; Kang et al. 2025) demonstrates that output plausibility is uncorrelated with physical truth. World models can produce visual or trajectory rollouts that "look right" while violating conservation of mass, non-negativity, and continuity.
3. **Absence of Ground Truth at Evaluation Time:** In prospective enterprise simulations, future empirical ground truth does not exist. Label-free metamorphic relations (Chen et al. 2020) are required to evaluate whether model invariants hold.
4. **Ungrounded Extrapolation:** Simulators extrapolating into unverified horizons or out-of-support parameter spaces must not output confident forecasts that deceive downstream decision optimizers (`ewm decide`).

---

## Decision

We establish an evaluation subsystem orthogonal to rollout error, making model honesty and reliability first-class scored features:

1. **Causal-Correctness & Intervention Selectivity Gates:**
   - Execute interventions $do(X = x)$ over the SCM compiled from `WorldSpec`.
   - Assert that $do(X = x)$ perturbs **only descendants of $X$** under Twin-Rollout coupled exogenous noise.
   - Assert that descendants with declared monotonic edge signs diverge in the correct mathematical direction.
   - Any non-descendant divergence or sign inversion triggers an `InterventionSelectivityViolation` and fails the gate.

2. **Metamorphic Relations as a Label-Free Oracle:**
   - Implement executable MR-cards:
     - `MR-1-DETERMINISM`: Seed + spec determinism (Invariant 5).
     - `MR-2-CONSERVATION`: Total mass balance across steps matches recorded applied flows (zero unmetered leakage).
     - `MR-3-SCALING`: Proportional scaling of initial resource capacities.
     - `MR-4-MONOTONICITY`: Supply adequacy; initial stock increases cannot drop terminal stock.
     - `MR-5-PERMUTATION`: Isomorphic entity ID permutation invariance.
     - `MR-6-CONSTRAINT-SLACK`: Adding non-binding slack constraints leaves valid rollouts unchanged.
   - Implement `MutationAdequacyEvaluator` injecting synthetic non-equivalent defects (`UnmeteredLeakageDynamics`, `UnseededJitterDynamics`, `PermutationBiasDynamics`). Require a mutation adequacy score $\ge 80\%$.

3. **Epistemic Abstention & Decision Integration:**
   - Implement `CompetenceEstimator` evaluating continuous competence $C \in [0, 1]$ via exponential horizon decay, historical support bounds, and constraint boundary proximity.
   - Wire competence scoring into `DecisionEngine.decide` (`weight_by_competence=True` and `filter_abstained=True`) so that ungrounded rollouts are down-weighted or filtered from candidate policy rankings.
   - Implement `PhysicsIQGuardrail` and formalize the doctrine: *plausibility/realism is never a correctness proxy*.

4. **Anti-Overclaiming Guarantee:**
   - Passing causal reliability gates certifies consistency with the assumed structural graph; it does not claim to have discovered unconditional physical truth, nor does it automatically upgrade `EvidenceLevel`.

---

## Consequences

### Positive
- Closes the critical evaluation gap between predictive fit and interventional correctness.
- Provides a robust label-free testing oracle for enterprise regimes without future ground truth.
- Protects downstream decision pipelines (`ewm decide`) from optimizing against ungrounded hallucinations.
- Fully operationalizes EWM's core thesis: "honesty is a scored feature."

### Negative / Trade-offs
- Running metamorphic transformations and counterfactual rollouts incurs computational overhead (mitigated by fast, deterministic gates in CI and heavy probes run nightly).
- Requires system modelers to explicitly declare monotonic edge directions if directional sign checking is desired.
