# ADR-049: Induction Honesty, Validation, and Fail-Closed Refusal Gate

## Context and Problem Statement

Automated world-model induction (process/schema and causal/dynamics) is a powerful capability for creating digital twins from enterprise telemetry. However, unconstrained automated induction presents severe epistemic and operational hazards:

1. **Observational Equivalence Overclaiming**: Observational data identifies only a Markov equivalence class (CPDAG or PAG), never a single unique Directed Acyclic Graph (DAG) (Spirtes et al., 2000; arXiv:2305.10032). Continuous score-based discoverers (e.g. NOTEARS) are vulnerable to variance scale ordering manipulation (arXiv:2206.07195). Asserting a single directed causal mechanism without interventional disambiguation or functional asymmetry is scientifically invalid.
2. **Spurious Invariants & Overfitting**: Symbolic regression and nullspace discovery can identify numerical regularities in training data that fail under real-world dynamics or out-of-distribution rollouts. Promoting an empirical regularity to a hard conservation law without rigorous stress testing produces fragile, invalid simulations.
3. **Hallucinatory Process Models**: Mined process models often overfit trace logs, failing basic replay conformance (fitness and precision) against unseen process executions (Carmona et al., 2018).
4. **Epistemic Dilution**: Traditional auto-modeling tools fail open, writing whatever heuristic graph they discover into the model. When downstream decision-makers optimize interventions against such models, policy failures are inevitable.

Enterprise World Model Engine enforces **a mandatory fail-closed induction gate that refuses to emit unidentifiable or refuted structure**. This architectural invariant prevents unverified discovery outputs from entering `WorldSpec`.

## Considered Options

1. **Fail-Open Advisory Validation**:
   - Run validation checks and print warnings or heuristic confidence badges in logs, but emit all candidate structures into `WorldSpec`.
   - *Rejected*: Allows unverified and directional-uncertain structures to contaminate the simulation kernel; violates ADR-039 anti-overclaiming invariants.
2. **Manual Review Only**:
   - Output raw discovery tables and require a human engineer to transcribe each relationship into `WorldSpec`.
   - *Rejected*: Impractical for enterprise systems with hundreds of entities and thousands of transitions.
3. **Mandatory, Fail-Closed Validation & Refusal Gate (`ewm induce validate`)**:
   - Establish a unified, mandatory gate every induction path must pass through.
   - Represent candidate structures honestly as equivalence classes (CPDAG/PAG) with posterior probabilities (DiBS, arXiv:2105.11839).
   - Apply shared empirical validators:
     - **DoWhy-style causal refuters** (placebo treatment, random common cause, data subset).
     - **Held-out trajectory reproduction** ($R^2 \ge 0.70$) for induced dynamics.
     - **Monte Carlo perturbation stress testing** for candidate invariants ($0.00\%$ violation rate required for hard constraints).
     - **Trace replay conformance** (minimum fitness $\ge 0.95$, precision $\ge 0.90$) for process models.
   - **Fail closed**: Decline to emit any directed causal edge, hard conservation law, or provisional link into `WorldSpec` unless it clears `ASSERTED` status.
   - Surface all refused structures to the human modeler as **explicit epistemic open questions with candidate options**, never silently dropping or asserting them.
   - Produce deterministic, fingerprinted induction audit reports with byte-identical reproducibility across runs.

## Decision Outcome

Chosen option: **Option 3 (Mandatory, Fail-Closed Validation & Refusal Gate)**.

### Architectural Rules & Invariants

1. **The Induction EvidenceLevel Ladder**:
   Extending ADR-039 (`core/trust.py` and `provenance/evidence.py`), all induced structural elements are assigned an honest epistemic standing from the formalized induction ladder:
   $$\text{ASSERTED} \succ \text{PROVISIONAL} \succ \text{DIRECTIONAL\_UNCERTAIN} \succ \text{UNIDENTIFIED} \succ \text{CONFOUNDED}$$
   - `ASSERTED`: Identifiable, ensemble-stable, oriented, and cleared empirical refutation. Emitted into `WorldSpec`.
   - `PROVISIONAL`: Relational link predicted without foreign key or verified schema constraint. Refused from `WorldSpec`.
   - `DIRECTIONAL_UNCERTAIN`: Symmetric edge in observational Markov equivalence class ($X \text{ --- } Y$). Refused from directed dependency.
   - `UNIDENTIFIED`: Candidate structure whose causal or functional parameters cannot be uniquely determined from available data. Refused from `WorldSpec`.
   - `CONFOUNDED`: Latent common cause indicated by FCI bidirected edge ($X \leftrightarrow Y$) or refuter failure. Refused from direct causal assertions.
   - **No Auto-Upgrade**: EvidenceLevels can only be upgraded through verified empirical evidence (interventional experiments or human domain assertions).

2. **Shared Empirical Validators**:
   - **DoWhy Refuters**:
     - *Placebo Treatment Refuter*: Replaces the treatment variable with random Gaussian noise; the estimated treatment effect must drop to near-zero ($p \ge 0.05$).
     - *Random Common Cause Refuter*: Adds an independent random confounder to the regression; the original effect estimate must remain invariant ($|\Delta \hat{\beta}| \le 0.20 \cdot |\hat{\beta}|$).
     - *Data Subset Refuter*: Re-estimates effects on bootstrap sub-samples (80%); estimates must remain statistically consistent.
   - **Held-Out Dynamics Reproduction**:
     - Induced rate laws must reproduce unseen held-out trajectories under seeded rollouts with $R^2 \ge 0.70$.
   - **Monte Carlo Invariant Stress Testing**:
     - Candidate conservation laws must demonstrate a $0.00\%$ violation rate under Monte Carlo perturbation rollouts to be promoted to hard constraints. Leaky candidates are refused and surfaced as open questions.
   - **Process Model Conformance Replay**:
     - Transition models must achieve $\ge 0.95$ replay fitness and $\ge 0.90$ precision against event logs (Carmona et al., 2018).

3. **The Refusal Gate Mechanics**:
   - An induced element enters `WorldSpec` **if and only if** it achieves `ASSERTED` status.
   - In `--strict` mode, any unasserted or refused element causes the gate to fail closed immediately (`InductionRefusalError`, CLI exit code 1).
   - In non-strict mode, unasserted elements are filtered out of `WorldSpec` and surfaced in the `InductionValidationReport`.

4. **Epistemic Open Questions**:
   - Every refused structural candidate is converted into an `OpenQuestion` containing:
     - Clear statement of epistemic ambiguity (e.g. observational symmetry, latent confounding, invariant leakage).
     - Concrete candidate options for human resolution (e.g. domain assertion, interventional data collection, instrumenting missing variables).
     - Recommended fail-safe action.

5. **Anti-Overclaiming Guarantee (ADR-039)**:
   - Passing the gate means **"identifiable given assumptions and surviving refutation"**, never "absolute ground truth".
   - Under no circumstances is the raw string `"causes"` emitted as a relationship type in `WorldSpec`. Allowed edge types are `"directed_dependency"`, `"structural_dependency"`, or `"associated_with"`.

6. **Deterministic Fingerprinting & Reproducibility**:
   - All validation routines are parameterized with deterministic random seeds.
   - `InductionValidationReport` computes a canonical SHA-256 fingerprint over its sorted admitted elements, refused elements, open questions, and validator results.
   - Two validation runs on identical data produce **byte-identical** reports.

7. **Protocol & CLI Integration**:
   - CLI command: `ewm induce validate [--report REPORT] [--spec SPEC] [--strict] [--out SPEC_OUT] [--report-out REPORT_OUT] [--markdown-out MD_OUT]`.
   - MCP Server: Induction reports are registered and queryable via `ewm.explain(identifier=fingerprint)` and MCP resource URI `ewm://induction/{fingerprint}`.

## Consequences

### Positive
- **Methodological Trust**: Prevents confident-looking but subtly wrong world models from entering automated decision pipelines.
- **Scientific Honesty**: Explicitly quantifies and displays Markov equivalence, latent confounders, and invariant leakage.
- **Actionable Guidance**: Surfacing structured open questions tells domain experts exactly what experiments or data collection are required to identify unknown mechanisms.
- **Auditability**: Complete cryptographic traceability from raw telemetry to validated `WorldSpec`.

### Negative
- **Conservative Admission**: Models may have fewer relationships than unconstrained auto-modelers because unoriented or confounded edges are refused. (This is intentional and desirable.)
- **Computation Overhead**: Running DoWhy refuters and Monte Carlo stress tests adds execution time to the induction pipeline (mitigated by vectorized NumPy routines and lazy imports).

## References

- Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search*. MIT Press.
- Vowels, M. J., et al. (2023). *D'ya like DAGs? A Survey on Structure Learning and Causal Discovery*. arXiv:2305.10032.
- Lorch, L., et al. (2021). *DiBS: Differentiable Bayesian Structure Learning*. NeurIPS 2021 / arXiv:2105.11839.
- Reisach, A. G., et al. (2022). *Tearing Apart NOTEARS: Structure Learning with Continuous Optimization*. arXiv:2206.07195.
- Ke, N. R., et al. (2019). *Learning Neural Causal Models from Unknown Interventions*. arXiv:1910.01075.
- Sharma, A., & Kiciman, E. (2020). *DoWhy: An End-to-End Library for Causal Inference*. MIT.
- Carmona, J., et al. (2018). *Conformance Checking: Relating Processes and Models*. Springer.
- Liu, Z., & Tegmark, M. (2021). *AI Poincaré: Machine Learning Conservation Laws from Trajectories*. Physical Review Letters / arXiv:2011.04698.
