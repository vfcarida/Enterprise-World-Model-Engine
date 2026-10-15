# ADR-042: Grey-Box Conservation Layer and Neuro-Symbolic Flywheel

<!-- This ADR is written according to the MADR 4.x standard -->

* Status: accepted
* Deciders: Vinicius Caridá (@vfcarida)
* Date: 2026-10-15
* Technical Story: Grey-box conservation layer + neuro-symbolic flywheel
* Supersedes: None

## Context and Problem Statement

Enterprise world models simulate organizational, logistical, financial, and socio-technical dynamics. A central challenge in contemporary world modeling is the tension between **known physical/accounting rules** and **opaque learned models**:
1. **Black-box neural simulators** predict transitions from scratch, but frequently hallucinate impossible states (e.g. creating inventory or capital out of thin air, or violating double-entry balance).
2. **Penalty-based conservation (PINNs)** adds soft penalty terms ($\lambda \|A \Delta - b\|^2$) to the loss function. While helpful in training, penalties fail under out-of-distribution shifts, leading to non-zero violation rates at inference time.
3. **Post-hoc rejection** discards violating states, which wastes compute, breaks rollout continuity, and introduces non-differentiable step boundaries.
4. **Static grey-boxes** keep empirical components permanently opaque, preventing newly discovered organizational friction (e.g., inventory leakage or demand drift) from being formalized into explicit, auditable policy rules.

How should EWM Engine enforce exact physical and accounting invariants architecturally, ensure strict zero-residual bit-reproducibility with the symbolic kernel, and provide a neuro-symbolic flywheel to systematically move mass from **LEARNED $\to$ KNOWN**?

## Decision Drivers

* **Universal Differential Equation (UDE) Framing:** Learned dynamics must predict **only the residual/delta** over the explicit known rule, never the full next-state (Rackauckas et al., 2020).
* **Invariant 5 Bit-Identity:** When the residual is zero or absent, the grey-box must reduce bit-for-bit to the symbolic kernel, producing identical cryptographic state fingerprints.
* **Architectural Conservation by Construction:** Conservation laws must be enforced via exact projection (KKT-hPINN nullspace projection; Port-Hamiltonian energy routing), making violations impossible by construction (structural 0.0% violation rate) rather than rejected post-hoc.
* **Neuro-Symbolic Self-Whitening:** Fit closed-form symbolic expressions to empirical residuals via SINDy STLSQ sparse regression, surfacing candidate rules with goodness-of-fit and parsimony metrics.
* **Audited Human-in-the-Loop Promotion:** Rule promotion from the empirical residual into the explicit ruleset (WSL) must require explicit human review and sign-off linked to an ADR/spec, never automatic.
* **Zero Mandatory Dependencies:** The core grey-box, KKT projection, and SINDy engine must run on pure Python/NumPy, with optional PyTorch (`[ml]`) and PySR (`[symbolic]`) as non-mandatory extras.

## Considered Options

* **Option 1: Soft Penalty Loss Regularization (Standard PINN)** — Train neural networks with conservation penalty loss. Rejected because constraint violations remain non-zero under distribution shift, violating enterprise safety.
* **Option 2: Post-Hoc Constraint Rejection & Rollout Invalidation** — Abort or discard rollout trajectories that violate conservation laws. Rejected because it wastes computation, causes rollout rejection cascades, and prevents continuous MPC planning.
* **Option 3: Architectural Projection Layer & Neuro-Symbolic Flywheel (Chosen)** — Implement UDE residual grey-box dynamics with exact KKT nullspace projection ($P = I - A^T (A A^T)^{-1} A$), Port-Hamiltonian dissipative structure, pure NumPy SINDy sparse regression, and an audited human-in-the-loop promotion workflow.

## Decision Outcome

Chosen option: **Option 3: Architectural Projection Layer & Neuro-Symbolic Flywheel**.

### Consequences

* **Good, because**:
  * **0.0% Violation Rate by Construction:** The KKT nullspace projection guarantees $A \Delta_{\text{proj}} = \mathbf{b}$ identically within machine precision for any neural or adversarial output.
  * **Lyapunov Stability:** Port-Hamiltonian parameterization ($(J - R) \nabla H$) eliminates explosive runaway instability in cyclical feedback systems.
  * **Strict Bit-Identity (S00 Invariant 5):** With zero residual, grey-box execution matches the symbolic kernel bit-for-bit, preserving exact state fingerprints and hashes.
  * **Self-Whitening Doctrine:** Discovered empirical residuals are translated into closed-form equations and promoted into the known ruleset, moving mass from LEARNED to KNOWN and shrinking unexplained variance.
  * **Epistemic Honesty:** All discovered dynamic terms default strictly to `EvidenceLevel.PREDICTIVE`, and honesty metrics contrast the structural-zero violation rate against unconstrained baselines.
  * **Human Authority:** No automated core mutations; promotion requires formal human review and spec linkage.
* **Bad, because**:
  * Highly non-linear manifolds require piecewise linear or iterative projection approximations.
  * Human-in-the-loop promotion requires operational discipline to review discovered candidates.

### Confirmation

* Unit tests in `tests/unit/test_grey_box.py` confirm zero-residual bit-reproducibility with symbolic kernel (Invariant 5) and PREDICTIVE evidence level transitions.
* Unit tests in `tests/unit/test_conservation_layer.py` confirm KKT nullspace projection zeroes out adversarial unconstrained deltas, verifies Port-Hamiltonian skew-symmetry, and reports 0.0% structural violation rates.
* Unit tests in `tests/unit/test_symbolic_flywheel.py` confirm SINDy recovers exact quadratic terms ($R^2 \ge 0.99$), validates advisory-only priors, enforces human approval gates, and demonstrates the self-whitening collapse of unexplained residual.
* Contract tests in `tests/contract/test_grey_box_contract.py` verify Invariant 5, structural conservation invariance, and strict observational evidence levels.

## More Information

* Rackauckas et al. "Universal Differential Equations for Scientific Machine Learning", arXiv:2001.04385
* KKT-hPINN: Hard Linear Equality Constraints for Neural Networks, arXiv:2402.07251
* Stable Port-Hamiltonian Neural Networks, arXiv:2502.02480
* Brunton et al. "Sparse Identification of Nonlinear Dynamics" (SINDy), PNAS 2016, arXiv:1509.03580
* Shojaee et al. "LLM-SR: Scientific Equation Discovery via Program Synthesis", ICLR 2025 oral, arXiv:2404.18400
