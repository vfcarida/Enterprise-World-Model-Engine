# ADR-033: Scientific ML Rigor — Provenance Spine, Compounding Dynamics Curves, UQ Pareto Frontiers, and Gated Causal Inference

## Status
Accepted

## Context
The Enterprise World Model Engine integrates learned dynamics, probabilistic rollouts, counterfactual simulations, and structural mechanism graphs. While these capabilities allow enterprise decision-makers to evaluate complex interventions, ML-based simulation frameworks in operational domains face severe credibility risks if evaluated with superficial point metrics or unverified causal claims.

Prior to Round 4 (R05), several scientific rigor gaps existed:
1. **Unstructured Provenance**: Runs lacked a canonical, cryptographic execution manifest mapping directly to international reproducibility standards (such as the NeurIPS Reproducibility Checklist).
2. **Scalar Rollout Error Metrics**: Dynamics models were evaluated using single scalar metrics (e.g. 1-step MAE), masking super-linear compounding drift over multi-step rollouts (Talvitie, 2014) and failing to distinguish latent-space representations from observable system state errors.
3. **Improper Probabilistic Scoring & Naive ECE**: Uncertainty was evaluated using ad-hoc quantiles or fixed-width Expected Calibration Error (ECE), which suffers from empty-bin distortion and masks severe overconfidence in sparse tail intervals. Furthermore, distribution-free finite-sample guarantees were absent.
4. **Ungated Causal Estimation & Auto-Upgrade Risk**: Causal estimators ran the risk of outputting numbers even when structural DAGs were unidentifiable, or when observational propensity overlap completely collapsed in high dimensions (D'Amour et al., 2017).
5. **Absence of Production Readiness Architecture**: There was no formal readiness rubric assessing technical debt across data, models, infrastructure, and monitoring (Breck et al., 2017).

## Decision
We establish a reference-grade scientific ML architecture across five core pillars, maintaining a lightweight core (pure Python/NumPy/Pydantic) while delegating heavy frameworks to optional extras (`[ml]`, `[uq]`, `[calibrate]`, `[causal]`, `[ope]`):

### 1. Reproducibility Spine (CORE)
- **`RunManifest`**: Every simulation and evaluation run emits a canonical JSON manifest containing master seed(s), resolved-config cryptographic hash, component/scenario IDs and versions, engine version, git commit, runtime dependency versions, hardware summary, determinism environment flags (`PYTHONHASHSEED`, `OMP_NUM_THREADS`, `CUBLAS_WORKSPACE_CONFIG`), and wall-clock execution duration.
  - The canonical SHA-256 `manifest_hash` maps 1:1 to the NeurIPS Reproducibility Checklist criteria and folds into system provenance and Model Cards.
- **`seed_everything(seed)`**: Centralized deterministic seeding recipe configuring Python's built-in `random`, NumPy's default RNG, single-threaded BLAS thread pinning, and optional PyTorch deterministic algorithms (with warnings if `CUBLAS_WORKSPACE_CONFIG` is unset).
- **Default Multi-Seed Reporting**: Mandate `evaluate_multiseed` with `SeedSequence.spawn` for multi-seed trials, reporting mean $\pm$ bootstrap confidence intervals. Point estimates without variance or confidence intervals are prohibited in standard reporting.

### 2. World-Model Dynamics Evaluation (CORE)
- **Per-Horizon Error Curves**: Rollout error is evaluated as a full horizon curve $\text{err}(h)$ for $h = 1 \dots H$, accompanied by the **compounding-error ratio** $C(h) = \frac{\text{err}(h)}{h \cdot \text{err}(1)}$ to quantify super-linear drift (Talvitie, 2014; Venkatraman et al., 2015).
- **Observation vs. Latent Space Error**: Explicitly decomposes rollout error into latent representational divergence and physical observation-space error for neural/GNN dynamics.
- **Horizon Selection**: `select_reliable_horizon` identifies the maximum planning horizon $h^*$ before compounding error exceeds simulation safety tolerances for Model Predictive Control (MPC).
- **Calibrated-Dynamics Gate**: An accurate point model that is overconfident fails the evaluation gate. Models must satisfy minimum coverage and CRPS bounds.

### 3. Uncertainty Quantification & Pareto Fronts (CORE + Extras)
- **Strictly Proper Scoring Rules**: Standardize on Continuous Ranked Probability Score (**CRPS**) as the continuous probabilistic evaluation metric (Gneiting & Raftery, 2007), Gaussian negative log-likelihood (log score / NLL), and multi-class Brier score for regime labels.
- **Adaptive Calibration Error (ACE)**: Replace naive fixed-width ECE with equal-mass quantile binning to avoid empty-bin distortion, paired with finite-sample variance debiasing (Broecker, 2012; Roelofs et al., 2022). For continuous forecasts, implement Probability Integral Transform (**PIT**) uniformity testing.
- **Conformal Prediction**: Implement split conformal / CQR-style `ConformalIntervalPredictor` providing distribution-free finite-sample coverage validity $P(Y \in \mathcal{C}(X)) \ge 1 - \alpha$.
- **Adaptive Conformal Inference (ACI)**: Provide streaming time-series conformal updates (Gibbs & Candès, 2021; Angelopoulos et al., 2023) where bursts of interval miscoverage dynamically flag out-of-distribution (OOD) regime shifts.
- **UQ Pareto Frontier**: Present model uncertainty as an explicit Pareto trade-off across sharpness (width), empirical coverage, calibration error (CRPS/ACE), and compute latency. We explicitly document that aleatoric and epistemic uncertainty disentanglement is fundamentally unidentifiable from finite data without untestable structural assumptions (Mucsányi et al., 2024).

### 4. Causal Rigor, Gates, and E-Values (CORE Gates + Extras)
- **Identifiability as a Blocking Gate**: Interventional queries $P(Y \mid \text{do}(X))$ must pass structural Backdoor/DAG checks before numerical estimation. If unidentifiable, the engine returns `NotIdentifiableResult` detailing open confounding paths; **it never outputs a heuristic numerical estimate**.
- **Off-Policy Evaluation (OPE)**: Implements Direct Method (DM), Inverse Propensity Scoring (IPS), Self-Normalized IPS (SNIPW), and Doubly Robust (**DR**, default) over logged trajectories.
  - **Mandatory Diagnostics**: Every OPE computation audits Kish's Effective Sample Size (ESS), max-weight ratio, weight percentiles ($p95, p99$), clip rate, and DM-vs-IPS-vs-DR triangulation agreement.
  - **Epistemic Downgrade (No Auto-Upgrade)**: If diagnostics fail (e.g., $ESS/N < 0.10$ or extreme weight concentration), confidence is downgraded from `INTERVENTIONAL` to `PREDICTIVE` or `ASSUMED`. Under no circumstances is `EvidenceLevel` auto-promoted.
- **Positivity and Overlap Gate**: Computes propensity overlap index and trimmed mass. Flags high-dimensional overlap vulnerability warnings when covariate dimension $d \ge 5$ (citing D'Amour et al., 2017).
- **E-Values**: Attaches the VanderWeele & Ding (2017) E-value to every interventional point estimate and confidence interval limit, establishing the minimum strength of unmeasured confounding needed to explain away the effect.
- **Refutation Battery**: Built-in refutation suite mirroring DoWhy (Sharma & Kiciman, 2020), including placebo treatment, random common cause, subset stability, and unobserved confounder sensitivity.

### 5. Production Readiness & Metamorphic Gates (CORE)
- **Google ML Test Score**: `compute_readiness_report` implements the rubric from Breck et al. (IEEE Big Data 2017), computing separate 0-4 point scores across Data Tests, Model Tests, Infrastructure Tests, and Monitoring Tests. The overall score is $\min(\text{Data}, \text{Model}, \text{Infra}, \text{Monitoring})$.
- **Metamorphic Regression Gates**: Uses the engine's explicit conservation rules, non-negativity guarantees, and symmetry laws as metamorphic relations (Chen et al., 1998) to verify candidate dynamics models against baseline versions without requiring ground-truth human annotations.

### 6. Citation Rigor & Dependency Boundaries
- We correct key literature citation traps:
  - V-JEPA v1: Assran et al., arXiv:2404.08471 (not earlier I-JEPA papers).
  - D'Amour et al.: Overlap in high-dimensional observational studies, arXiv:1711.02582.
  - Farahmand et al. (VAML): Value-Aware Model Learning, PMLR/NeurIPS 2017 (no arXiv version).
  - Mucsányi et al.: Aleatoric/epistemic unidentifiability, arXiv:2406.07663.
- Core dependencies remain strictly `numpy`, `pydantic`, and standard Python. Heavy libraries (`torch`, `scipy`, `scikit-learn`, `networkx`) remain lazy-imported extras.

## Consequences

### Positive
- **Auditable Provenance**: Every run produces an immutable, cryptographically verifiable fingerprint.
- **Scientifically Honest Forecasting**: Rollout errors report compounding drift curves rather than deceptively optimistic 1-step MAE metrics; probabilistic predictions carry proper scoring rules and distribution-free conformal guarantees.
- **Epistemic Integrity in Causality**: Eliminates false causal discoveries from observational data through strict identifiability gating, OPE diagnostics, common support trimming, and E-values.
- **Clear Production Gate**: Google's ML Test Score establishes an objective readiness benchmark.

### Negative / Trade-offs
- **Computation Overhead**: Multi-seed bootstrap evaluations, ACI updates, and refutation batteries require additional CPU cycles during model validation.
- **Stricter Gate Failures**: Models that appear accurate under uncalibrated metrics will be blocked by the calibrated dynamics gate and OPE diagnostic gates.
