# Scientific ML Rigor, Provenance, UQ, and Causal Gates

!!! info "Discipline: Scientific Rigor & Epistemic Honesty"
    In simulation and enterprise world models, superficial scalar accuracy metrics (such as 1-step MAE or raw $R^2$) create a dangerous illusion of understanding.
    The Enterprise World Model Engine establishes a **scientifically reference-grade** ML evaluation, provenance, uncertainty quantification, and causal gating layer.
    Core contracts remain pure Python and NumPy; heavy frameworks reside in optional extras (`[ml]`, `[uq]`, `[calibrate]`, `[causal]`, `[ope]`).

---

## 1. The Reproducibility Spine & Provenance Manifests

Scientific credibility requires complete, auditable determinism. The engine maps 1:1 to the **NeurIPS Reproducibility Checklist** through an immutable, canonical execution manifest:

### `RunManifest` Architecture

Every simulation, calibration, and evaluation run emits a cryptographic `RunManifest`:

```python
from ewm_engine.provenance.manifest import create_run_manifest

manifest = create_run_manifest(
    component_id="neural_residual_v2",
    scenario_id="global_supply_chain_shock",
    seed=1337,
    resolved_config={"learning_rate": 1e-3, "batch_size": 64},
    engine_version="1.1.0",
    deterministic=True,
)

print(manifest.manifest_hash)  # Canonical SHA-256 fingerprint
print(manifest.reproducibility_checklist)
```

The manifest cryptographically seals:
1. **Master seed(s)** and full multi-seed streams.
2. **Canonical configuration hash** computed over all resolved parameters.
3. **Participating component IDs and semantic versions** across dynamics, actors, and event sources.
4. **Git commit SHA** and working tree state.
5. **Exact environment dependency versions** (`python`, `numpy`, `torch`, `scipy`, `networkx`).
6. **Hardware platform summary** (CPU architecture, OS platform).
7. **Determinism flags in effect** (`PYTHONHASHSEED`, `OMP_NUM_THREADS=1`, `CUBLAS_WORKSPACE_CONFIG`).

The canonical `manifest_hash` is folded into all downstream model cards and `SystemicTrace` provenance graphs.

### Deterministic Seeding & Multi-Seed Reporting

- `seed_everything(seed, deterministic=True)`: Synchronizes Python's `random`, NumPy's default RNG, single-threaded BLAS/OpenMP routines, and PyTorch deterministic algorithms (emitting clear warnings if `CUBLAS_WORKSPACE_CONFIG` is unset).
- **Zero Bare Point Estimates**: `evaluate_multiseed` spawns statistically independent child seeds via `np.random.SeedSequence.spawn` and reports **mean $\pm$ non-parametric bootstrap confidence intervals**. Bare point estimates without uncertainty are prohibited in formal reporting.

---

## 2. World-Model Dynamics Evaluation: Beyond 1-Step Errors

A known pitfall of learned simulators is that **one-step predictive error does not predict multi-step rollout stability** (Talvitie, 2014; Venkatraman et al., 2015). A model with negligible 1-step MAE can rapidly drift into unphysical regimes when rolled out autoregressively.

### The Compounding-Error Curve

The engine evaluates rollout divergence as an explicit per-horizon curve $\text{err}(h)$ for $h = 1 \dots H$, tracking the **Compounding Error Ratio**:

$$C(h) = \frac{\text{err}(h)}{h \cdot \text{err}(1)}$$

```
Compounding
Ratio C(h)
    ▲
3.0 ┼                                  ╭── Super-linear drift (C(h) > 1.5)
2.0 ┼                           ╭──────╯   [Unsafe for deep planning]
1.0 ┼───────────────────────────┴───────── Linear error growth (C(h) = 1.0)
0.0 ┼────────────────────────────────────► Horizon h
    1    2    3    4    5    6    7    8
```

- $C(h) \approx 1.0$: Error accumulates linearly (additive errors without state feedback amplification).
- $C(h) \gg 1.0$: Super-linear divergence where small autoregressive errors compound feedback loops into severe hallucination.
- **Horizon Selection**: `select_reliable_horizon(error_curve, max_compounding_ratio=2.0)` determines the maximum reliable planning horizon $h^*$ for receding-horizon Model Predictive Control (MPC).

### Latent vs. Observation Error Decomposition

For latent models (e.g. Graph Neural Dynamics or auto-encoded world states), the harness decouples error into:
1. **Latent Representation Divergence**: Drift in internal embeddings.
2. **Physical Observation Error**: Divergence in measurable system resources (inventory, cash, capacity).

### The Calibrated-Dynamics Gate

An accurate-but-overconfident model fails verification:

```python
from ewm_engine.experimental.dynamics_eval import evaluate_calibrated_dynamics_gate

passed = evaluate_calibrated_dynamics_gate(
    calibration_result,
    min_coverage=0.80,  # Must cover >= 80% under 90% nominal interval
    max_crps=0.25,      # Continuous Ranked Probability Score upper ceiling
)
```

---

## 3. Uncertainty Quantification & Pareto Frontiers

### Strictly Proper Scoring Rules

The engine adopts **strictly proper scoring rules** as primary probabilistic evaluation metrics (Gneiting & Raftery, 2007):

1. **Continuous Ranked Probability Score (CRPS)** (Continuous default):
   $$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} (F(x) - \mathbf{1}(x \ge y))^2 \, dx = \frac{1}{M}\sum_{i=1}^M |x_i - y| - \frac{1}{2M^2}\sum_{i=1}^M \sum_{j=1}^M |x_i - x_j|$$
2. **Gaussian Log Score (Negative Log-Likelihood)**:
   $$\text{NLL}(y) = \frac{1}{2}\ln(2\pi\sigma^2) + \frac{(y - \mu)^2}{2\sigma^2}$$
3. **Brier Score** (Categorical & Regime Shifts):
   $$\text{Brier} = \frac{1}{N}\sum_{i=1}^N \sum_{k=1}^K (p_{ik} - y_{ik})^2$$

### Adaptive Calibration Error (ACE) vs. Fixed-Width ECE

Naive Expected Calibration Error (ECE) divides confidence into fixed-width bins (e.g., $[0.8, 0.9]$), which causes severe distortions:
- Bins in tail regions are frequently empty or contain $< 5$ samples, inflating variance.
- Overconfidence in rare failure modes is masked by dominant central bins.

`compute_adaptive_calibration_error` implements **quantile-spaced equal-mass binning** (Roelofs et al., 2022) with finite-sample variance debiasing (Broecker, 2012):

$$\text{Debiased ACE} = \max\left(0, \sum_{b=1}^B \frac{|B_b|}{N}|\text{conf}(B_b) - \text{acc}(B_b)| - \sqrt{\sum_{b=1}^B \frac{|B_b|}{N}\frac{\text{Var}(y \in B_b)}{|B_b|}}\right)$$

For continuous regression forecasts, `compute_pit_calibration` executes Probability Integral Transform (**PIT**) testing: if forecasts $F_i$ are well-calibrated, $u_i = F_i(y_i) \sim \mathcal{U}(0, 1)$.

### Distribution-Free Conformal Prediction & ACI

To eliminate reliance on Gaussian or parametric assumptions, the engine provides distribution-free conformal calibration:

- **`ConformalIntervalPredictor`**: Computes non-conformity quantiles $\hat{q} = \text{Quantile}\left(\frac{\lceil (n+1)(1-\alpha)\rceil}{n}\right)$ on calibration hold-outs, guaranteeing exact finite-sample coverage $P(Y \in \mathcal{C}(X)) \ge 1 - \alpha$ under exchangeability (Vovk et al., 2005; Romano et al., 2019).
- **`AdaptiveConformalInference` (ACI / Conformal-PID)**: Under non-stationary simulation drift or real-world regime shifts, exchangeability fails. ACI (Gibbs & Candès, 2021; Angelopoulos et al., 2023) dynamically updates target miscoverage rates online:
  $$\alpha_{t+1} = \alpha_t + \gamma (\alpha - \text{err}_t)$$
  Bursts of interval miscoverage trigger an automated **OOD / Regime-Shift Alert**, linking directly into the engine's out-of-distribution detector.

### The UQ Pareto Frontier

```python
from ewm_engine.evaluation.conformal import compute_uq_pareto_front

front = compute_uq_pareto_front(candidates)
print(front.frontier)  # Non-dominated models
```

The engine formulates uncertainty quantification as a multi-criteria optimization across 4 conflicting objectives:
1. **Sharpness** (minimize interval width)
2. **Empirical Coverage** (maximize coverage rate)
3. **Calibration Error** (minimize ACE or CRPS)
4. **Latency / Compute Overhead** (minimize evaluation time in ms)

!!! warning "Epistemic Humility: Unidentifiability of Aleatoric vs. Epistemic Uncertainty"
    Following **Mucsányi et al. (2024, arXiv:2406.07663)**, *pure disentanglement of aleatoric and epistemic uncertainty is mathematically unidentifiable from finite observational data alone*.
    Claims of evidential or single-pass neural networks that claim to perfectly isolate epistemic risk should not be over-trusted. The engine treats UQ selection as a Pareto decision over empirical trade-offs.

---

## 4. Causal Rigor: Identifiability, Gated OPE, and E-Values

The engine strictly rejects the notion of "automated causal discovery from observational data alone." All causal estimates are **hard-gated**:

### 1. Identifiability as a Blocking Prerequisite

Before calculating an interventional effect $P(Y \mid \text{do}(X))$, the declared causal DAG undergoes formal Backdoor Criterion analysis (`check_backdoor_identifiability`):
- If all backdoor paths are blocked by observed covariates without conditioning on descendants of treatment $\to$ returns `InterventionalQueryResult`.
- If confounding paths remain open $\to$ returns `NotIdentifiableResult` detailing the open confounding paths. **The engine NEVER outputs a numerical effect estimate when unidentifiable**.

### 2. Gated Off-Policy Evaluation (OPE)

When evaluating counterfactual policies over logged simulation rollouts, `evaluate_off_policy` calculates:
- Direct Method (DM): $\hat{V}_{\text{DM}} = \mathbb{E}[\hat{Q}(s, \pi(s))]$
- Inverse Propensity Scoring (IPS): $\hat{V}_{\text{IPS}} = \frac{1}{N}\sum_i w_i r_i$
- Self-Normalized IPS (SNIPW / Hajek): $\hat{V}_{\text{SNIPW}} = \frac{\sum_i w_i r_i}{\sum_i w_i}$
- **Doubly Robust (DR)** (Default): $\hat{V}_{\text{DR}} = \hat{V}_{\text{DM}} + \frac{1}{N}\sum_i w_i (r_i - \hat{Q}(s_i, a_i))$

#### Mandatory Diagnostic Gates Every Time

Every OPE estimate mandates an importance weight audit:
- **Kish's Effective Sample Size (ESS)**: $\text{ESS} = \frac{(\sum w_i)^2}{\sum w_i^2}$. If $\text{ESS}/N < 0.10$, flags severe overlap breach.
- **Max Weight Concentration**: If $\max(w) / \sum w > 0.20$, flags excessive single-sample leverage.
- **Weight Clipping Rate**: Audits fraction of weights truncated by clipping thresholds.
- **Triangulation Disagreement**: Audits divergence between DM and IPS.

#### The "No Auto-Upgrade" Rule

If diagnostic gates fail, the engine **downgrades the epistemic standing** (e.g. from `INTERVENTIONAL` to `PREDICTIVE` or `ASSUMED`). **Under NO circumstances is `EvidenceLevel` auto-promoted.**

### 3. Positivity & High-Dimensional Overlap Warnings

Observational propensity scores must satisfy common support:
- `check_positivity_overlap` calculates the Bhattacharyya overlap coefficient and **trimmed mass** (fraction of samples in extreme propensity tails $< \epsilon$ or $> 1 - \epsilon$).
- **High-Dimensional Warning**: Citing **D'Amour et al. (2017, arXiv:1711.02582)**, strict overlap collapses exponentially as covariate dimension $d$ grows. When $d \ge 5$, the engine attaches a high-dimensional overlap fragility warning.

### 4. VanderWeele & Ding E-Values

Every causal point estimate and confidence interval limit carries an **E-value** (VanderWeele & Ding, 2017, *Ann. Intern. Med.*):

$$E = \text{RR} + \sqrt{\text{RR}(\text{RR} - 1)}$$

The E-value specifies the minimum strength of association on the risk ratio scale that an unmeasured confounder must have with both the intervention and the outcome to nullify the observed effect.

### 5. Causal Refutation Battery

Mirroring DoWhy (Sharma & Kiciman, 2020), `run_causal_refutations` executes:
1. **Placebo Treatment Refuter**: Shuffles treatment; effect must collapse near zero.
2. **Random Common Cause Refuter**: Injects uninformative noise into conditioning sets; effect must remain invariant.
3. **Data Subset Stability Refuter**: Subsamples 80% of transitions; effect must remain stable.
4. **Unobserved Confounder Sensitivity**: Verifies effect stability against simulated hidden bias.

---

## 5. Google ML Test Score & Metamorphic Readiness

### The ML Test Score Rubric

`compute_readiness_report` implements the production readiness rubric developed by Google (Breck et al., IEEE Big Data 2017). Readiness is evaluated on a 0-4 point scale across four pillars:

1. **Data Tests**: Schema consistency, config hash, dependency tracking, OOD grounded support fraction.
2. **Model Tests**: Compounding error curves, calibrated dynamics gate, metamorphic gates, gated OPE.
3. **Infrastructure Tests**: Canonical `RunManifest`, deterministic execution recipe, git commit tracking, hardware recording.
4. **Monitoring Tests**: OOD extrapolation detection, runtime invariant monitoring, ESS weight tracking, interventional shift auditing.

$$\text{ML Test Score} = \min(\text{Data}, \text{Model}, \text{Infra}, \text{Monitoring})$$

The overall score is strictly capped by the weakest category—preventing high modeling scores from masking infrastructure or monitoring debt.

### Metamorphic Regression Gates

The engine exploits its explicit structural rule layer as a **metamorphic oracle** (Chen et al., 1998; Murphy et al., 2008). On model version changes, `run_metamorphic_regression_gate` verifies:
- **Non-negativity preservation**: Candidate model does not predict negative values for bounded resources.
- **Bounded version divergence**: Model drift relative to baseline stays within compounding tolerances.
- **Monotonicity preservation**: Increasing resource investment does not decrease invariant returns.

---

## 6. Citation Rigor & Literature Anchors

To prevent academic drift, the engine codifies precise citations for core concepts:

| Topic | Canonical Research Anchor | Common Citation Trap (Avoid!) |
|---|---|---|
| **Latent Video/Dynamics Eval** | Assran et al. (2024), *V-JEPA*, **arXiv:2404.08471** | Do NOT cite earlier I-JEPA (2023) for video/dynamics evaluation. |
| **High-Dim Overlap Collapse** | D'Amour et al. (2017), *Overlap in High-Dimensional Observational Studies*, **arXiv:1711.02582** | Overlap fragility is a dimensional phenomenon, not sample-size only. |
| **Value-Aware Model Learning** | Farahmand et al. (2017), *VAML*, **PMLR / NeurIPS 2017** | No arXiv preprint exists; cite official PMLR proceedings. |
| **Uncertainty Disentanglement** | Mucsányi et al. (2024), *Proper UQ*, **arXiv:2406.07663** | Aleatoric vs. epistemic split is unidentifiable without structural priors. |
| **Compounding Rollout Error** | Talvitie (2014), *Model Diagnostics for RL*, **AAAI 2014** | One-step MAE $\neq$ rollout stability. Quantify $C(h) = \text{err}(h) / (h \cdot \text{err}(1))$. |
| **E-Values for Sensitivity** | VanderWeele & Ding (2017), **Annals of Internal Medicine** | Minimum unmeasured confounding strength on Risk Ratio scale. |
| **ML Production Debt** | Breck et al. (2017), *The ML Test Score*, **IEEE Big Data 2017** | Total Score $= \min(\text{Data}, \text{Model}, \text{Infra}, \text{Monitoring})$. |
| **Adaptive Conformal Inference** | Gibbs & Candès (2021), **NeurIPS 2021**; Angelopoulos et al. (2023) | Time-series online quantile updates double as OOD regime shift signal. |
