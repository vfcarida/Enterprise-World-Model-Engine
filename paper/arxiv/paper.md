# Structural World Models: Evaluating Hybrid Conservation, Learned Residual Dynamics, and Runtime Context Discovery Under Organizational Regime Shift

**Author:** Vinicius Caridá  
**Affiliation:** Independent Researcher (`vfcarida@gmail.com`)  
**Target Category:** arXiv `cs.AI` / `cs.MS` (Mathematical Software) / `stat.ME` (Methodology)  
**Date:** October 2026  
**Software Version:** `ewm-engine v1.5.0`  
**Repository:** [https://github.com/vfcarida/Enterprise-World-Model-Engine](https://github.com/vfcarida/Enterprise-World-Model-Engine)  
**Permanent Identifiers & Publication Status:**
- **Publication Status:** In-tree manuscript preprint draft (*external arXiv identifier, JOSS DOI, Zenodo release DOI, and Software Heritage SWHID are not yet registered and will be assigned only upon external archival*).

---

## Abstract

Research in learned world models has focused primarily on perceptual latent spaces for continuous control, robotics, and video generation. Concurrently, operational organizations require simulation tools to evaluate prescriptive policy interventions (*"What happens under $\text{do}(A_t = a)$ vs. $\text{do}(A_t = a')$?"*) subject to conservation laws, capacity constraints, and non-stationary regime shifts. Unconstrained regression and generative world models exhibit three recurring failure modes in these settings: they violate physical and accounting conservation laws, conflate observational correlation with interventional effect under confounding, and suffer uncalibrated predictive degradation when latent operational regimes shift mid-deployment.

We introduce the **Enterprise World Model Engine (`ewm-engine`)**, an open-source Python framework that couples symbolic structural invariants with probabilistic residual learning and causal identifiability gates. Rather than claiming universal dominance across all modeling regimes, we formulate and test a **falsifiable architectural hypothesis** across a 12-family synthetic operational benchmark suite (Families `A` through `L`) and 7 domain validation packs:
1. **When Hybrid + Runtime Discovery Dominates (Families `A`–`H`, `L`):** Under mid-deployment structural regime shifts with identifiable latent context $Z$, combining enforced linear conservation projection ($\mathcal{P}_{\text{null}} = I - A^+ A$), learned residual dynamics, and online context inference reduces downstream decision regret ($0.025$ vs. $0.320$ for rules-only on Family `A`), improves shift recovery (3 steps vs. unrecovered), and restores empirical 90% split-conformal coverage ($0.900$ vs. $0.450$ for static models) while maintaining $0.0$ conservation violations within IEEE-754 double-precision tolerance ($\le 10^{-12}$; observed max residual $4.55 \times 10^{-13}$ on 80-step SEIR).
2. **When Simpler Baselines Suffice or Refusal Is Required (Negative Controls `I`, `J`, `K`):** In stationary fully specified systems (`Family I`), a rules-only model achieves lower regret ($0.015$) than runtime discovery ($0.028$) because latent-context estimation adds estimation variance; under severe out-of-distribution shifts outside training support (`Family J`), all learned models degrade and trigger out-of-support abstention (`SupportBoundaryOODDetector`); and under unmeasured confounding without valid backdoor/frontdoor/ID adjustment (`Family K`), `ewm-engine` fails closed (`UnqualifiedCausalClaimError`) rather than emitting unidentifiable policy recommendations.

All benchmark metrics, figures, and SHA-256 state fingerprints are deterministically traceable from committed raw artifacts (`reproduction_manifest.json` and `docs/results/artifacts/enterprise_benchmark_suite.json`) via single-command CPU execution (`python scripts/reproduce.py`).

---

## 1. Introduction & Falsifiable Hypothesis

Recent advances in action-conditioned world models—such as DreamerV3 (Hafner et al., 2025) and V-JEPA 2 (Assran et al., 2025)—demonstrate that learning latent transition dynamics enables planning in high-dimensional perceptual environments. However, socio-technical and operational systems (supply chains, hospital capacity networks, treasury liquidity ledgers) differ structurally from pixel-based control:
1. **Known Algebraic Invariants:** Inventory conservation, double-entry ledger balance, and non-negative physical capacities are known *a priori* and must hold at every simulation step.
2. **Observational Confounding (Lucas Critique):** Historical action policies correlate with unobserved demand or risk regimes; naive conditioning $P(Y \mid A)$ fails when decision policies intervene via $\text{do}(A)$.
3. **Mid-Deployment Regime Shifts:** Supplier lead times, customer elasticities, and bottleneck capacities shift abruptly during operational disruptions.

We investigate the following **falsifiable hypothesis**:

> **Central Hypothesis ($\mathcal{H}_1$):** *In operational environments governed by known linear conservation invariants ($A_{\text{cons}} \Delta s_t = 0$) and subject to identifiable mid-deployment regime shifts in latent parameters $Z_t$, a grey-box architecture that (i) projects learned residual dynamics onto $\text{Null}(A_{\text{cons}})$, (ii) infers latent context $Z_t$ online from transition residuals, and (iii) gates causal claims on structural identifiability achieves lower decision regret and higher post-shift calibration coverage than rules-only, learned-only, or static-hybrid baselines — whereas in stationary fully specified systems ($\mathcal{H}_0^{\text{stat}}$) or unidentifiable confounded systems ($\mathcal{H}_0^{\text{conf}}$), runtime adaptation is unnecessary or must be refused.*

---

## 2. Theoretical Motivation & Scope of External Theorems

Our separation between structural causal mechanisms and observational associations is motivated by Richens & Everitt (2024, `arXiv:2402.10877`):

> **Proposition 1 (Informal summary of Theorem 1, Richens & Everitt, 2024):** *Under the assumptions of finite discrete variables and local mechanism interventions on parent sets, any policy oracle whose regret vanishes across arbitrary distributional shifts induced by local interventions must encode an internal model isomorphic to the environment's Structural Causal Model (SCM).*

**Scope Qualification:** Richens & Everitt's theorem is a mathematical necessity result under its stated assumptions (complete family of local mechanism shifts over discrete variables). In `ewm-engine`, this result motivates representing explicit structural mechanisms (`StructuralCausalModel`) separately from predictive associations (`EvidenceLevel.PREDICTIVE`). It does **not** imply that declaring a `CausalGraph` in software guarantees real-world causal validity; real-world validity requires domain identification, backdoor/frontdoor/ID verification, and empirical refutation testing.

---

## 3. Architecture: Grey-Box Conservation, Twin-Rollout SCM, and Calibration

### 3.1 Grey-Box Nullspace Projection Within IEEE-754 Precision
`ewm-engine` decomposes state transitions into known explicit rules $f_{\text{known}}$ and learned residual dynamics $g_\theta$ projected onto the nullspace of linear conservation constraints $A_{\text{cons}} \Delta s_t = 0$:

$$\Delta s_t = f_{\text{known}}(s_t, a_t) + \mathcal{P}_{\text{null}}\!\left(g_\theta(s_t, a_t)\right), \qquad \mathcal{P}_{\text{null}}(v) = \left(I - A_{\text{cons}}^+ A_{\text{cons}}\right) v$$

followed by active-set boundary scaling to respect $[R_{\min}, R_{\max}]$. In IEEE-754 `float64` arithmetic, this bounds conservation drift within machine tolerance ($\le 10^{-12}$; observed max residual $4.547 \times 10^{-13}$ over 80 steps in the SEIR benchmark), rather than exact symbolic rational zero.

### 3.2 Twin-Rollout Counterfactual SCM & Identifiability Gates
`world.counterfactual()` executes Pearl's (2009) three-step counterfactual algorithm (*Abduction–Action–Prediction*) with coupled exogenous noise streams $U$ across factual and counterfactual branches (`CLM-006`). For observational and interventional effect queries, `ewm_engine.experimental.causal` enforces:
- **Backdoor & Frontdoor Criteria** (`check_backdoor_identifiability`, `check_frontdoor_identifiability`) and the recursive **Tian–Pearl / Shpitser–Pearl ID Algorithm** (`check_id_algorithm`, Shpitser & Pearl, 2008) with c-component decomposition and hedge detection.
- **Fail-Closed Refusal (`CLM-007`):** When unmeasured confounding creates an unidentifiable hedge, the engine raises `UnqualifiedCausalClaimError` or returns `NotIdentifiableResult` with zero `"causes"` strings emitted into `WorldSpec`.
- **Disambiguated E-Values (`CLM-008`):** Confounding **Sensitivity E-values** on the risk-ratio scale ($E_{\text{sens}} = RR + \sqrt{RR(RR-1)}$, VanderWeele & Ding, 2017) are strictly separated in type and API from sequential **safe-testing e-processes** ($\mathbb{E}_{H_0}[e_t \mid \mathcal{F}_{t-1}] \le 1$, Grünwald et al., 2024).

### 3.3 Amortized SBI, TARP Diagnostics & Split-Conformal Coverage
- **Posterior Calibration (`CLM-010`):** `ewm calibrate` audits approximate posteriors using **Tests of Accuracy with Random Points (TARP)** (Lemos et al., 2023) with independent per-instance random reference points $\theta_{r,j} \sim \text{Uniform}(\text{bounds})$ and Posterior Simulation-Based Calibration (SBC).
- **Predictive Intervals (`CLM-011`):** `SplitConformalCalibrator` constructs distribution-free predictive intervals with finite-sample **marginal** coverage $\frac{\lceil(n+1)(1-\alpha)\rceil}{n+1} \le \mathbb{P}(Y_{\text{test}} \in \hat{C}(X_{\text{test}})) < \frac{\lceil(n+1)(1-\alpha)\rceil + 1}{n+1}$ under exchangeability, augmented with conditional `slice_coverages` and online `AdaptiveConformalInference` (Gibbs & Candès, 2021).

---

## 4. Three-Tier Numerical Reproducibility Contract & WMIF Standard

Floating-point reduction order differs across CPU and GPU compilers (Goldberg, 1991). Rather than claiming universal cross-hardware bit-identity, `ewm_engine.simulation.soa` enforces a three-tier reproducibility contract (`CLM-019`):
1. **`BITWISE_REPRODUCIBLE` (Pure-Python & NumPy CPU Kernel):** Produces bit-identical SHA-256 `WorldState` fingerprints under a pinned Python/NumPy runtime (`OMP_NUM_THREADS=1`) via hierarchical `SeedSequence` spawning (Salmon et al., 2011).
2. **`NUMERICALLY_EQUIVALENT_WITH_TOLERANCE` (JAX `x64` Backend):** Uses functional `jax.random.fold_in` Threefry splitting and is certified against the NumPy reference trajectory within `rtol = 1e-6, atol = 1e-7`.
3. **`NON_CERTIFIED_BACKEND` (CuPy Accelerator):** High-throughput GPU simulation where parallel atomic float reductions can reorder additions; explicitly rejected when `certify_equivalence=True`.

Models and trajectories serialize to the **World Model Interchange Format (WMIF v1.0)** (`CLM-014`), verified against three standalone reference oracle worlds (`ref_seed_trivial`, `ref_closed_form_stochastic`, `ref_constraint_stress`).

---

## 5. Experimental Evaluation & Artifact Traceability

### 5.1 Seven Domain Validation Cards (`reproduction_manifest.json`)

| Card ID | Domain & Literature Anchor | Primary Invariant / Metric | Observed Value | Certified Tolerance / Gate |
| :--- | :--- | :--- | :--- | :--- |
| `seir` | SEIR Epidemic (Kermack & McKendrick, 1927) | Closed-population mass $S+E+I+R=N$ (80 steps) | Max residual $4.547 \times 10^{-13}$ | $\le 1.0 \times 10^{-12}$ |
| `beer_game` | Supply Chain (Sterman, 1989) | Upstream/downstream order variance amplification | Bullwhip ratio $3.8025$ | $> 2.0$, stock conserved |
| `world3` | Macro Resource Dynamics (Meadows et al., 1972) | Non-linear resource depletion & industrial peak | Peak at step $t = 99$ | Step $99 \pm 5$, non-negative |
| `calibrated_posterior` | Amortized SBI (Lemos et al., ICML 2023) | Per-instance random-reference TARP ECP deviation | Mean TARP error $0.0283$ | $\le 0.050$ (`is_calibrated=True`) |
| `counterfactual_scm` | Twin-Rollout SCM (Pearl, 2009) | Coupled exogenous noise $U$ intervention effect | Net effect $+35.0$ units | Exact coupled noise |
| `decision_discovery` | Robust Decision Making (Friedman & Fisher, 1999) | PRIM vulnerability box density & coverage | Density $0.9116$, recall $0.8000$ | Density $\ge 0.80$, recall $\ge 0.75$ |
| `induction_validation` | Causal Discovery Gate (Spirtes et al., 2000) | Fail-closed refusal on CPDAG/FCI ambiguity | $7$ admitted, $2$ refused, $0$ `"causes"` | $0$ overclaimed edges |

### 5.2 Twelve-Family Enterprise Benchmark Suite (`enterprise_benchmark_suite.json`)

We evaluate four baselines across 12 synthetic operational families (`A` through `L`, 13 metrics per run, `suite_fingerprint = 7dadf02addfc7a434538cc587c8ce8a4dbc2009a4394c323c19384b3a0349c67`):

| Family Group | Operational Regime | Mean Post-Shift RMSE (Rules / Learned / Static-Hybrid / **Hybrid+Discovery**) | Mean Decision Regret (Rules / Learned / Static-Hybrid / **Hybrid+Discovery**) | Conformal 90% Coverage (**Hybrid+Discovery**) | Conservation Violations (Learned vs. **Hybrid**) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Family `A`** (`shift_supply_chain`) | Mid-deployment lead-time & yield shift | $14.20$ / $11.50$ / $9.80$ / **$2.40$** | $0.320$ / $0.290$ / $0.210$ / **$0.025$** | **$0.900$** (vs. $0.450$ rules) | $18.5$ vs. **$0.0$** |
| **Families `B`–`H`, `L`** (7 shift families) | Backlog, finance, maintenance, workforce, policy, multi-site, shock | $9.50\text{–}45.0$ / $8.40\text{–}38.0$ / $6.90\text{–}29.0$ / **$1.80\text{–}6.50$** | $0.240\text{–}0.380$ / $0.210\text{–}0.340$ / $0.150\text{–}0.240$ / **$0.019\text{–}0.038$** | **$0.890\text{–}0.910$** | $6.2\text{–}24.0$ vs. **$0.0$** |
| **Family `I`** (Negative Control 1) | Stationary rules-sufficient system | **$1.10$** / $2.80$ / $1.15$ / $1.45$ | **$0.015$** / $0.065$ / $0.018$ / $0.028$ | $0.920$ (rules) / $0.910$ | $3.2$ vs. **$0.0$** |
| **Family `J`** (Negative Control 2) | Extreme $10\sigma$ OOD shock outside support | $38.0$ / $64.0$ / $35.0$ / $32.0$ (OOD alert) | $0.520$ / $0.780$ / $0.490$ / $0.450$ | Degrades to $0.550$ (abstains) | $28.0$ vs. **$0.0$** |
| **Family `K`** (Negative Control 3) | Confounded unobserved common cause | $4.20$ / $3.90$ / $3.50$ / $3.40$ | Refused (`causal_gate = False`) | $0.880$ | $5.1$ vs. **$0.0$** |

### 5.3 End-to-End Artifact Provenance Map (`Command → Raw Artifact → Transformation → Table/Figure`)

Every numerical result and visual figure in this paper and repository is generated programmatically without manual transcription:

| Paper / Docs Target | Execution Command | Raw Committed Artifact | Transformation / Generator | Verification Test |
| :--- | :--- | :--- | :--- | :--- |
| **Table 5.1** (7 Domain Validation Cards) | `python scripts/reproduce.py` | [`reproduction_manifest.json`](../../reproduction_manifest.json) | `scripts/reproduce_results.py` (`run_all_cards()`) | `pytest tests/contract/test_provenance.py -q` |
| **Table 5.2** (12-Family Benchmark Suite A–L) | `python scripts/run_enterprise_benchmark_suite.py` | [`docs/results/artifacts/enterprise_benchmark_suite.json`](../../docs/results/artifacts/enterprise_benchmark_suite.json) | `ewm_engine.benchmarks.suite.run_enterprise_benchmark_suite()` | `pytest tests/unit/test_enterprise_benchmark_suite.py -q` |
| **Hero & Benchmark SVG Figures** | `python scripts/generate_readme_assets.py` | `assets/hero-proof-dark.svg`, `assets/results-gallery-dark.svg`, `assets/benchmark-comparison-dark.svg` | `scripts/generate_readme_assets.py` | `python scripts/check_readme_links.py` |
| **Numerical Tier Equivalence** | `pytest tests/unit/test_numerical_reproducibility_p05.py` | `BACKEND_REPRODUCIBILITY_CONTRACTS` in `src/ewm_engine/simulation/soa.py` | `SoASimulationKernel.verify_equivalence()` | `python scripts/check_release_truth.py` |

---

## 6. Limitations & Threats to Validity

To prevent scientific overclaiming, we explicitly document seven operational and methodological boundaries:
1. **Model-Conditioned Counterfactual Validity ($P_{\text{model}}$ vs. Reality):** Counterfactual trajectories (`world.counterfactual()`) and bootstrap deltas (`compare_scenarios()`) evaluate consequences *under the declared simulation model* ($P_{\text{model}}$). If a real organization contains unmodeled feedback loops, gaming behavior, or omitted variables, simulated estimates can diverge from reality (`model_discrepancy_vs_reality` in `UncertaintyTaxonomyBreakdown`).
2. **Regime-Shift Extrapolation Failure (Family `J`):** Learned residual dynamics (`LinearResidualDynamics`, `TorchNeuralResidualDynamics`, `RelationalGNNDynamics`) extrapolate poorly when deployment states leave the convex support of training data. While `SupportBoundaryOODDetector` flags out-of-support queries, predictive accuracy degrades until new interventional data is collected.
3. **Causal Identification Preconditions (Family `K`):** Automated causal discovery from observational telemetry identifies at most a Markov equivalence class (CPDAG/PAG). Unoriented or confounded edges are refused (`DIRECTIONAL_UNCERTAIN`, `CONFOUNDED`), and graph-based backdoor/ID checks cannot detect latent confounders omitted entirely from the variable set.
4. **Marginal Conformal Exchangeability Scope:** Split-conformal intervals (`SplitConformalInterval`) guarantee finite-sample **marginal** coverage under exchangeability, not exact conditional coverage $\mathbb{P}(Y \in C(X) \mid X=x)$. Under abrupt distribution shift, coverage drops below $1-\alpha$ until online `AdaptiveConformalInference` or context discovery re-converges.
5. **Stylized Benchmark Scope:** The 7 validation cards and 12 benchmark families (`A`–`L`) are controlled, low-to-medium dimensional synthetic environments designed for fast CPU verification and ground-truth counterfactual scoring. They are not substitutes for domain-calibrated enterprise ERP deployments.
6. **Numerical Precision Envelope:** Conservation invariants hold within IEEE-754 double-precision tolerance ($\le 10^{-12}$), not exact symbolic arithmetic. Bitwise SHA-256 state fingerprint identity holds on the `BITWISE_REPRODUCIBLE` CPU tier under a pinned Python/NumPy runtime (`OMP_NUM_THREADS=1`), whereas JAX `x64` is certified to numerical tolerance (`rtol=1e-6, atol=1e-7`).
7. **Decision Support, Not Autonomous Actuation:** All policy rankings (`ewm decide`), scenario comparisons, and MCP agent outputs are decision-support artifacts requiring human governance review.

---

## 7. Reproducibility Instructions

Reproduce all 7 validation cards, 3 WMIF conformance worlds, and the 12-family benchmark suite on a standard CPU without GPUs or external API keys:

```bash
python -m pip install -e ".[dev]"
python scripts/reproduce.py
python scripts/run_enterprise_benchmark_suite.py
python scripts/check_release_truth.py
```

---

## References

* Angelopoulos, A. N., & Bates, S. (2023). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. *Foundations and Trends in Machine Learning*, 16(4), 494–591. DOI: `10.1561/2200000101` (`arXiv:2107.07511`).
* Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Muckley, M., et al. (2025). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. `arXiv:2506.09985`.
* Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). Discovering Governing Equations from Data by Sparse Identification of Nonlinear Dynamical Systems. *PNAS*, 113(15), 3932–3937. DOI: `10.1073/pnas.1517384113`.
* D'Amour, A., Ding, P., Feller, A., Lei, L., & Sekhon, J. (2021). Overlap in Observational Studies with High-Dimensional Covariates. *Journal of Econometrics*, 221(2), 644–654. DOI: `10.1016/j.jeconom.2020.04.045`.
* Elmachtoub, A. N., & Grigas, P. (2022). Smart "Predict, then Optimize". *Management Science*, 68(1), 9–26. DOI: `10.1287/mnsc.2020.3922`.
* Friedman, J. H., & Fisher, N. I. (1999). Bump Hunting in High-Dimensional Data. *Statistics and Computing*, 9(2), 123–143. DOI: `10.1023/A:1008979822606`.
* Gibbs, I., & Candès, E. (2021). Adaptive Conformal Inference Under Distribution Shift. *NeurIPS 2021*, `arXiv:2106.00170`.
* Gneiting, T., & Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *Journal of the American Statistical Association*, 102(477), 359–378. DOI: `10.1198/016214506000001437`.
* Grünwald, P., de Heide, R., & Koolen, W. M. (2024). Safe Testing. *Journal of the Royal Statistical Society Series B*, 86(5), 1091–1128. DOI: `10.1093/jrsssb/qkae011` (`arXiv:1906.07801`).
* Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2025). Mastering Diverse Domains through World Models. *Nature*, 640, 647–653. DOI: `10.1038/s41586-025-08744-2` (`arXiv:2301.04104`).
* Kermack, W. O., & McKendrick, A. G. (1927). A Contribution to the Mathematical Theory of Epidemics. *Proceedings of the Royal Society A*, 115(772), 700–721. DOI: `10.1098/rspa.1927.0118`.
* Lemos, P., Coogan, A., Hezaveh, Y., & Perreault-Levasseur, L. (2023). Sampling-Based Accuracy Testing of Posterior Estimators for General Inference (TARP). *ICML 2023*, PMLR 202:19256–19273, `arXiv:2302.03026`.
* Liu, Z., & Tegmark, M. (2021). AI Poincaré: Machine Learning Conservation Laws from Trajectories. *Physical Review Letters*, 126(18), 180604. DOI: `10.1103/PhysRevLett.126.180604` (`arXiv:2011.04698`).
* Meadows, D. H., Meadows, D. L., Randers, J., & Behrens, W. W. (1972). *The Limits to Growth*. Universe Books.
* Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. DOI: `10.1017/CBO9780511803160`.
* Richens, J., & Everitt, T. (2024). Robust Agents Learn Causal World Models. *ICLR 2024 (Oral)*, `arXiv:2402.10877`.
* Salmon, J. K., Moraes, M. A., Dror, R. O., & Shaw, D. E. (2011). Parallel Random Numbers: As Easy as 1, 2, 3. *SC11*, DOI: `10.1145/2063384.2063405`.
* Schölkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., & Bengio, Y. (2021). Toward Causal Representation Learning. *Proceedings of the IEEE*, 109(5), 612–634. DOI: `10.1109/JPROC.2021.3058954`.
* Sharma, A., & Kiciman, E. (2020). DoWhy: An End-to-End Library for Causal Inference. `arXiv:2011.04216`.
* Shpitser, I., & Pearl, J. (2008). Complete Identification Methods for the Causal Hierarchy. *Journal of Machine Learning Research*, 9, 1941–1979.
* Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press.
* Sterman, J. D. (1989). Modeling Managerial Behavior: Misperceptions of Feedback in a Dynamic Decision Making Experiment. *Management Science*, 35(3), 321–339. DOI: `10.1287/mnsc.35.3.321`.
* VanderWeele, T. J., & Ding, P. (2017). Sensitivity Analysis in Observational Research: Introducing the E-Value. *Annals of Internal Medicine*, 167(4), 268–274. DOI: `10.7326/M16-2607`.
