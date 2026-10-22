# ADR-054: Enterprise World-Model Evaluation & Failure-Mode Benchmark Suite (A–L)

- **Status**: Accepted
- **Date**: 2026-10-21
- **Author**: Vinicius Caridá
- **Related Specifications**: [`docs/specs/spec-driven-development.md`](../specs/spec-driven-development.md), [`ADR-025`](ADR-025-scientific-benchmark-protocol-and-shift-families.md), [`ADR-033`](ADR-033-scientific-ml-provenance-uq-and-causal-gates.md), [`ADR-039`](ADR-039-responsible-ai-trust-and-anti-overclaiming-enforcement.md), [`ADR-044`](ADR-044-causal-correctness-metamorphic-evaluation.md), [`ADR-053`](ADR-053-formal-enterprise-world-model-semantics-and-context.md)

---

## 1. Context and Problem Statement

Standard machine-learning benchmarks evaluate predictive models using a single aggregate accuracy metric (e.g., observational one-step MSE or $R^2$) on independent and identically distributed (i.i.d.) splits. However, recent literature on enterprise workflow modeling, long-horizon business simulation, and world-model evaluation—including *World of Workflows*, *CascadeBench* (*Do Enterprise Systems Need Learned World Models?*), *Vending-Bench 2*, *Vending-Bench Arena*, Vafa et al. (2024, 2025), *Physics-IQ* (Motamed et al., 2025), and *RWM-U*—demonstrates that surface one-step accuracy fails to answer the core enterprise question:

> *"When is an enterprise world model actually useful, correct, calibrated, robust, and decision-relevant?"*

Specifically, evaluating world models for socio-technical and enterprise systems must expose failure modes across:
1. **Indirect workflow cascades** where an action triggers multi-hop downstream side-effects and SLA constraints that were rarely excited in observational training logs (*World of Workflows* / *CascadeBench*).
2. **Cross-tenant runtime configuration ($K_t$) and deployment shift** where nominal state/action schemas remain identical while underlying workflow thresholds, tax rules, or reserve ratios change mid-operation (Lucas, 1976; Schölkopf et al., 2021).
3. **Partial observability ($O_t, B_t$)** where actors and models observe filtered projections and delayed audit telemetry rather than omniscient internal state.
4. **Long-horizon business operations and multi-agent economies** with inventory, cash, accounts payable, supplier pipelines, spoilage, commitments, and strategic market clearing (*Vending-Bench 2* / *Vending-Bench Arena*).
5. **Causal/counterfactual identifiability and decision regret**, where low observational prediction error can coexist with high decision regret under confounding, or moderate baseline offset error can coexist with optimal policy selection.

Crucially, **a scientific benchmark suite must not be optimized to make any single architecture look artificially perfect**; its purpose is to expose failure modes across all paradigms, including context-aware hybrid models (such as transient error during audit-discovery latency, partial-observability filtering gaps, unmodeled OOD structural physics requiring abstention, and computational overhead).

---

## 2. Decision

### 2.1 Twelve Independent Benchmark Families (A–L)

We extend `benchmarks/` with twelve independent, reproducible enterprise benchmark families (`EnterpriseBenchmarkFamily`), while preserving full backward compatibility with the five original `BenchmarkFamily` probes from [`ADR-025`](ADR-025-scientific-benchmark-protocol-and-shift-families.md):

| Code | Family Identifier | Primary Scientific Probe & Failure Mode Exposed | Oracle Type |
| :--- | :--- | :--- | :--- |
| **A** | `A_DirectDynamics` | Known linear transfer coupled with empirical congestion friction | Deterministic exact oracle ($\le 10^{-12}$) |
| **B** | `B_HiddenWorkflowCascades` | Multi-hop indirect compliance holds, capacity throttling, and SLA penalties (*World of Workflows*) | Deterministic structural cascade oracle |
| **C** | `C_RuntimeConfiguration` | Cross-tenant deployments sharing nominal APIs but differing in `ContextSnapshot` $K_t$ rules | Stochastic multi-seed tenant oracle |
| **D** | `D_DeploymentShift` | Unseen mid-rollout configuration/mechanism shift with delayed audit telemetry ($\Delta t_{\text{audit}} > 0$) | Stochastic multi-seed shift oracle |
| **E** | `E_PartialObservability` | Filtered observations $O_t = O(S_t, K_t)$ with withheld latent supplier backlog and controlled audit hints | Stochastic multi-seed POMDP oracle |
| **F** | `F_LongHorizonBusinessOperation` | 60-step inventory, cash, payables, supplier delays, spoilage, commitments, and contract changes (*Vending-Bench 2*) | Stochastic multi-seed business simulator |
| **G** | `G_MultiAgentEconomy` | Strategic buyer, supplier, and arbitrage actors competing/clearing under budget and supply conservation (*Vending-Bench Arena*) | Stochastic multi-seed market clearing |
| **H** | `H_CausalCounterfactualCorrectness` | Synthetic SCMs with exact interventional $do(X=x)$, twin-rollout counterfactuals, and unobserved confounding refusal | Stochastic multi-seed SCM oracle |
| **I** | `I_ConservationConstraintStress` | Adversarial actions and infeasible learned residuals attempting mass fabrication and bound violations | Deterministic exact manifold oracle ($\le 10^{-12}$) |
| **J** | `J_UncertaintyCalibration` | Heteroscedastic state/context-dependent noise testing 90% interval coverage, ACE, and CRPS | Stochastic multi-seed distributional oracle |
| **K** | `K_OODRegimeShift` | Out-of-support action scales + unmodeled structural phase transition requiring epistemic abstention | Stochastic multi-seed OOD oracle |
| **L** | `L_DecisionQuality` | Decoupling predictive error from decision regret $U(\pi^*) - U(\hat{\pi})$ under confounded history vs. additive offset | Stochastic multi-seed utility oracle |

### 2.2 Four Mandatory Baseline Paradigms

Every family evaluates four standardized baseline paradigms (`BaselineParadigm`) on identical inputs:
1. **`rules-only` (`RulesOnlyBaseline`)**: Explicit deterministic structural rules (`KNOWN_DETERMINISTIC`) with fixed compile-time context $K_0$ and zero learned residual.
2. **`learned-only` (`LearnedOnlyBaseline`)**: Pure empirical transition regressor (`LEARNED`) fitted strictly on the training partition `TransitionDataset` without structural conservation projections or runtime context $K_t$.
3. **`static hybrid` (`StaticHybridBaseline`)**: Structural rules plus feasible-set projected learned residual (`HYBRID` with `InfeasibleResidualPolicy.PROJECT`), locked to compile-time context $K_0$.
4. **`hybrid + runtime context discovery` (`ContextualHybridBaseline`)**: Structural rules plus feasible-set projected learned residual (`HYBRID`), `AuditMetadataContextDiscovery` evolving $K_t$ from permitted runtime audit metadata, `ExponentialSmoothingBeliefUpdater` for partial observability, and `CompetenceEstimator` for OOD abstention.

### 2.3 Eleven Task Types and Thirteen Orthogonal Metric Dimensions

The benchmark suite evaluates eleven explicit task types (`BenchmarkTaskType`: `next_state_prediction`, `action_effect_prediction`, `constraint_prediction`, `workflow_side_effect_prediction`, `audit_trace_reconstruction`, `configuration_context_inference`, `counterfactual_query`, `policy_ranking`, `robust_action_selection`, `model_abstention`, `confidence_calibration`) across thirteen formally defined, non-collapsed metric dimensions (`MetricDimension`):

1. `TRANSITION_FIDELITY`: Normalized state transition fidelity $\frac{1}{1 + \text{MAE}(S_{t+1}, \hat{S}_{t+1})} \in (0, 1]$ alongside raw MAE/RMSE.
2. `CONSTRAINT_PRESERVATION`: Fraction of transitions satisfying all hard capacity, non-negativity, and conservation laws $1 - \frac{N_{\text{viol}}}{N_{\text{steps}}} \in [0, 1]$.
3. `CAUSAL_CORRECTNESS`: Composite of descendant-selectivity pass rate, twin-rollout counterfactual fidelity, and fail-closed confounding refusal accuracy $\in [0, 1]$.
4. `CALIBRATION`: Calibration score $1 - \min(1, |\hat{C}_{0.90} - 0.90| + \text{ACE}) \in [0, 1]$ alongside empirical 90% coverage, ACE, and CRPS.
5. `LONG_HORIZON_ERROR_GROWTH`: Autoregressive error growth slope $\frac{\text{MAE}(H) - \text{MAE}(1)}{H - 1} \ge 0$.
6. `SHIFT_ROBUSTNESS`: Post-shift fidelity retention $\frac{1}{1 + \text{MAE}_{\text{post-shift}}} \in (0, 1]$.
7. `TIME_TO_RECOVERY_AFTER_SHIFT`: Steps $\Delta t_{\text{recovery}} \in \mathbb{N}_0$ after a deployment shift until step error returns below recovery threshold $\tau_{\text{rec}}$ (or $H_{\text{post}}$ if unrecovered).
8. `DECISION_REGRET`: Expected utility gap $U^*(\pi^*) - U^*(\hat{\pi}) \ge 0$ evaluated in the ground-truth oracle environment.
9. `ABSTENTION_QUALITY`: Youden's $J$ / F1 score of epistemic abstention across in-competence and out-of-competence/unidentifiable queries $\in [0, 1]$.
10. `TRACE_PROVENANCE_COMPLETENESS`: Recall of ground-truth causal mechanism, rule, and context version edges in `SystemicTrace` $\in [0, 1]$.
11. `LATENCY`: Mean wall-clock execution latency per transition step ($\text{ms/step}$).
12. `MEMORY`: Peak memory allocation delta per evaluation run ($\text{KB}$).
13. `THROUGHPUT`: Executed transition steps per second ($\text{steps/s}$).

**Anti-Collapse Invariant**: No function in `benchmarks/` may average or collapse these thirteen dimensions into a single aggregate "world-model leaderboard score."

### 2.4 Statistical Protocol, Shift Protocol, and Data-Leakage Guardrails

1. **Statistical Protocol (`MetricStatisticalSummary`)**:
   - **Deterministic Exact-Oracle Families (A, I)**: Report `exact_value`, `oracle_tolerance` ($10^{-12}$), and `within_tolerance: bool` with `statistical_significance_applicable = False`.
   - **Stochastic Families (B, C, D, E, F, G, H, J, K, L)**: Evaluate across $M \ge 5$ independent seeds spawned via `np.random.SeedSequence`. Report `mean`, `median`, `iqr`, 95% bootstrap CI `[ci_lower, ci_upper]`, **Monte Carlo standard error** ($\text{SE}_{\text{MC}} = s / \sqrt{M}$) separated from **internal model uncertainty** ($\sigma_{\text{model}}$), paired differences against `rules-only` with paired Cohen's $d_z$ and Holm-Bonferroni step-down adjusted $p$-values, and explicit counts of `failed_runs` / `total_runs`.
2. **Shift Protocol (`ShiftProtocolRecord`)**:
   - Records `training_context_version`, `evaluation_context_version`, `what_changed`, `drift_signals`, `model_informed`, `runtime_information_available`, `adaptation_budget_steps`, `audit_discovery_delay_steps`, and `steps_until_recovery`.
3. **Hard Data-Leakage Boundaries (`LeakageGuard` & `DataLeakageCertificate`)**:
   - Enforces cryptographic seed-partition separation between training/induction generation and evaluation generation, zero state-fingerprint overlap, strict temporal precedence ($t_{\text{train}} < t_{\text{eval}}$), held-out context version isolation, and oracle hiding.
4. **Automated Artifact-to-Documentation Pipeline**:
   - All Markdown tables and SVG visual figures in `docs/research/enterprise-benchmark-suite.md` and `docs/results/artifacts/` are generated programmatically from `docs/results/artifacts/enterprise_benchmark_suite.json`.

---

## 3. Consequences

### Positive
- Provides a rigorous, self-tested 12-family benchmark instrument that tests both the strengths and honest limitations of symbolic, learned, static-hybrid, and context-aware hybrid world models.
- Enforces strict separation between deterministic exact-oracle verification and multi-seed stochastic inference with effect sizes, uncertainty intervals, and multiple-testing correction.
- Prevents benchmark contamination via audited `DataLeakageCertificate` checks.

### Trade-offs and Limitations
- Executing all 12 families across 4 baselines and multiple seeds requires ~1–3 seconds in standard mode (`--fast` mode completes in <0.5 seconds for CI smoke checks).
- Synthetic enterprise scenarios, while mathematically grounded and structurally rich, remain stylized representations of enterprise workflows; real-world deployment still requires domain-specific backtesting and causal identification.
