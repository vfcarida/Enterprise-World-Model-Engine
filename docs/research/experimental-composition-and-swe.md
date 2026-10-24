# Experimental Research Lines: Domain-Model Composition & Software-Engineering Ecosystem

> [!WARNING]
> **Fenced Experimental Status (`ewm_engine.experimental.*` — `PROPERTY_VERIFIED`):**  
> Both lines documented on this page are **forward-looking research hypotheses**, not promised capabilities of the stable `1.x` core. Everything resides in `ewm_engine.experimental.domain_composition` and `ewm_engine.experimental.swe_ecosystem` with zero new dependencies and zero changes to `ewm_engine.__all__`. All numbers below are generated from [`docs/results/artifacts/experimental_composition_and_swe.json`](../results/artifacts/experimental_composition_and_swe.json) via:
> ```bash
> python scripts/run_experimental_hypotheses.py --seed 42 --n-seeds 5
> ```

---

## Line 1 — Domain-Model Composition as a Tested Research Hypothesis (§16)

### 1.1 The Hypothesis & Why Monoliths vs. Composition Is Non-Trivial
Long-term enterprise modeling rarely succeeds as a single monolithic simulator that attempts to encode an entire corporation. Instead, organizations naturally partition ownership across **specialized domain models** (for example, a Fraud & Customer-Reactions model owned by Risk, coupled across a contract boundary to a Contact-Center Capacity & Ledger model owned by Operations).

However, decomposing an integrated cascade across model boundaries introduces five structural costs that our research program measures explicitly rather than hiding:
1. **Error Propagation & Nonlinear Boundary Amplification:** Small calibration biases in an upstream domain model can amplify nonlinearly when crossing into a downstream queuing domain operating near capacity ($\rho \to 1$).
2. **Hypothesis & Epistemic Assumption Incompatibility:** An upstream model may export observational regressions (`DependencyEvidenceNature.STATISTICAL_ASSOCIATION`) or gross decline counts (including blocked fraud), while a downstream capacity planner assumes interventional `do()` validity or legitimate false-decline cohorts.
3. **Temporal Synchronization Lag:** If the upstream model publishes batch state updates every $\Delta t_{\text{sync}} = 3$ steps while the downstream queue evolves every step, boundary staleness during a deployment shift delays staffing adaptation.
4. **Accumulated Uncertainty:** Naive addition of marginal prediction intervals across a boundary inflates uncertainty bands when covariance is ignored, or under-covers when shared exogenous shocks ($E_t$) are treated as independent.
5. **Specification & Maintenance Overhead ("Benefit per Unit Cost"):** Defining, versioning, and validating boundary contracts increases total specification complexity and runtime marshaling overhead.

### 1.2 Minimal Typed Composition Interface (Reusing Core Contracts)
Rather than inventing a parallel composition framework, `ewm_engine.experimental.domain_composition` reuses the core engine's existing contracts:
- **`DomainBoundaryContract`:** Wraps `ObservationSpec` (enforced by `ObservationFilter.observe()`), `exported_resource_ids`, `exported_context_keys` (projected via `ContextSnapshot.project()`), `required_evidence_nature` (`DependencyEvidenceNature`), `expected_step_cadence_hours`, and `unit_scale_assumptions`.
- **`DomainBoundaryExchange`:** Carries the leak-free `projected_observation_state` (`WorldState`), `projected_context`, `EpistemicState`, `MechanismProvenance`, `DependencyEvidenceNature`, `UncertaintyTaxonomyBreakdown`, `declared_units`, and a canonical SHA-256 `exchange_fingerprint`.
- **`validate_boundary_exchange()`:** Enforces zero leakage of internal upstream resources (`internal_fraud_rule_score_buffer`), exact semantic unit/cohort matching, and fail-closed `validate_dependency_for_query()` gating before the downstream submodel can step.

### 1.3 Measured Composition-Cost Findings (`seed=42, M=5`)

| Composition Cost Dimension | Monolithic Baseline (`FraudCascadeEnvironment`) | Composed Two-Model System (`FraudDeclinesCustomerSubmodel` $\to$ `CallCenterCapacitySubmodel`) | Measured Finding & Verdict |
| :--- | :--- | :--- | :--- |
| **1. Error Propagation Across Boundary** | Single-system internal state | **`+5.3%`** exported boundary arrival bias $\to$ **`+10.4%`** downstream wait-time error (**`1.96x`** nonlinear amplification; **`$4,462.58`** post-shift cash MAE) | **Confirmed (`C-N3`):** Queue nonlinearity near saturation amplifies small upstream calibration errors across the boundary by nearly $2\times$. |
| **2. Epistemic & Unit Assumption Incompatibility** | Shared internal definitions | Correlational `STATISTICAL_ASSOCIATION` export blocked for `INTERVENTIONAL_DO` (`True`); cohort unit mismatch blocked (`True`). If ungated, silent unit corruption costs **`$1,399.63`** | **Confirmed:** Without explicit `DependencyEvidenceNature` and unit-contract validation at the boundary, cohort mismatches silently corrupt downstream staffing. |
| **3. Temporal Synchronization Lag** | Synchronous 1-step state (`$0.00` regret) | 3-step batch boundary cadence incurs **`$4,435.34`** post-shift utility loss and **`1.00`** extra SLA breach step | **Confirmed (`C-N1`):** Asynchronous boundary cadence delays shift detection across departments, degrading decision quality during policy rollouts. |
| **4. Uncertainty Accumulation** | Coupled rollout (`17.27s` 90% wait interval width) | Naive boundary interval summation inflates 90% wait interval width to **`23.17s`** (**`+34.2%`** interval bloat) | **Confirmed:** Propagating uncertainty across boundaries without shared-noise coupling inflates interval widths by `+34.2%`. |
| **5. Specification & Maintenance Cost** | **`11`** specification components | **`16`** specification elements (`6` upstream + `5` downstream + `5` boundary contract = **`+45.5%`** spec overhead) for **`$0.00`** regret gain when synchronous | **Confirmed (`C-N2`):** When two domains can be co-located in memory, composition adds `+45.5%` specification overhead with zero decision-quality benefit. |

---

## Line 2 — Software-Engineering Ecosystem as a Fenced Second Domain (§9)

### 2.1 Specific Testable Hypothesis & Anti-Renaming Demarcation
- **What an EWM does NOT claim in Software Engineering:** A world model does **not** introduce dependency mapping to software engineering. Static call/import analysis (ASTs), build graphs (Bazel/Nx), distributed tracing (OpenTelemetry), and CI test runners already map dependencies—and as we verify in **`SWE-N1`** below, a static dependency graph is faster, simpler, and 100% accurate for compile-time signature or schema break detection.
- **What Static Graphs Miss (Visible by Construction):** Static dependency graphs do **not** capture **concurrency contention** (shared database connection pools saturating under burst load), **stateful accumulation** (retry storms, cache flushes), or **runtime configuration ($K_t$)** (feature-flag activation, retry multipliers, per-caller connection bulkheads).
- **Specific Testable Hypothesis (`H_SWE`):** *Under burst concurrency and runtime configuration shifts ($K_t$), does an integrated dynamic model of structural dependencies + shared resource contention + bitemporal incident telemetry improve **change-impact prediction, test selection, failure analysis, and deployment planning** over static dependency graphs and historical ML baselines?*

### 2.2 Controlled Synthetic Software-Ecosystem Topology (`SWEEcosystemEnvironment`)
We model a 6-component distributed architecture (`auth_lib`, `inventory_cache`, `orders_db`, `checkout_service`, `api_gateway`, `payment_worker`) undergoing a release of `auth_lib` (`v2.4 -> v2.5`):
- **Static Call/Import Graph:** `api_gateway -> checkout_service -> {auth_lib, inventory_cache, orders_db}`, while `payment_worker -> {orders_db}`. Crucially, **`payment_worker` has NO static dependency path to `auth_lib` or `checkout_service`**.
- **Dynamic Contention Mechanism:** Enabling `strict_jwt_flag=True` in $K_t$ adds $+14\text{ ms}$ verification latency in `auth_lib`. Under burst load (`850 RPS`) and `checkout_service`'s `3.0x` retry multiplier, held connections saturate the shared `orders_db` connection pool (`capacity = 100`), starving `payment_worker` (`p99 > 100 ms` SLO breach) unless a per-caller connection-pool bulkhead (`db_pool_cap_per_caller <= 60`) and bounded retries (`1.2x`) are configured.
- **Bitemporal CI & Incident Ledger (`BitemporalLedger.as_of(5.0)`):** Post-deployment canary DLQ alerts (`knowledge_time = 8.0`) and confirmed postmortem root-cause labels (`knowledge_time = 10.0`) are strictly withheld at pre-deployment decision time `t = 5.0` (`2` future-knowledge records withheld).

### 2.3 Measured 5-Rung Comparative Results Across the 4 SWE Tasks (`seed=42, M=5`)

| Rung / Baseline | Spec Components | Task 1: Change-Impact F1 | Task 1: Latency MAE (ms) | Caught `payment_worker` Contention? | Task 2: Test Selection Fault Recall | Task 3: Root-Cause Precision@1 | Task 4: Selected Rollout Plan | Task 4: Deployment Regret ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **`Rung 1`: Static Call/Import Dependency Graph** | `1` | `0.5714` | `90.85` | `False` | `0.50` | `0.00` | `plan_static_graph_scoped_canary` | `$60,900.00` |
| **`Rung 2`: Historical CI Co-Change ML (`GBT`)** | `2` | `0.3333` | `89.53` | `False` | `0.50` | `0.00` | `plan_naive_full_rollout_3x_retries` | `$59,700.00` |
| **`Rung 3`: ML + Static Graph Centrality Features** | `3` | `0.5714` | `88.64` | `False` | `0.75` | `0.00` | `plan_static_graph_scoped_canary` | `$60,900.00` |
| **`Rung 4`: Dynamic Simulator Without Config $K_t$ or Causal Gate** | `7` | `0.0000` | `81.55` | `False` | `0.75` | `0.00` | `plan_naive_full_rollout_3x_retries` | `$59,700.00` |
| **`Rung 5`: Experimental SWE World Model (`State + K_t + Gate`)** | `10` | **`1.0000`** | **`0.56`** | **`True`** | **`1.00`** | **`1.00`** | `plan_dynamic_pool_isolated_bounded_retries` | **`$0.00`** |

### 2.4 Single-Component Ablations (`SWE-Ablation 1`–`4`)

| Ablation ID | Removed Component from `Rung 5` | Impact F1 | Caught `payment_worker`? | Root-Cause Prec@1 | Deployment Regret ($) | Diagnostic Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`swe_ablation_1_no_concurrency_pool_state`** | Stateful DB connection-pool & retry contention | `0.5714` | `False` | `1.00` | `$60,900.00` | Reverts to static-graph reachability; misses shared DB pool starvation on `payment_worker`. |
| **`swe_ablation_2_no_runtime_config_context`** | Runtime config $K_t$ (`strict_jwt_flag`, `retry_multiplier`) | `0.0000` | `False` | `1.00` | `$59,700.00` | Assumes baseline flag/retry settings and fails to predict the flag-triggered retry storm. |
| **`swe_ablation_3_no_evidence_nature_gate`** | `DependencyEvidenceNature` causal gate on telemetry | `1.0000` | `True` | `0.00` | `$60,900.00` | Mistakes high correlation between `payment_worker` DLQ alerts and `orders_db` latency for a `payment_worker` bug. |
| **`swe_ablation_4_bitemporal_leakage_control`** | `BitemporalLedger.as_of(5.0)` point-in-time filter | — | — | — | — | Leaks `2` future postmortem/DLQ records (`knowledge_time = 8.0, 10.0`), creating illusory offline accuracy. |

### 2.5 Enumerated Negative Results & Domain-Transfer Limits (`SWE-N1`–`SWE-N3`)
1. **`SWE-N1` — Static Dependency Graph Wins Pure Compile/Schema API-Break Detection (*Confirmed*):**
   - For deterministic compile-time signature or protobuf schema breaks with no concurrency or runtime configuration interaction, **`Rung 1` (`Static Dependency Graph`)** achieves **`F1 = 1.000`** with **`1` specification component** (vs. `10` for `Rung 5`). An EWM is unnecessary and over-engineered for static dependency analysis.
2. **`SWE-N2` — Historical ML Beats EWM on Routine Unit-Test Selection Under Stationary Config (*Confirmed*):**
   - When runtime feature flags and concurrency load are stationary, **`Rung 3` (`GBT + Static Graph Centrality`)** achieves **`0.950` recall** on routine unit-test regressions vs. **`0.920` for `Rung 5`** at $3.3\times$ lower specification burden (`3` vs. `10` components).
3. **`SWE-N3` — Enterprise Stock-and-Flow Conservation Laws Do NOT Transfer to Discrete Logic Bugs (*Confirmed*):**
   - Unlike financial ledgers or physical inventory where linear stoichiometric conservation ($A \Delta x = 0$) holds via `ConservationProjectionLayer`, discrete software logic bugs (off-by-one errors, null dereferences, race conditions) do **not** obey conservation laws. The EWM architecture transfers **only** to bounded operational resources (connection pools, thread queues, rate limits, progressive delivery), **never** to semantic program verification.
