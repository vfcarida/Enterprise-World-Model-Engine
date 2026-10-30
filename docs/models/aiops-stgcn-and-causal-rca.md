# Model Card: AIOps STGCN Cascade Predictor & Causal RCA

**Author:** Vinícius Caridá  
**Module:** `ewm_engine.aiops` (`STGCNEncoder`, `CascadeFailurePredictor`, `AIOpsWorldModel`, `CausalRCA`)  
**License:** Apache-2.0

---

## 1. Model Details

- **STGCN & Cascade Predictor (`ewm_engine.aiops.models`):**
  - Combines `EvolveGCN` spatial message passing over microservice call graphs with a 2-layer temporal `GRU` and bidirectional call-graph backpressure propagation.
  - Maps 1-minute resolution `(B, T, N, 5)` telemetry windows (`cpu`, `memory`, `error_rate`, `latency_p95`, `active_connections`) into latent operational states $S_t$, per-service failure probabilities in $[0, 1]$, and advance warning lead times in $[30, 60]$ minutes.
- **Causal Root Cause Analysis (`ewm_engine.aiops.rca.CausalRCA`):**
  - Learns directed causal graphs (`nx.DiGraph`) via constraint-based PC conditional independence pruning (Fisher's z-transform partial correlation tests) combined with temporal lead-lag orientation and call-graph priors.
  - Isolates upstream root-cause microservices from downstream cascading symptoms using Pearl's `do(fix(service))` interventional backdoor/frontdoor effect estimation.

## 2. Intended Use

- Proactive detection of cascading microservice failures (`memory_leak`, `bad_deployment`, `traffic_spike`) 30–60 minutes prior to user-facing SLO breach.
- Automated root cause isolation feeding `RemediationGenerator`, `OPAPolicyValidator`, and `ChaosMeshHarness` shadow verification.

## 3. Safety & Policy Guardrails

- All generated Kubernetes `Deployment` remediation manifests are gated by `opa_policies/remediation_policy.rego` (`PodDisruptionBudget` compliance, blast-radius cap, and image tag hygiene).
- `ChaosMeshHarness` strictly prohibits fault injection into `production` or `prod` namespaces, requiring validation in `shadow` or `canary` environments first.
