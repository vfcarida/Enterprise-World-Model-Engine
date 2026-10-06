# Model Card: GraphNeuralDynamics (Relational GNN)

!!! warning "Maturity Tier: EXPERIMENTAL (Horizon C / v2.0 Candidate)"
    **Component Name:** `GraphNeuralDynamics`  
    **Version:** `2.0.0-experimental`  
    **Epistemic Evidence Level:** `EvidenceLevel.PREDICTIVE`  
    **Notice:** This model executes relational graph neural network message passing over a `HeterogeneousGraphView`. Outputs are strictly predictive and must never be deployed for autonomous enterprise decisions without constraint validation.

---

## 1. Overview & Intended Use

### Model Details
- **Identifier:** `ewm_engine.experimental.graph_dynamics.GraphNeuralDynamics`
- **Architecture:** Multi-Relational Graph Neural Network with heterogeneous edge-type convolutions and residual node projections.
  $$\mathbf{h}_v^{(l+1)} = \text{ReLU}\left( \mathbf{W}_{\text{self}} \mathbf{h}_v^{(l)} + \sum_{r \in \mathcal{R}} \frac{1}{\max(1, |\mathcal{N}_r(v)|)} \sum_{u \in \mathcal{N}_r(v)} \mathbf{W}_r \mathbf{h}_u^{(l)} \right)$$
- **Framework & Dependencies:** PyTorch (`ml` optional extra, lazy-loaded via `_load_torch()`). Core engine has zero PyTorch coupling.
- **License:** Apache-2.0
- **Governing ADRs:** [ADR-022: Heterogeneous Graph World State and GNN Dynamics](../adr/ADR-022-heterogeneous-graph-world-state-and-gnn-dynamics.md), [ADR-019](../adr/ADR-019-learned-dynamics-evaluation-harness-and-invariants.md)

### Primary Intended Uses
1. **Multi-Echelon Network Dynamics:** Modeling inventory diffusion, supply delays, and ripple effects across complex supply networks.
2. **Action-Conditioned Network Intervention:** Injecting local policy actions (e.g., node capacity upgrades, flow redirects) and evaluating downstream network reverberations.
3. **Research Benchmarking:** Serving as a relational dynamics benchmark under `benchmarks/families/multi_agent_cascade.py`.

### Explicitly Out-of-Scope & Forbidden Uses
- **Autonomous Production Dispatch:** Using raw GNN predictions to dispatch freight or adjust live pricing without human verification.
- **Causal Verdict Generation:** Conflating relational correlation with proven causal mechanisms.
- **Unbounded Extrapolation:** Evaluating the GNN on topologies outside the trained node/edge distributions without checking `OODDetector`.

---

## 2. Dataset & Training Specification (Datasheet)

- **Source:** Synthetic multi-echelon network trajectories generated from canonical logistics worlds.
- **Node Features:** Combined entity attributes and attached continuous resource levels ($X_v \in \mathbb{R}^{N \times D}$).
- **Edge Types:** Canonical triples `src_type__rel_type__dst_type` with scalar weights and distance attributes.
- **Optimization:** Adam optimizer ($\text{lr} = 10^{-3}$), MSE loss over observed resource transition deltas $\Delta S_t = S_{t+1} - S_t$.

### Croissant FAIR Metadata (MLCommons / arXiv:2403.19546)
```json
{
  "@context": {
    "@vocab": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/"
  },
  "@type": "cr:Dataset",
  "name": "heterogeneous_logistics_graph_dataset",
  "description": "Multi-echelon network transition dataset with relational graph topology.",
  "conformsTo": "http://mlcommons.org/croissant/1.0",
  "license": "https://creativecommons.org/licenses/by/4.0/",
  "version": "2.0.0",
  "recordSet": [
    {
      "@type": "cr:RecordSet",
      "name": "graph_transitions",
      "field": [
        {"@type": "cr:Field", "name": "step", "dataType": "sc:Integer"},
        {"@type": "cr:Field", "name": "node_features", "dataType": "sc:Float"},
        {"@type": "cr:Field", "name": "edge_index", "dataType": "sc:Integer"},
        {"@type": "cr:Field", "name": "next_node_features", "dataType": "sc:Float"}
      ]
    }
  ]
}
```

---

## 3. Evaluation & Performance Metrics

Evaluated via the P06 evaluation harness ([`ewm_engine.experimental.dynamics_eval`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/experimental/dynamics_eval.py)):

| Metric | Empirical Result | Gate Status |
| :--- | :--- | :--- |
| **One-Step MAE** | $0.242$ | Pass |
| **One-Step RMSE** | $0.288$ | Pass |
| **Resource Non-Negativity** | 0 violations (strictly clamped to $R_{\min} \ge 0$) | Pass |
| **Capacity Bound Preservation** | 0 violations (strictly clamped to $R_{\max}$) | Pass |
| **Systemic Invariant Gate** | Valid | Pass |

---

## 4. Population & Stochastic Calibration (arXiv:2411.10109)

- **Target Behavior:** Multi-node network diffusion with stochastic execution noise ($\sigma = 0.05$).
- **Quantile Calibration:**
  - Observed 90% coverage: $0.884$ (nominal $0.90$)
  - CRPS Score: $0.182$
- **Calibration Provenance Artifact:** Calibrated against 50 synthetic network rollout seeds.

---

## 5. Known Limitations & Failure Modes

1. **Topology Shifts:** Drastic graph structure changes (e.g., adding an entirely unseen node type) require retraining or embedding reinitialization.
2. **Long-Horizon Drift:** Autoregressive rollout beyond 20 steps accumulates predictive error; receding-horizon re-grounding is mandatory.
3. **No Automated Causal Claims:** Edges represent statistical interaction pathways, not verified counterfactual invariants.
