# Model Card: Continuous-Time Temporal Graph Neural Network (EvolveGCN)

**Author:** Vinícius Caridá  
**Module:** `ewm_engine.graph` (`ContinuousGraphRegistry`, `GCNConv`, `EvolveGCNLayer`, `EvolveGCN`, `EvolveGCNTrainer`)  
**Maturity Tier:** **Experimental (`Synthetic Research Prototype`)**  
**License:** Apache-2.0

> [!WARNING]
> **Scope & Maturity Notice (`EWM-007` Audit):** `ewm_engine.graph` is an **Experimental** temporal GNN research module verified on synthetic in-memory graph topologies. For **Supported (`Stable 1.x`)** deterministic DAG dependency enforcement ($O(V+E)$ cycle detection, bounded graph capacity $\le 10{,}000$ nodes, and topological merge gating), use [`ewm_engine.domains.swe`](../../domains/swe/README.md) (`EWM-005`).

---

## 1. Model Details

- **Architecture:** EvolveGCN (Pareja et al., *AAAI 2020*) evolving Graph Convolutional Network weight matrices $W_t^{(l)}$ across discrete or continuous-time topology snapshots via recurrent `GRUCell` or `LSTMCell` units.
- **Memory Infrastructure:** Paired with `ContinuousGraphRegistry` for $O(1)$ in-memory node/edge updates (`>10,000` edge updates/sec) without static graph database traversal bottlenecks.
- **Inputs:**
  - Node feature matrix $X_t \in \mathbb{R}^{N_t \times F}$ (financial equity features or microservice telemetry features).
  - Directed edge index $E_t \in \mathbb{N}^{2 \times |\mathcal{E}_t|}$.
- **Outputs:**
  - Contextualized temporal-structural node embeddings $H_t \in \mathbb{R}^{N_t \times D}$.

## 2. Intended Use

- Dynamic topology representation for both the Financial World Model (supply-chain and sector correlation networks) and the AIOps World Model (microservice distributed tracing call graphs).
- Link prediction and multi-hop contagion propagation under node additions, pod restarts, and shifting edge weights.

## 3. Performance & Verification

- Processes `10,000` streaming edge updates in `<20ms` on CPU.
- Forward pass over `1,000` nodes completes in `<10ms` on CPU.
- Verified for deterministic rollout reproducibility after `reset_hidden_states()`.
