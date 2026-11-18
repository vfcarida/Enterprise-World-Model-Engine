# Step-by-Step Tutorial: Simulating a Microservice Incident Cascade (E7 SE Domain Pack)

**Domain Package**: `ewm_engine.domains.software_engineering` (`ewm_engine.domains.swe`)  
**Verification Suite**: [`tests/test_swe_dynamic_rca.py`](../../tests/test_swe_dynamic_rca.py)  
**Historical Legacy Tutorial**: [`docs/legacy/swe_tutorial_legacy.md`](../legacy/swe_tutorial_legacy.md)

---

## 1. Architectural Separation: Core Engine vs. SE Domain Pack

The **E7 Software Engineering (SE) Domain Pack** builds on top of the EWM core without polluting core dependencies:
- **Core Engine (`ewm_engine.core`)**: Provides domain-neutral state (`WorldState`, `StateEncoder`), retrieve-then-reason dynamics (`RetrieveAndReasonEngine`), and DoWhy-GCM causal mechanisms (`EngineCausalInterface`).
- **SE Domain Pack (`ewm_engine.domains.software_engineering`)**: Isolates `scipy.sparse` and `networkx` graph operations inside `DynamicRCASimulator` to model **DynaCausal** anomaly-conditioned resilience and **Enhanced Iterative Random Walk with Restart (eIRWR)** over microservice call graphs.

---

## 2. Mathematical Model: Why Static Call Graphs Fail Under Retry Storms

In a microservice chain $A \rightarrow B \rightarrow C$ (where caller $A$ invokes $B$, and $B$ invokes downstream database/payment service $C$), a latency spike at $C$ triggers timeout retries from $B$, which in turn exhausts thread pools in $A$.

`DynamicRCASimulator` models this cascade through three mechanisms:

1. **Forward Call Adjacency ($W$) & Backward Retry-Storm Edges ($\rho W^\top$)**:
   - $\rho \in (0, 1]$ (`rho`) is the **backward discount factor** governing how strongly downstream timeouts propagate upstream from callee $v$ to caller $u$.
2. **Anomaly-Conditioned Resilience ($\beta$, $R_{\text{base}}$)**:
   - Each node's resilience degrades exponentially with its normalized stress $\hat{s}_i^{(t)} \in [0, 1]$:
     $$R_i^{(t)} = R_{\text{base}} \cdot \exp\!\left(-\beta \, \hat{s}_i^{(t)}\right)$$
   - $\beta > 0$ (`beta`) is the **sensitivity parameter**: larger $\beta$ causes stressed nodes to lose resilience faster and shed retry load upstream.
3. **Strictly Row-Normalized eIRWR Propagation**:
   - At each step $t \le 100$, the sparse transition matrix $P^{(t)}$ is strictly row-normalized ($\sum_j P_{ij}^{(t)} = 1$) and updated via:
     $$s^{(t+1)} = (1 - \alpha) \left(s^{(t)} P^{(t)}\right) + \alpha \, v^{(0)}$$

---

## 3. Step-by-Step Executable Walkthrough

Copy and paste the following self-contained Python block directly into a Python REPL:

```python
import networkx as nx
import numpy as np
from ewm_engine.domains.software_engineering import DynamicRCASimulator

# Step 1: Define the microservice call graph (A calls B, B calls C)
call_graph = nx.DiGraph([("A", "B"), ("B", "C")])

# Step 2: Initialize the DynamicRCASimulator with sensitivity beta and backward discount rho
simulator = DynamicRCASimulator(
    call_graph,
    r_base=0.85,
    beta=2.5,
    rho=0.75,
    forward_weight=0.15,
    restart_alpha=0.30,
    max_iterations=50,
)

# Step 3: Inject a severe latency fault at downstream service C
result = simulator.simulate({"A": 0.0, "B": 0.0, "C": 10.0})

# Step 4: Verify upstream retry-storm propagation (C -> B -> A) and root-cause localization
assert result.converged is True
assert result.final_scores["C"] > result.final_scores["B"] > result.final_scores["A"] > 0.0
assert result.identified_origin == "C"

# Step 5: Verify strict row-normalization of the final transition matrix P^(T)
row_sums = np.sum(np.asarray(result.final_transition_matrix), axis=1)
assert np.allclose(row_sums, np.ones(3), atol=1e-12)
```

---

## 4. Multi-Service Mesh Comparison Against Static PageRank

In a 5-service mesh (`Gateway -> Auth/Order -> Payment/Database`) where an incident originates at `Payment` and causes secondary retry symptoms on `Order` and `Gateway`, static reverse PageRank drifts upstream to `Gateway`, whereas `DynamicRCASimulator` accurately localizes `Payment`:

```python
import networkx as nx
from ewm_engine.domains.software_engineering import DynamicRCASimulator

mesh_dag = nx.DiGraph(
    [
        ("Gateway", "Auth"),
        ("Gateway", "Order"),
        ("Order", "Payment"),
        ("Order", "Database"),
        ("Auth", "Database"),
    ]
)
mesh_sim = DynamicRCASimulator(mesh_dag, r_base=0.85, beta=3.0, rho=0.80)
mesh_result = mesh_sim.simulate(
    {
        "Gateway": 0.20,
        "Auth": 0.02,
        "Order": 0.45,
        "Payment": 1.00,
        "Database": 0.05,
    }
)

assert mesh_result.identified_origin == "Payment"
assert mesh_result.static_pagerank_top1 == "Gateway"
```
