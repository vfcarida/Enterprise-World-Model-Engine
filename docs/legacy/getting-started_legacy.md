# Getting Started with EWM Engine

Get up and running with the **Enterprise World Model Engine** in 5 minutes.

---

## Installation

### Baseline Installation (Zero Heavy ML / No Cloud Dependencies)
```bash
pip install ewm-engine
```

### With Optional Extensions
```bash
# Graph inspection & NetworkX export
pip install "ewm-engine[graphs]"

# Learned dynamics & neural transitions
pip install "ewm-engine[ml]"

# Documentation tools
pip install "ewm-engine[docs]"

# Full development & test suite
pip install "ewm-engine[all]"
```

---

## 60-Second Minimal Example

Try it directly in your browser without installing: **[Open WASM Playground](playground.md)** · **[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vfcarida/Enterprise-World-Model-Engine/blob/main/notebooks/quickstart.ipynb)**

The following self-contained script initializes a world state with bounded resources, simulates baseline dynamics, branches an interventional scenario, and outputs counterfactual deltas in **12 lines of code**:

```python
import ewm_engine as ewm
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics

# 1. Initialize world state with bounded resources
world = ewm.World(
    state=ewm.WorldState(
        resources=[
            ewm.Resource(id="stock_a", current=100.0, max_value=200.0),
            ewm.Resource(id="stock_b", current=20.0, max_value=200.0),
        ]
    ),
    dynamics=DeterministicTransferDynamics(),
)
# 2. Simulate baseline vs counterfactual policy branch
baseline = world.simulate(horizon=5, seed=42)
counterfactual = world.branch().simulate(
    actions=[
        ewm.Action(
            type="transfer_resource",
            parameters={
                "quantity": 30.0,
                "source_resource": "stock_a",
                "target_resource": "stock_b",
            },
        )
    ],
    horizon=5,
    seed=42,
)
# 3. Evaluate counterfactual delta across baseline and policy branch
diff = ewm.compare_scenarios(
    baseline, [counterfactual], metrics=["resource_stock_a", "resource_stock_b"]
)
print(diff.summary_table())
```

### Execution Output

```text
=== Scenario Comparison (Baseline: Baseline) ===
---------------------------------------------------------------------------------------------------------------------------
Metric                    | Scenario             | Mean (Std)         | p50 [p05, p95]       | Delta vs Base [CI] (*sig)   
---------------------------------------------------------------------------------------------------------------------------
resource_stock_a          | Baseline             | 100.00 (+/-0.00)   | 100.00 [100.00, 100.00] | -                           
                          | Counterfactual       | 70.00 (+/-0.00)    | 70.00 [70.00, 70.00] | -30.00 (-30.0%) [-30.00, -30.00] (det)
---------------------------------------------------------------------------------------------------------------------------
resource_stock_b          | Baseline             | 20.00 (+/-0.00)    | 20.00 [20.00, 20.00] | -                           
                          | Counterfactual       | 50.00 (+/-0.00)    | 50.00 [50.00, 50.00] | +30.00 (+150.0%) [+30.00, +30.00] (det)
---------------------------------------------------------------------------------------------------------------------------
```

---

## Running from the CLI

Run built-in reference simulations or execute custom declarative models directly from your terminal:

```bash
# Run minimal two-warehouse inventory balancing
ewm example minimal

# Run flagship CivicFlow flood response research simulation
ewm example civicflow

# Validate a declarative WorldSpec YAML or JSON file without executing code
ewm validate path/to/world_spec.yaml

# Run simulation directly from a declarative spec with custom horizon, samples, and output
ewm run path/to/world_spec.yaml --horizon 15 --samples 10 --seed 42 --out results.json

# List available committed JSON schemas
ewm schema list

# Inspect a specific schema definition
ewm schema show world-state
```
