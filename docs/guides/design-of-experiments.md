# Design of Experiments (DoE) & Parameter Sweeps

This guide demonstrates how to declare multi-dimensional parameter spaces, generate experimental designs (full factorial, One-At-a-Time, Latin Hypercube, and Halton sequences), execute reproducible simulation sweeps, and inspect sensitivity tornado diagrams using `ewm_engine.experimentation`.

---

## 1. Declaring a Parameter Space

Define continuous, integer, and categorical parameter dimensions targeted at resources, memory, or scenario attributes:

```python
from ewm_engine.experimentation import ParameterDef, ParameterSpace

space = ParameterSpace(
    name="WarehouseOperationsSpace",
    parameters=(
        ParameterDef(
            name="buffer_capacity",
            type="continuous",
            bounds=(50.0, 300.0),
            default=150.0,
            target="resource",
            target_id="stock",
            description="Warehouse maximum stock buffer.",
        ),
        ParameterDef(
            name="reorder_threshold",
            type="integer",
            bounds=(10, 50),
            default=25,
            target="memory",
            target_id="reorder_level",
            description="Inventory level triggering purchase orders.",
        ),
        ParameterDef(
            name="shipping_tier",
            type="categorical",
            categories=("standard", "expedited", "freight"),
            default="standard",
            target="context",
            target_id="tier",
            description="Transportation service tier.",
        ),
    ),
)

# Every space has a deterministic canonical cryptographic hash
print(f"Space Hash: {space.space_hash}")
```

---

## 2. Generating Experimental Designs

The core engine provides four zero-dependency design generators:

```python
from ewm_engine.experimentation import (
    generate_full_factorial,
    generate_halton,
    generate_lhs,
    generate_oat,
)

# 1. Full Factorial (grid sweep)
grid_points = generate_full_factorial(space, levels_per_param=3)

# 2. One-At-a-Time (OAT) marginal perturbation around baseline
oat_points = generate_oat(space, steps_per_param=2)

# 3. Latin Hypercube Sampling (LHS - stratified quasi-random)
lhs_points = generate_lhs(space, n_samples=20, seed=42)

# 4. Low-Discrepancy Halton Sequences (deterministic quasi-random)
halton_points = generate_halton(space, n_samples=20)
```

---

## 3. Executing a Reproducible Parameter Sweep

Execute the sweep using `run_doe_sweep`. Each design point receives an independent, reproducible seed derived deterministically via `np.random.SeedSequence(seed).spawn()`:

```python
from ewm_engine.core.world import World
from ewm_engine.core.state import WorldState
from ewm_engine.core.resources import Resource
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.experimentation import run_doe_sweep

# Base world and scenario
world = World(initial_state=WorldState(resources=[Resource(id="stock", current=150.0)]))
scenario = Scenario(scenario_id="ops_sweep", horizon=10, samples=1, seed=42)

# Run sweep over Latin Hypercube design points
sweep_result = run_doe_sweep(
    world=world,
    scenario=scenario,
    space=space,
    design_points=lhs_points,
    design_type="lhs",
    seed=100,
)

print(f"Evaluated {sweep_result.point_count} design points.")
print(f"Sweep Provenance Fingerprint: {sweep_result.fingerprint}")

# Results table is in tidy dictionary format
first_row = sweep_result.results_table[0]
print(f"Sample Row: Run ID {first_row['run_id']}, Final Stock: {first_row['resource_stock']}")
```

---

## 4. Tornado Diagram Sensitivity Rankings

Tornado diagrams rank input parameters by the absolute magnitude of their effect on a target metric ($|y_{\text{high}} - y_{\text{low}}|$):

```python
# Inspect tornado data computed automatically for the 'stock' resource
tornado = sweep_result.tornado_data["stock"]

print(f"Metric: {tornado.metric_name} (Baseline: {tornado.baseline_metric:.2f})")
print("Parameter Sensitivity Rankings:")
for bar in tornado.bars:
    print(
        f"  - {bar.parameter_name:<20}: Swing = {bar.swing:.2f} "
        f"[Low: {bar.low_metric:.2f} @ {bar.low_value} -> High: {bar.high_metric:.2f} @ {bar.high_value}]"
    )
```
