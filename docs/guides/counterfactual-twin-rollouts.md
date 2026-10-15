# How-To Guide: Simulating Counterfactuals via Twin-Rollout Noise Coupling

> **Goal**: Evaluate a Pearlian counterfactual intervention on an existing factual episode while keeping exogenous noise perfectly coupled between worlds.

---

## 1. Overview & Prerequisite Concepts

Under Pearl’s Structural Causal Model (SCM) framework ([ADR-040](../adr/ADR-040-twin-rollout-counterfactual-scm-kernel.md)), counterfactual reasoning requires three formal steps:
1. **Abduction:** Infer the exogenous noise variables $U$ from the observed factual episode.
2. **Action:** Perform graph surgery $do(X=x')$ by replacing the structural equation of target $X$.
3. **Prediction:** Re-run the dynamical equations using the updated graph and the abducted noise $U$.

In discrete-event and stochastic organizational simulations, step 1 and 3 are operationalized via **Twin-Rollout noise coupling**: the factual and counterfactual simulations consume the exact same sequence of pseudo-random draws.

---

## 2. Step-by-Step Implementation

### Step 1: Execute the Factual Simulation
First, run the baseline scenario and obtain the factual trajectory:

```python
from ewm_engine.core.state import WorldState
from ewm_engine.core.resources import Resource
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.simulation.scenario import Scenario

# Set up world
state = WorldState(
    resources=[
        Resource(id="hub_a", current=20.0, min_value=0.0),
        Resource(id="hub_b", current=100.0, min_value=0.0),
    ]
)
world = World(state=state, dynamics=DeterministicTransferDynamics())

# Run factual scenario
scenario = Scenario(scenario_id="factual", horizon=3, samples=1, seed=42)
factual_result = world.simulate(scenario)
factual_traj = factual_result.trajectories[0]
```

### Step 2: Formulate the Counterfactual Intervention
Define the target resource and the intervention value:

```python
from ewm_engine.core.actions import Intervention

intervention = Intervention(
    target="hub_a",
    value=50.0,
    step=1,
    description="Emergency replenishment of hub_a",
)
```

### Step 3: Execute the Coupled Counterfactual
Pass the factual trajectory and the intervention to `world.counterfactual()`:

```python
cf_result = world.counterfactual(factual_traj, intervention)

cf_traj = cf_result.counterfactual_trajectory
print(f"Factual Ending Stock Hub A:        {factual_traj.final_metric('hub_a')}")
print(f"Counterfactual Ending Stock Hub A: {cf_traj.final_metric('hub_a')}")
print(f"Evidence Level:                    {cf_result.evidence_level}")
print(f"Epistemic Disclaimer:              {cf_result.epistemic_disclaimer}")
```

---

## 3. Comparing Provenance Traces
Inspect how the intervention altered downstream state transitions:

```python
from ewm_engine.provenance.trace import TraceDiff

diff = TraceDiff.compute(
    factual_trace=factual_traj.provenance,
    counterfactual_trace=cf_traj.provenance,
)
print(diff.summary())
```
