# Reinforcement Learning with Gymnasium

The `EnterpriseGymEnv` adapter bridges an EWM Engine [`World`](../api/reference.md) to the standard [Gymnasium (OpenAI Gym)](https://gymnasium.farama.org/) interface (`gymnasium.Env`).

This enables modern reinforcement learning libraries (such as Stable-Baselines3, CleanRL, or Ray RLlib) to train policies directly inside enterprise simulation environments with **first-class, phase-aware constraint enforcement**.

---

## Key Features

- **Standard API**: Implements `reset(seed=..., options=...)` and `step(action)`.
- **Vectorized Observations**: Automatically maps numeric bounded resources into continuous observation spaces (`spaces.Box`), or accepts a custom `observation_fn: Callable[[WorldState], np.ndarray]`.
- **Action Spaces**: Supports discrete action indexes (`spaces.Discrete`) mapped to concrete `Action` definitions, or continuous parameter spaces mapped via a custom `action_mapping` function.
- **Constraint-Aware Rewards**: Incorporates hard and soft constraint violations into policy feedback:
  - If an action violates a `HARD` pre-action constraint, the action is rejected and a violation penalty is subtracted from reward.
  - If configured with `terminate_on_hard_violation=True`, encountering an invalid state terminates the episode with a structured violation report in `info`.

---

## Example: Training a Policy on Warehouse Logistics

```python
from ewm_engine import Action, Resource, World, WorldState
from ewm_engine.constraints.standard import ActionTransferAvailabilityConstraint, ResourceCapacityConstraint
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.integrations.gym import EnterpriseGymEnv

# 1. Define the World State & Rules
state = WorldState(
    resources=[
        Resource(id="stock_north", current=100.0, min_value=0.0, max_value=200.0),
        Resource(id="stock_south", current=20.0, min_value=0.0, max_value=200.0),
    ]
)
world = World(
    state=state,
    dynamics=DeterministicTransferDynamics(),
    constraints=[
        ResourceCapacityConstraint(resource_id="stock_north"),
        ResourceCapacityConstraint(resource_id="stock_south"),
        ActionTransferAvailabilityConstraint(),
    ],
)

# 2. Define Candidate Actions
candidate_actions = [
    Action(id="noop", type="no_operation"),
    Action(
        id="transfer_10",
        type="transfer_resource",
        parameters={"source_resource": "stock_north", "target_resource": "stock_south", "quantity": 10.0},
    ),
    Action(
        id="transfer_30",
        type="transfer_resource",
        parameters={"source_resource": "stock_north", "target_resource": "stock_south", "quantity": 30.0},
    ),
]

# 3. Create the Gym Environment
env = EnterpriseGymEnv(
    world=world,
    max_steps=20,
    action_mapping=candidate_actions,
    violation_penalty=50.0,
    terminate_on_hard_violation=True,
)

# 4. Standard Gym Episode Loop
obs, info = env.reset(seed=42)
for step in range(10):
    # Select action (e.g. from an RL policy or heuristic)
    action_idx = 1  # transfer_10
    obs, reward, terminated, truncated, info = env.step(action_idx)
    
    if terminated or truncated:
        break

print("Final Observation (Resources):", obs)
print("Violations Encountered:", info["violations_count"])
```
