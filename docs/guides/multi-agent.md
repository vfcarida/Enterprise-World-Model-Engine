# Multi-Agent Simulation: Observation Views & Deterministic Adjudication

This guide demonstrates how to simulate decentralized, competing actors with role-specific observation views, action proposals, and deterministic constraint mediation in `ewm_engine.multiagent`.

---

## 1. Decentralized Multi-Agent Concepts

In enterprise environments, departments and business units operate with:
1. **Partial Observability:** Different roles only see subsets of entities and confidential fields.
2. **Simultaneous Proposals:** Agents act simultaneously at each time step.
3. **Deterministic Constraint Mediation:** Shared resources (budgets, inventory, server quotas) are finite. Conflicts are adjudicated using priority, timestamps, and capacity limits without stochastic LLM arbitration (ADR-004).

---

## 2. Role-Specific Observation Views

```python
from ewm_engine.core.state import WorldState
from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.multiagent import ActorObservationView

# Construct world state
state = WorldState(
    step=1,
    entities={
        "warehouse_1": Entity(id="warehouse_1", type="warehouse", attributes={"capacity": 500}),
        "hq_finances": Entity(id="hq_finances", type="finance", attributes={"cash": 1_000_000}),
    },
    resources={"stock": Resource(id="stock", current=250.0)},
)

# Logistics manager only sees warehouse_1, with confidential financial data masked
obs = ActorObservationView.from_world_state(
    state,
    actor_id="logistics_agent",
    visible_entity_ids={"warehouse_1"},
    masked_attributes={"hq_finances": ["cash"]},
)
print("Visible entities:", obs.entities.keys())
```

---

## 3. Simultaneous Action Mediation

When multiple agents propose actions that exceed resource pools or violate constraints, `ConstraintMediator` deterministically decides which actions succeed:

```python
from ewm_engine.core.actions import Action
from ewm_engine.multiagent import ActorActionProposal, ConstraintMediator

mediator = ConstraintMediator(resource_locks=True)

# Agent 1 and Agent 2 both propose spending from a shared budget
proposals = [
    ActorActionProposal(
        actor_id="dept_a",
        action=Action(id="a1", type="custom", parameters={"cost": {"budget": 60.0}}),
        priority=10,
    ),
    ActorActionProposal(
        actor_id="dept_b",
        action=Action(id="b1", type="custom", parameters={"cost": {"budget": 50.0}}),
        priority=5,
    ),
]

# State has only 70.0 budget available
adjudication = mediator.adjudicate(proposals, state, constraints=())
print("Accepted actions count:", len(adjudication.accepted_actions))
print("Rejected actions:", adjudication.rejected_actions)
```

---

## 4. Game-Theoretic Equilibria & Multi-Agent Gym (PettingZoo)

For 2-player strategic interactions, `NashpyGameSolver` computes pure/mixed Nash equilibria and evolutionary replicator dynamics:

```python
from ewm_engine.multiagent import NashpyGameSolver

# Prisoner's Dilemma payoff matrix
payoff_row = [[-1, -3], [0, -2]]
payoff_col = [[-1, 0], [-3, -2]]

solver = NashpyGameSolver(payoff_row, payoff_col)
equilibria = solver.solve_nash()
print("Equilibria found:", len(equilibria))
```

To train multi-agent reinforcement learning (MARL) policies, wrap any EWM `World` in a PettingZoo Parallel Environment:

```python
from ewm_engine.multiagent.adapters import PettingZooParallelAdapter

env = PettingZooParallelAdapter(
    world=world,
    agent_ids=["dept_a", "dept_b"],
    mediator=mediator,
)
obs, infos = env.reset()
```
