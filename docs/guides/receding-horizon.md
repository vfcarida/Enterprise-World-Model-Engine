# Guide: Receding-Horizon Control & Online Re-Grounding

This guide demonstrates how to configure and deploy receding-horizon controllers (`RecedingHorizonSimulator` and `PlanningActor`) using the pluggable planning layer in `ewm_engine.experimental.planning`.

---

## 1. Quickstart: Receding-Horizon Simulation

The simplest workflow uses `RecedingHorizonSimulator` to run closed-loop simulation over competing policy candidates:

```python
from ewm_engine.core.actions import Action
from ewm_engine.core.world import World
from ewm_engine.experimental.planning import PlanningCandidate, ExpectedObjectiveScorer
from ewm_engine.simulation.mpc import RecedingHorizonSimulator

# 1. Define candidate actions
candidates = [
    PlanningCandidate.from_action(
        Action(id="c_idle", type="restock", parameters={"quantity": 0.0}), id="idle"
    ),
    PlanningCandidate.from_action(
        Action(id="c_order", type="restock", parameters={"quantity": 25.0}), id="order_25"
    ),
]

# 2. Configure the receding-horizon controller
simulator = RecedingHorizonSimulator(
    lookahead_horizon=3,  # Short lookahead window
    samples_per_candidate=5,  # Monte Carlo rollouts per candidate
    objective_metric="resource:unserved_demand",
    minimize=True,  # Minimize customer stockouts
    seed=42,
)

# 3. Execute closed-loop re-grounded simulation
result = simulator.run(
    world=world,
    total_steps=10,
    candidates=candidates,
)

# 4. Inspect final state and decision audit trail
print(f"Final Unserved Demand: {result.final_state.get_resource('unserved_demand').current}")
for record in result.decision_history:
    print(
        f"Step {record.step}: Selected {record.selected_candidate} with scores {record.candidate_scores}"
    )
```

---

## 2. Choosing and Configuring Rollout Scorers

By default, `RecedingHorizonSimulator` uses `ExpectedObjectiveScorer`. You can substitute any of the built-in scorers to reflect specific risk tolerances and operational priorities:

### Risk-Averse Planning with CVaR
To protect against tail disaster (e.g., worst 10% cash burn or stockouts):

```python
from ewm_engine.experimental.planning import CVaRScorer

# Penalize the worst 10% tail outcomes when minimizing cost
cvar_scorer = CVaRScorer(
    metric="resource:unserved_demand",
    alpha=0.10,
    minimize=True,
)

simulator = RecedingHorizonSimulator(
    lookahead_horizon=4,
    samples_per_candidate=10,
    scorer=cvar_scorer,
    seed=101,
)
```

### Penalizing Constraint Violations
When candidate actions risk breaching soft or hard organizational boundaries:

```python
from ewm_engine.experimental.planning import ConstraintPenalizedScorer

scorer = ConstraintPenalizedScorer(
    base_metric="resource:profit",
    violation_penalty_weight=50.0,  # Penalty per constraint violation
    invalid_trajectory_penalty=200.0,  # Severe penalty for invalid state transitions
    minimize=False,  # Maximize profit
)
```

### Uncertainty and Variance Penalization
To encourage robust decisions with narrow outcome spreads:

```python
from ewm_engine.experimental.planning import UncertaintyPenalizedScorer

# Score = E[profit] - 1.5 * StdDev(profit)
uncertainty_scorer = UncertaintyPenalizedScorer(
    metric="resource:profit",
    uncertainty_weight=1.5,
    minimize=False,
)
```

---

## 3. Autonomous Agents: Using `PlanningActor`

To deploy a model-predictive planner as an autonomous agent interacting with other actors inside a `World`:

```python
from collections.abc import Sequence
from ewm_engine.actors.base import ActorContext
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.experimental.planning import (
    PlanningActor,
    PlanningCandidate,
    RolloutPlanner,
    ExpectedObjectiveScorer,
)
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario


# Define candidate generation logic given current state
def candidate_generator(state: WorldState, ctx: ActorContext) -> Sequence[PlanningCandidate]:
    current_inventory = state.get_resource("inventory").current
    if current_inventory < 30.0:
        return [
            PlanningCandidate.from_action(
                Action(id="reorder", type="order", parameters={"qty": 20.0}), id="reorder"
            ),
            PlanningCandidate.from_action(
                Action(id="wait", type="order", parameters={"qty": 0.0}), id="wait"
            ),
        ]
    return [
        PlanningCandidate.from_action(
            Action(id="wait", type="order", parameters={"qty": 0.0}), id="wait"
        )
    ]


# Wrap in PlanningActor
planning_agent = PlanningActor(
    actor_id="inventory_mpc_agent",
    world_model=world,
    candidate_generator=candidate_generator,
    scorer=ExpectedObjectiveScorer(metric="resource:unserved_demand", minimize=True),
    lookahead_horizon=3,
    samples_per_candidate=5,
)

# Add to world and simulate
world.add_actor(planning_agent)
engine = SimulationEngine()
res = engine.run(world=world, scenario=Scenario(name="MultiAgentPlanning", horizon=10, samples=1))
```

---

## 4. Auditing Decision Provenance

Every planning execution generates structured decision provenance (`PlanningDecision`):

```python
for p_decision in result.planning_decisions:
    print(f"Decision ID: {p_decision.decision_id}")
    print(f"Step: {p_decision.step}")
    print(f"Chosen Candidate: {p_decision.chosen_candidate_id}")
    print(f"Lookahead Seed: {p_decision.seed}")
    print(f"Evaluated Scores: {p_decision.candidate_scores}")
    print(f"Pre-action State Hash: {p_decision.state_hash_before}")
    print(f"Scorer Details: {p_decision.score_details}")

    # Full JSON serialization for external governance systems
    json_audit_record = p_decision.model_dump_json(indent=2)
```
