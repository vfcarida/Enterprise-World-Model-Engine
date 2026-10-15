# Recipe: Auditing Counterfactual Invariance via Mermaid Trace Diffs

> **Task**: Perform a Pearlian counterfactual intervention and generate an inspectable visual Mermaid trace diff to prove which subgraphs remained invariant vs. which diverged.

---

## 1. The Recipe Code

```python
from ewm_engine.core.actions import Intervention
from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.provenance.trace import TraceDiff
from ewm_engine.simulation.scenario import Scenario


def execute_counterfactual_audit_recipe() -> None:
    # 1. Build Minimal World
    state = WorldState(
        entities=[Entity(id="dept_a"), Entity(id="dept_b"), Entity(id="treasury")],
        resources=[
            Resource(id="stock_a", current=15.0, min_value=0.0),
            Resource(id="stock_b", current=85.0, min_value=0.0),
            Resource(id="treasury_cash", current=10000.0, min_value=0.0),
        ],
    )
    world = World(state=state, dynamics=DeterministicTransferDynamics())

    # 2. Simulate Factual Baseline Episode
    scenario = Scenario(scenario_id="factual-baseline", horizon=2, samples=1, seed=42)
    factual_result = world.simulate(scenario)
    factual_traj = factual_result.trajectories[0]

    # 3. Perform Counterfactual Graph Surgery do(stock_a = 50.0)
    cf_intervention = Intervention(
        target="stock_a",
        value=50.0,
        step=1,
        description="Emergency inventory injection to dept_a",
    )
    cf_result = world.counterfactual(factual_traj, cf_intervention)

    # 4. Generate Mermaid Visual Trace Diff
    diff = TraceDiff.compute(
        factual_trace=factual_traj.provenance,
        counterfactual_trace=cf_result.counterfactual_trajectory.provenance,
    )
    mermaid_diagram = diff.to_mermaid()

    print("==================================================")
    print("COUNTERFACTUAL AUDIT TRACE DIFF")
    print("==================================================")
    print(f"Evidence Level: {cf_result.evidence_level}")
    print(f"Disclaimer:     {cf_result.epistemic_disclaimer}")
    print("\nRendered Mermaid Diagram:\n")
    print(mermaid_diagram)


if __name__ == "__main__":
    execute_counterfactual_audit_recipe()
```

---

## 2. Interpreting the Mermaid Diff
* **Invariant Nodes (Grey):** Graph entities (such as `treasury_cash`) that were mathematically unaffected by the intervention remain shaded in grey, confirming no unintended side effects.
* **Divergent Nodes (Red/Dashed):** Entities whose trajectory diverged directly as a result of the policy change.
* **Audit Ready:** The resulting Mermaid diagram can be pasted directly into markdown reports or viewed in GitHub / IDE previews.
