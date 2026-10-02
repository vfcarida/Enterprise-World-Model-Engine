"""Minimal Warehouse Acceptance Fixture (AC-012).

Normative Specification:
  Warehouses: A (cap=150, inv=100), B (cap=150, inv=20)
  Demand at B: 50
  Horizon: 1
  Seed: 42

Normative Results:
  - Baseline (no transfer):
      served_demand = 20.0
      unserved_demand = 30.0
      final_inventory = {"warehouse_a": 100.0, "warehouse_b": 0.0}
      hard_constraint_violations = 0
  - Intervention (transfer 30 A->B before demand):
      served_demand = 50.0
      unserved_demand = 0.0
      final_inventory = {"warehouse_a": 70.0, "warehouse_b": 0.0}
      hard_constraint_violations = 0
  - Rejection (transfer 120):
      Action rejected, hard violation on 'available_inventory' constraint.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ewm_engine.core.actions import Action, Intervention
from ewm_engine.evaluation.comparison import ScenarioComparison, compare_scenarios
from ewm_engine.simulation.scenario import Scenario, ScheduledAction
from examples.minimal_warehouse.world import create_warehouse_world


def run_minimal_warehouse() -> ScenarioComparison:
    print("=" * 80)
    print("MINIMAL WAREHOUSE: NORMATIVE ACCEPTANCE SCENARIO (AC-012)")
    print("=" * 80)

    # 1. Build world from validated YAML specification
    world = create_warehouse_world()

    # 2. Simulate Baseline (No transfer)
    baseline_scenario = Scenario(
        scenario_id="warehouse-transfer-v1-baseline",
        name="Baseline (No Transfer)",
        horizon=1,
        samples=1,
        seed=42,
    )
    result_baseline = world.simulate(scenario=baseline_scenario)
    traj_baseline = result_baseline.trajectories[0]

    print("\n--- Baseline Results ---")
    print(f"Served Demand:   {traj_baseline.final_metric('served_demand'):.1f}")
    print(f"Unserved Demand: {traj_baseline.final_metric('unserved_demand'):.1f}")
    print(f"Final Inv A:     {traj_baseline.final_state.get_resource('warehouse_a').current:.1f}")
    print(f"Final Inv B:     {traj_baseline.final_state.get_resource('warehouse_b').current:.1f}")
    print(f"Hard Violations: {traj_baseline.hard_violations}")

    # 3. Simulate Intervention (Transfer 30 A -> B before demand)
    transfer_action = Action(
        id="transfer_30_a_to_b",
        type="transfer_resource",
        parameters={
            "source_resource": "warehouse_a",
            "target_resource": "warehouse_b",
            "quantity": 30.0,
        },
    )
    intervention_scenario = Scenario(
        scenario_id="warehouse-transfer-v1-intervention",
        name="Intervention (Transfer 30)",
        horizon=1,
        samples=1,
        seed=42,
        intervention=Intervention(
            id="transfer_30_intervention",
            description="Transfer 30 units from Warehouse A to Warehouse B before demand",
        ),
        scheduled_actions=(ScheduledAction(step=0, action=transfer_action),),
    )
    result_intervention = world.simulate(scenario=intervention_scenario)
    traj_intervention = result_intervention.trajectories[0]

    print("\n--- Intervention Results ---")
    print(f"Served Demand:   {traj_intervention.final_metric('served_demand'):.1f}")
    print(f"Unserved Demand: {traj_intervention.final_metric('unserved_demand'):.1f}")
    print(
        f"Final Inv A:     {traj_intervention.final_state.get_resource('warehouse_a').current:.1f}"
    )
    print(
        f"Final Inv B:     {traj_intervention.final_state.get_resource('warehouse_b').current:.1f}"
    )
    print(f"Hard Violations: {traj_intervention.hard_violations}")

    # 4. Demonstrate Over-Transfer Rejection (Transfer 120 exceeds source inventory of 100)
    over_transfer_action = Action(
        id="over_transfer_120",
        type="transfer_resource",
        parameters={
            "source_resource": "warehouse_a",
            "target_resource": "warehouse_b",
            "quantity": 120.0,
        },
    )
    rejection_scenario = Scenario(
        scenario_id="warehouse-transfer-v1-rejection",
        name="Rejected Action (Transfer 120)",
        horizon=1,
        samples=1,
        seed=42,
        scheduled_actions=(ScheduledAction(step=0, action=over_transfer_action),),
    )
    result_rejection = world.simulate(scenario=rejection_scenario)
    traj_rejection = result_rejection.trajectories[0]
    first_step = traj_rejection.steps[0]

    print("\n--- Over-Transfer Rejection Results ---")
    print(f"Proposed Actions Count: {len(first_step.actions_proposed)}")
    print(f"Accepted Actions Count: {len(first_step.actions_accepted)}")
    print(f"Violations Recorded:    {len(first_step.constraint_violations)}")
    for v in first_step.constraint_violations:
        print(f"  Violation: {v.constraint_id} (severity: {v.severity.value}) - {v.message}")

    # 5. Comparative Evaluation
    comparison = compare_scenarios(
        baseline=result_baseline,
        candidates=[result_intervention],
        metrics=[
            "served_demand",
            "unserved_demand",
            "resource_warehouse_a",
            "resource_warehouse_b",
        ],
    )
    print("\n" + comparison.summary_table() + "\n")
    return comparison


if __name__ == "__main__":
    run_minimal_warehouse()
