"""Acceptance test (AC-012): Minimal Warehouse Normative Specification."""

from __future__ import annotations

import pytest

from ewm_engine.constraints.results import ConstraintSeverity
from ewm_engine.core.actions import Action, Intervention
from ewm_engine.simulation.scenario import Scenario, ScheduledAction
from examples.minimal_warehouse.world import create_warehouse_world, load_warehouse_spec


@pytest.mark.example
def test_minimal_warehouse_acceptance_scenario() -> None:
    """AC-012: Minimal warehouse normative values, intervention, rejection, and provenance."""
    # 1. Spec and World construction from YAML
    spec = load_warehouse_spec()
    assert spec.world.name == "minimal_warehouse"
    assert len(spec.entities) == 2
    assert len(spec.resources) == 2

    world = create_warehouse_world()
    initial_fingerprint = world.initial_state.fingerprint

    # 2. Baseline Rollout (No transfer)
    baseline_scenario = Scenario(
        scenario_id="warehouse-transfer-v1",
        name="Baseline",
        horizon=1,
        samples=1,
        seed=42,
    )
    result_baseline = world.simulate(scenario=baseline_scenario)
    assert len(result_baseline.trajectories) == 1
    traj_base = result_baseline.trajectories[0]

    # Normative baseline assertions:
    # served_demand: 20.0, unserved_demand: 30.0, final_inv: A=100.0, B=0.0, hard_violations: 0
    assert traj_base.final_metric("served_demand") == 20.0
    assert traj_base.final_metric("unserved_demand") == 30.0
    assert traj_base.final_state.get_resource("warehouse_a").current == 100.0
    assert traj_base.final_state.get_resource("warehouse_b").current == 0.0
    assert traj_base.hard_violations == 0

    # 3. Counterfactual Intervention (Transfer 30 from A to B before demand)
    transfer_action = Action(
        id="transfer_30",
        type="transfer_resource",
        parameters={
            "source_resource": "warehouse_a",
            "target_resource": "warehouse_b",
            "quantity": 30.0,
        },
    )
    intervention_scenario = Scenario(
        scenario_id="warehouse-transfer-v1-intervention",
        name="Intervention",
        horizon=1,
        samples=1,
        seed=42,
        intervention=Intervention(
            id="transfer_30_intervention",
            description="Transfer 30 units from A to B before demand",
        ),
        scheduled_actions=(ScheduledAction(step=0, action=transfer_action),),
    )
    result_transfer = world.simulate(scenario=intervention_scenario)
    traj_transfer = result_transfer.trajectories[0]

    # Normative intervention assertions:
    # served_demand: 50.0, unserved_demand: 0.0, final_inv: A=70.0, B=0.0, hard_violations: 0
    assert traj_transfer.final_metric("served_demand") == 50.0
    assert traj_transfer.final_metric("unserved_demand") == 0.0
    assert traj_transfer.final_state.get_resource("warehouse_a").current == 70.0
    assert traj_transfer.final_state.get_resource("warehouse_b").current == 0.0
    assert traj_transfer.hard_violations == 0

    # 4. Provenance Fingerprint Identity (exact same world baseline state)
    assert (
        result_baseline.provenance.initial_state_fingerprint
        == result_transfer.provenance.initial_state_fingerprint
        == initial_fingerprint
    )

    # 5. Over-Transfer Rejection (transfer 120 exceeds source inventory of 100)
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
        name="Rejection",
        horizon=1,
        samples=1,
        seed=42,
        scheduled_actions=(ScheduledAction(step=0, action=over_transfer_action),),
    )
    result_rejection = world.simulate(scenario=rejection_scenario)
    traj_rejection = result_rejection.trajectories[0]
    step_0 = traj_rejection.steps[0]

    # Assert action was rejected
    assert len(step_0.actions_accepted) == 0
    assert len(step_0.actions_proposed) == 1
    assert step_0.actions_proposed[0].id == "over_transfer_120"

    # Assert hard violation occurred on 'available_inventory' constraint
    violations = [
        v for v in step_0.constraint_violations if v.constraint_id == "available_inventory"
    ]
    assert len(violations) == 1
    assert violations[0].severity == ConstraintSeverity.HARD
    assert violations[0].satisfied is False
    assert "Insufficient stock" in violations[0].message


@pytest.mark.example
def test_minimal_warehouse_decision_evaluation() -> None:
    """Verify decision-grade scenario evaluation on Minimal Warehouse counterfactuals."""
    from ewm_engine.evaluation.comparison import compare_scenarios
    from ewm_engine.evaluation.pareto import ObjectiveDirection

    world = create_warehouse_world()

    base_scn = Scenario(name="Baseline", horizon=1, samples=3, seed=42)
    res_base = world.simulate(scenario=base_scn)

    act_transfer = Action(
        id="transfer_30",
        type="transfer_resource",
        parameters={
            "source_resource": "warehouse_a",
            "target_resource": "warehouse_b",
            "quantity": 30.0,
        },
    )
    transfer_scn = Scenario(
        name="Intervention",
        horizon=1,
        samples=3,
        seed=42,
        scheduled_actions=(ScheduledAction(step=0, action=act_transfer),),
    )
    res_transfer = world.simulate(scenario=transfer_scn)

    comparison = compare_scenarios(
        baseline=res_base,
        candidates=[res_transfer],
        metrics=["served_demand", "unserved_demand"],
        objectives={
            "served_demand": ObjectiveDirection.MAXIMIZE,
            "unserved_demand": ObjectiveDirection.MINIMIZE,
        },
        objective=lambda m: m["served_demand"] - 2.0 * m["unserved_demand"],
    )

    # Unserved demand delta is -30.0 (improvement)
    delta_unserved = comparison.bootstrap_deltas["Intervention"]["unserved_demand"]
    assert delta_unserved.absolute_delta == -30.0
    assert delta_unserved.is_significant

    # Served demand delta is +30.0 (improvement)
    delta_served = comparison.bootstrap_deltas["Intervention"]["served_demand"]
    assert delta_served.absolute_delta == 30.0
    assert delta_served.is_significant

    # Pareto analysis: Intervention dominates Baseline
    assert comparison.pareto_frontier is not None
    assert comparison.pareto_frontier.frontier == ["Intervention"]
    assert (
        "Intervention" in comparison.pareto_frontier.dominates["Intervention"]
        or "Baseline" in comparison.pareto_frontier.dominated_by
    )

    # User objective ranks Intervention above Baseline
    assert comparison.rankings is not None
    assert comparison.rankings[0][0] == "Intervention"
