"""Acceptance test: scenario branching and counterfactual policy comparison."""

from __future__ import annotations

from ewm_engine.actors.rule_based import ThresholdReplenishmentActor
from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.standard import (
    ActionTransferAvailabilityConstraint,
    ResourceCapacityConstraint,
    ResourceNonNegativeConstraint,
)
from ewm_engine.core.actions import Intervention
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.composite import CompositeDynamics
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.dynamics.stochastic import StochasticDemandDynamics
from ewm_engine.evaluation.comparison import compare_scenarios
from ewm_engine.simulation.branching import branch_world
from ewm_engine.simulation.scenario import Scenario


def test_acceptance_counterfactual_branching_and_comparison(sample_world_state: WorldState) -> None:
    """Core Acceptance Test (Section 56):

    Given:
        The same initial world state and dynamics
    When:
        Intervention A (Conservative Order Size) is simulated
    And:
        Intervention B (Aggressive Bulk Order Size) is simulated
    Then:
        - Both scenarios produce reproducible trajectories
        - Each trajectory records its provenance and cryptographic fingerprint
        - Constraint violations are detected if thresholds are exceeded
        - Systemic traces can be inspected (nodes and edges recorded)
        - Scenario metrics can be compared with clear delta analysis
    """
    # 1. Given identical initial world state and dynamics
    initial_hash = sample_world_state.state_hash

    dynamics = CompositeDynamics(
        [
            DeterministicTransferDynamics(),
            StochasticDemandDynamics(resource_id="stock_wh2", mean_demand=25.0, std_demand=5.0),
        ]
    )

    constraints = ConstraintRegistry(
        [
            ResourceCapacityConstraint(resource_id="stock_wh1"),
            ResourceCapacityConstraint(resource_id="stock_wh2"),
            ResourceNonNegativeConstraint(resource_id="stock_wh2"),
            ActionTransferAvailabilityConstraint(),
        ]
    )

    base_world = World(
        state=sample_world_state,
        dynamics=dynamics,
        constraints=constraints,
    )

    # 2. When Intervention A is simulated (Conservative policy: small reorders)
    intervention_a = Intervention(
        id="policy_conservative",
        description="Conservative Small Batches",
        parameters={"reorder_point": 30.0, "order_qty": 20.0},
    )
    actor_a = ThresholdReplenishmentActor(
        actor_id="agent_a",
        source_resource="stock_wh1",
        target_resource="stock_wh2",
        reorder_point=30.0,
        order_quantity=20.0,
    )
    world_a = branch_world(base_world)
    world_a.add_actor(actor_a)

    scenario_a = Scenario(
        name="ConservativePolicy",
        horizon=12,
        samples=10,
        seed=999,
        intervention=intervention_a,
    )
    result_a = world_a.simulate(scenario=scenario_a)

    # And Intervention B is simulated (Aggressive policy: large bulk reorders)
    intervention_b = Intervention(
        id="policy_aggressive",
        description="Aggressive Bulk Batches",
        parameters={"reorder_point": 60.0, "order_qty": 80.0},
    )
    actor_b = ThresholdReplenishmentActor(
        actor_id="agent_b",
        source_resource="stock_wh1",
        target_resource="stock_wh2",
        reorder_point=60.0,
        order_quantity=80.0,
    )
    world_b = branch_world(base_world)
    world_b.add_actor(actor_b)

    scenario_b = Scenario(
        name="AggressivePolicy",
        horizon=12,
        samples=10,
        seed=999,
        intervention=intervention_b,
    )
    result_b = world_b.simulate(scenario=scenario_b)

    # 3. Then both scenarios produce reproducible trajectories with provenance
    assert result_a.metadata.world_hash == initial_hash
    assert result_b.metadata.world_hash == initial_hash
    assert result_a.metadata.fingerprint != result_b.metadata.fingerprint

    # Reproducibility check: rerunning scenario A with identical seed yields identical final state
    rerun_result_a = world_a.simulate(scenario=scenario_a)
    for t_orig, t_rerun in zip(result_a.trajectories, rerun_result_a.trajectories, strict=True):
        assert t_orig.final_state.state_hash == t_rerun.final_state.state_hash

    # 4. Systemic traces can be inspected
    for traj in result_a.trajectories:
        assert len(traj.systemic_trace.nodes) > 0
        mermaid_diag = traj.systemic_trace.to_mermaid()
        assert "flowchart TD" in mermaid_diag

    # 5. Scenario metrics can be compared
    comparison = compare_scenarios(
        baseline=result_a,
        candidates=[result_b],
        metrics=["resource_stock_wh2", "violations_count"],
    )

    comparison_dict = comparison.to_dict()
    assert "ConservativePolicy" in comparison_dict["scenarios"]
    assert "AggressivePolicy" in comparison_dict["scenarios"]
    assert "resource_stock_wh2" in comparison_dict["deltas_vs_baseline"]["AggressivePolicy"]

    table_report = comparison.summary_table()
    assert "=== Scenario Comparison (Baseline: ConservativePolicy) ===" in table_report
    assert "AggressivePolicy" in table_report


def test_actor_state_isolation_on_branch(sample_world_state: WorldState) -> None:
    """Verify that actors in a branched world are distinct instances from the original."""
    actor = ThresholdReplenishmentActor(
        actor_id="actor_base",
        source_resource="stock_wh1",
        target_resource="stock_wh2",
        reorder_point=40.0,
        order_quantity=20.0,
    )
    base_world = World(state=sample_world_state, actors=[actor])
    branched = branch_world(base_world)

    # Actor lists are distinct
    assert len(branched.actors) == 1
    assert branched.actors[0] is not actor  # Isolated cloned object

    # Modifying branched actor does not affect base actor
    branched.actors[0].reorder_point = 99.0
    assert actor.reorder_point == 40.0
