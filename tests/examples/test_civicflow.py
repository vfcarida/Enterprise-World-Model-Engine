"""Acceptance test (AC-013): CivicFlow Flagship Policy Comparison & Systemic Trace."""

from __future__ import annotations

from pathlib import Path

import pytest

from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.standard import ResourceNonNegativeConstraint
from ewm_engine.core.actions import Intervention
from ewm_engine.core.spec import WorldSpec
from ewm_engine.core.world import World
from ewm_engine.dynamics.composite import CompositeDynamics
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.simulation.branching import branch_world
from ewm_engine.simulation.scenario import Scenario
from examples.civicflow.constraints import (
    AllocatedSupplyAvailableConstraint,
    RoadPassabilityConstraint,
    ShelterBedCapacityConstraint,
    VehicleLoadCapacityConstraint,
)
from examples.civicflow.dynamics import (
    DisasterReliefLogisticsDynamics,
    FloodHydrologyDynamics,
)
from examples.civicflow.policies import (
    CapacityAwareAllocationPolicy,
    NearestShelterFirstPolicy,
)
from examples.civicflow.run import FloodSurgeEventSource


@pytest.mark.example
def test_civicflow_tiny_flood_acceptance_scenario() -> None:
    """AC-013: CivicFlow two-policy comparison, 0 hard violations, trace with reroutes & mixed evidence."""
    fixture_path = (
        Path(__file__).resolve().parent.parent / "fixtures" / "civicflow" / "tiny_flood_v1.yaml"
    )
    assert fixture_path.exists(), f"Fixture file not found: {fixture_path}"

    # 1. Load initial world state from declarative YAML specification
    spec = WorldSpec.from_yaml(fixture_path)
    state_s0 = spec.build_state()

    # 2. Setup Coupled Dynamics (Predictive Hydrology + Structural Logistics)
    dynamics = CompositeDynamics(
        models=[
            FloodHydrologyDynamics(evidence_level=EvidenceLevel.PREDICTIVE),
            DisasterReliefLogisticsDynamics(evidence_level=EvidenceLevel.STRUCTURAL),
        ],
        name="CivicFlowCoupledDynamics",
    )

    # 3. Setup Explicit Constraints
    constraints = ConstraintRegistry(
        [
            RoadPassabilityConstraint(),
            VehicleLoadCapacityConstraint(max_load=300.0),
            AllocatedSupplyAvailableConstraint(),
            ShelterBedCapacityConstraint(shelter_id="shelter_s1"),
            ShelterBedCapacityConstraint(shelter_id="shelter_s2"),
            ShelterBedCapacityConstraint(shelter_id="shelter_s3"),
            ResourceNonNegativeConstraint(resource_id="rations_depot_valley"),
            ResourceNonNegativeConstraint(resource_id="water_depot_valley"),
        ]
    )

    base_world = World(
        state=state_s0,
        dynamics=dynamics,
        constraints=constraints,
        event_sources=[FloodSurgeEventSource()],
    )

    # 4. Branch Policy A: Status Quo Nearest-Shelter-First
    world_a = branch_world(base_world)
    world_a.add_actor(NearestShelterFirstPolicy())

    scenario_a = Scenario(
        scenario_id="civicflow_policy_a_nearest",
        name="Policy A (Nearest Shelter First)",
        horizon=10,
        samples=1,
        seed=101,
        intervention=Intervention(
            id="policy_a_nearest",
            description="Nearest-shelter-first allocation prioritizing S1",
        ),
    )
    result_a = world_a.simulate(scenario=scenario_a)
    traj_a = result_a.trajectories[0]

    # 5. Branch Policy B: Capacity-Aware & Preemptive Causeway Buffering
    world_b = branch_world(base_world)
    world_b.add_actor(CapacityAwareAllocationPolicy())

    scenario_b = Scenario(
        scenario_id="civicflow_policy_b_capacity_aware",
        name="Policy B (Capacity-Aware & Preemptive)",
        horizon=10,
        samples=1,
        seed=101,
        intervention=Intervention(
            id="policy_b_capacity_aware",
            description="Preemptively buffer coastal S2 and reroute deliveries when C1 closes",
        ),
    )
    result_b = world_b.simulate(scenario=scenario_b)
    traj_b = result_b.trajectories[0]

    # --- Normative Assertions for tiny_flood_v1 ---
    # 1. Exact initial state fingerprint identity across both branches
    assert (
        result_a.provenance.initial_state_fingerprint
        == result_b.provenance.initial_state_fingerprint
        == state_s0.fingerprint
    )

    # 2. Both policies respect all hard physical/operational constraints (0 hard violations)
    assert traj_a.hard_violations == 0
    assert traj_b.hard_violations == 0

    # 3. Capacity-aware policy strictly outperforms nearest-shelter-first on unserved demand
    unserved_a = traj_a.final_metric("unserved_demand")
    unserved_b = traj_b.final_metric("unserved_demand")
    assert unserved_b < unserved_a, (
        f"Expected capacity_aware unserved ({unserved_b}) < nearest ({unserved_a})"
    )

    # 4. Systemic trace inspection: contains a 'reroutes' relation edge
    trace_b = traj_b.systemic_trace
    edge_relations = {e.relation for e in trace_b.edges}
    assert "reroutes" in edge_relations, f"Trace expected 'reroutes' relation, got {edge_relations}"

    # 5. Systemic trace inspection: contains mixed EvidenceLevels (structural + predictive)
    node_evidence_levels = {n.evidence_level for n in trace_b.nodes.values()}
    assert EvidenceLevel.STRUCTURAL in node_evidence_levels
    assert EvidenceLevel.PREDICTIVE in node_evidence_levels
    assert EvidenceLevel.INTERVENTIONAL in node_evidence_levels
