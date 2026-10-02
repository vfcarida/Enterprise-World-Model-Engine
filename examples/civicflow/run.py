"""CivicFlow: Flagship research demonstration of flood-response resource allocation.

RESEARCH DISCLAIMER:
This simulation is strictly for research, educational, and computational evaluation.
It does NOT represent an operational disaster response system, nor should it be used
for real-world emergency management without certified validation and human oversight.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.standard import ResourceNonNegativeConstraint
from ewm_engine.core.actions import Intervention
from ewm_engine.core.events import ExogenousEvent, ExogenousEventSource
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import RandomGenerator
from ewm_engine.core.world import World
from ewm_engine.dynamics.composite import CompositeDynamics
from ewm_engine.evaluation.comparison import compare_scenarios
from ewm_engine.simulation.branching import branch_world
from ewm_engine.simulation.scenario import Scenario
from examples.civicflow.constraints import (
    RoadPassabilityConstraint,
    ShelterBedCapacityConstraint,
)
from examples.civicflow.dynamics import (
    DisasterReliefLogisticsDynamics,
    FloodHydrologyDynamics,
)
from examples.civicflow.policies import (
    CapacityAwareRegionalActor,
    MyopicNearestFirstActor,
)
from examples.civicflow.world import create_civicflow_world_state


class FloodSurgeEventSource(ExogenousEventSource):
    """Generates an intense rainfall surge during the early phase of the flood emergency."""

    def sample(
        self, state: WorldState, step: int, rng: RandomGenerator
    ) -> Sequence[ExogenousEvent]:
        # At step 2, a major cloudburst hits the valley and upland basin
        if step == 2:
            return [
                ExogenousEvent(
                    id="cloudburst_basin",
                    type="rainfall_surge",
                    severity=3.5,
                    parameters={"rain_increase_mm_h": 45.0},
                    description="Severe convective storm cell dumps heavy rainfall across the basin.",
                )
            ]
        return []


def run_civicflow_simulation() -> None:
    print("=" * 80)
    print("CIVICFLOW: DISASTER RELIEF RESOURCE ALLOCATION WORLD MODEL")
    print("=" * 80)
    print("RESEARCH DISCLAIMER: FOR SCIENTIFIC & EDUCATIONAL DEMONSTRATION ONLY.")
    print("NOT CERTIFIED FOR REAL-WORLD EMERGENCY LOGISTICS DEPLOYMENT.\n")

    # 1. Initialize Baseline World State
    state_s0 = create_civicflow_world_state()

    # 2. Register Coupled Hydrology + Logistics Dynamics
    dynamics = CompositeDynamics(
        models=[
            FloodHydrologyDynamics(),
            DisasterReliefLogisticsDynamics(),
        ],
        name="CivicFlowCoupledDynamics",
    )

    # 3. Register Inviolable Physical and Operational Constraints
    constraints = ConstraintRegistry(
        [
            RoadPassabilityConstraint(),
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

    # 4. Branch Policy A: Status Quo Myopic Nearest-First Allocation
    world_a = branch_world(base_world)
    world_a.add_actor(MyopicNearestFirstActor())

    scenario_a = Scenario(
        name="Policy A (Myopic Nearest)",
        horizon=10,
        samples=15,
        seed=101,
        intervention=Intervention(
            id="policy_a_myopic",
            description="Nearest-shelter-first allocation prioritizing S1 over S2 and S3",
        ),
    )
    result_a = world_a.simulate(scenario=scenario_a)

    # 5. Branch Policy B: Capacity-Aware & Preemptive Causeway Buffering
    world_b = branch_world(base_world)
    world_b.add_actor(CapacityAwareRegionalActor())

    scenario_b = Scenario(
        name="Policy B (Proactive Regional)",
        horizon=10,
        samples=15,
        seed=101,
        intervention=Intervention(
            id="policy_b_proactive",
            description="Preemptively buffer coastal S2 before C1 inundates; direct route to S3",
        ),
    )
    result_b = world_b.simulate(scenario=scenario_b)

    # 6. Comparative Evaluation
    comparison = compare_scenarios(
        baseline=result_a,
        candidates=[result_b],
        metrics=[
            "mem_cumulative_unserved_water",
            "mem_cumulative_unserved_rations",
            "resource_water_shelter_s2",
            "violations_count",
        ],
    )

    print(comparison.summary_table())

    # 7. Systemic Trace Demonstration
    print("\n=== CivicFlow Systemic Dependency Trace (Policy B - Trajectory 0) ===")
    sample_trace = result_b.trajectories[0].systemic_trace
    print(sample_trace.to_mermaid())


if __name__ == "__main__":
    run_civicflow_simulation()
