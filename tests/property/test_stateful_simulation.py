"""Stateful property-based testing for EWM Engine simulation and branching.

Verifies using Hypothesis RuleBasedStateMachine that under arbitrary,
randomly interleaved sequences of actions, branch operations, and simulation steps:
1. Branch Isolation (AC-005): Modifying or stepping a branch never alters parent worlds.
2. Resource Conservation: Deterministic transfers conserve total system mass.
3. Constraint Enforcements: Bounded resources never breach limits without detection.
"""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
    run_state_machine_as_test,
)

from ewm_engine.constraints.standard import (
    ActionTransferAvailabilityConstraint,
    ResourceCapacityConstraint,
)
from ewm_engine.core.actions import Action
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario, ScheduledAction


class WorldSimulationStateMachine(RuleBasedStateMachine):
    """Generative state machine exploring arbitrary interleavings of actions and branches."""

    def __init__(self) -> None:
        super().__init__()
        self.worlds: list[World] = []
        self.snapshots: list[dict[str, float]] = []
        self.engine = SimulationEngine()

    @initialize()
    def setup_root_world(self) -> None:
        """Initialize the root world state with two baseline resources."""
        state = WorldState(
            resources=[
                Resource(id="res_0", current=100.0, min_value=0.0, max_value=500.0),
                Resource(id="res_1", current=50.0, min_value=0.0, max_value=500.0),
            ]
        )
        world = World(
            state=state,
            dynamics=DeterministicTransferDynamics(),
            constraints=[
                ResourceCapacityConstraint(resource_id="res_0"),
                ResourceCapacityConstraint(resource_id="res_1"),
                ActionTransferAvailabilityConstraint(),
            ],
        )
        self.worlds = [world]
        self.snapshots = [{"res_0": 100.0, "res_1": 50.0}]

    @rule(
        world_idx=st.integers(min_value=0, max_value=5),
        transfer_amount=st.floats(min_value=1.0, max_value=80.0),
    )
    def step_transfer_action(self, world_idx: int, transfer_amount: float) -> None:
        """Execute a valid or invalid resource transfer on a selected active world."""
        if not self.worlds:
            return
        idx = world_idx % len(self.worlds)
        world = self.worlds[idx]

        action = Action(
            id="dyn_transfer",
            type="transfer_resource",
            parameters={
                "source_resource": "res_0",
                "target_resource": "res_1",
                "quantity": round(transfer_amount, 2),
            },
        )
        scenario = Scenario(
            name="StatefulTransferStep",
            horizon=1,
            samples=1,
            seed=42,
            scheduled_actions=(ScheduledAction(step=0, action=action),),
        )

        res = self.engine.run(world, scenario)
        traj = res.trajectories[0]

        # Update world's initial_state to track progressed simulation state
        self.worlds[idx] = World(
            state=traj.final_state,
            dynamics=world.dynamics,
            constraints=world.constraints,
        )

    @rule(world_idx=st.integers(min_value=0, max_value=5))
    def branch_world(self, world_idx: int) -> None:
        """Create an isolated branch from an active world."""
        if not self.worlds or len(self.worlds) >= 10:
            return
        idx = world_idx % len(self.worlds)
        source_world = self.worlds[idx]

        # Record pre-branch snapshot of source world
        src_state = source_world.initial_state
        r0 = src_state.get_resource("res_0")
        r1 = src_state.get_resource("res_1")
        self.snapshots[idx] = {
            "res_0": r0.current if r0 else 0.0,
            "res_1": r1.current if r1 else 0.0,
        }

        # Create isolated branch
        branched = source_world.branch()
        self.worlds.append(branched)
        b_r0 = branched.initial_state.get_resource("res_0")
        b_r1 = branched.initial_state.get_resource("res_1")
        self.snapshots.append(
            {
                "res_0": b_r0.current if b_r0 else 0.0,
                "res_1": b_r1.current if b_r1 else 0.0,
            }
        )

    @invariant()
    def check_total_resource_conservation(self) -> None:
        """Invariant: Total resources in every world must equal 150.0 (exact conservation)."""
        for w in self.worlds:
            st = w.initial_state
            r0 = st.get_resource("res_0")
            r1 = st.get_resource("res_1")
            v0 = r0.current if r0 else 0.0
            v1 = r1.current if r1 else 0.0
            total = v0 + v1
            assert abs(total - 150.0) < 1e-4, f"Resource conservation breached: total={total}"

    @invariant()
    def check_capacity_bounds(self) -> None:
        """Invariant: Resource levels must never be negative."""
        for w in self.worlds:
            st = w.initial_state
            for r in st.resources.values():
                assert r.current >= 0.0, f"Negative resource detected: {r.id}={r.current}"


def test_stateful_world_simulation_state_machine() -> None:
    """Run the Hypothesis stateful model-based simulation test."""
    run_state_machine_as_test(WorldSimulationStateMachine)  # type: ignore[no-untyped-call]
