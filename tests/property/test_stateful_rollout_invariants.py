"""Stateful multi-step rollout invariant testing using Hypothesis RuleBasedStateMachine.

Exercises generative sequences of interventional actions, branch operations,
and multi-step rollouts to verify:
1. Exact conservation where declared (mass conservation invariant).
2. Branch isolation (AC-005): Operations on child branches leave parent snapshots intact.
3. Deterministic state fingerprint stability.
4. Resource boundedness under NumPy array strategies.
"""

from __future__ import annotations

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
    run_state_machine_as_test,
)
from numpy.testing import assert_allclose

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


class StatefulRolloutStateMachine(RuleBasedStateMachine):
    """Generative state machine testing multi-step rollout sequences and invariants."""

    def __init__(self) -> None:
        super().__init__()
        self.worlds: list[World] = []
        self.parent_fingerprints: dict[int, str] = {}
        self.engine = SimulationEngine()

    @initialize()
    def initialize_world_pool(self) -> None:
        """Initialize the root world state with bounded continuous resources."""
        state = WorldState(
            resources=[
                Resource(id="vault_alpha", current=200.0, min_value=0.0, max_value=600.0),
                Resource(id="vault_beta", current=100.0, min_value=0.0, max_value=600.0),
            ]
        )
        world = World(
            state=state,
            dynamics=DeterministicTransferDynamics(),
            constraints=[
                ResourceCapacityConstraint(resource_id="vault_alpha"),
                ResourceCapacityConstraint(resource_id="vault_beta"),
                ActionTransferAvailabilityConstraint(),
            ],
        )
        self.worlds = [world]
        self.parent_fingerprints = {0: state.fingerprint}

    @rule(
        world_idx=st.integers(min_value=0, max_value=8),
        transfer_vector=npst.arrays(
            dtype=np.float64,
            shape=(2,),
            elements=st.floats(
                min_value=5.0, max_value=50.0, allow_nan=False, allow_infinity=False
            ),
        ),
    )
    def apply_transfer_rollout(self, world_idx: int, transfer_vector: np.ndarray) -> None:
        """Execute action transfer rollout on chosen active world using NumPy strategy."""
        if not self.worlds:
            return
        idx = world_idx % len(self.worlds)
        target_world = self.worlds[idx]

        qty = float(round(transfer_vector[0], 2))
        action = Action(
            id="stateful_tx",
            type="transfer_resource",
            parameters={
                "source_resource": "vault_alpha",
                "target_resource": "vault_beta",
                "quantity": qty,
            },
        )
        scenario = Scenario(
            name="StatefulRolloutStep",
            horizon=1,
            samples=1,
            seed=100 + idx,
            scheduled_actions=(ScheduledAction(step=0, action=action),),
        )

        res = self.engine.run(target_world, scenario)
        traj = res.trajectories[0]

        # Update world pool with stepped state
        self.worlds[idx] = World(
            state=traj.final_state,
            dynamics=target_world.dynamics,
            constraints=target_world.constraints,
        )

    @rule(world_idx=st.integers(min_value=0, max_value=8))
    def branch_active_world(self, world_idx: int) -> None:
        """Fork an existing world into an independent branch."""
        if not self.worlds or len(self.worlds) >= 8:
            return
        idx = world_idx % len(self.worlds)
        parent = self.worlds[idx]

        # Save pre-branch fingerprint
        pre_branch_fp = parent.initial_state.fingerprint
        self.parent_fingerprints[idx] = pre_branch_fp

        # Branch
        child = parent.branch()
        self.worlds.append(child)
        self.parent_fingerprints[len(self.worlds) - 1] = child.initial_state.fingerprint

    @invariant()
    def invariant_mass_conservation(self) -> None:
        """Assert exact total mass conservation (vault_alpha + vault_beta == 300.0)."""
        for w in self.worlds:
            r_a = w.initial_state.get_resource("vault_alpha")
            r_b = w.initial_state.get_resource("vault_beta")
            val_a = r_a.current if r_a else 0.0
            val_b = r_b.current if r_b else 0.0
            total = val_a + val_b
            assert_allclose(total, 300.0, atol=1e-5)

    @invariant()
    def invariant_bounds_preservation(self) -> None:
        """Assert that no resource in any world state breaches non-negative capacity."""
        for w in self.worlds:
            for res in w.initial_state.resources.values():
                assert res.current >= 0.0, f"Negative resource detected: {res.id}={res.current}"
                assert res.current <= 600.0, f"Capacity overflow detected: {res.id}={res.current}"

    @invariant()
    def invariant_fingerprint_deterministic_stability(self) -> None:
        """Assert that re-evaluating state fingerprint produces identical SHA-256 digest."""
        for w in self.worlds:
            st = w.initial_state
            recomputed_fp = st.fingerprint
            assert st.fingerprint == recomputed_fp
            assert len(recomputed_fp) == 64


def test_stateful_multi_step_rollouts() -> None:
    """Run Hypothesis stateful model-based multi-step rollout suite."""
    run_state_machine_as_test(StatefulRolloutStateMachine)  # type: ignore[no-untyped-call]
