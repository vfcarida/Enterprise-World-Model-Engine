"""Unit tests for ecosystem adapters (external agents and solvers)."""

from __future__ import annotations

import numpy as np
import pytest

from ewm_engine.actors.base import ActorContext
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.integrations.agents import CallableActorAdapter
from ewm_engine.integrations.solvers import Z3ConstraintAdapter


def test_callable_actor_adapter(sample_world_state: WorldState) -> None:
    """Verify that CallableActorAdapter wraps arbitrary functions into Actor protocol."""

    def custom_agent_fn(state: WorldState, context: ActorContext) -> list[Action]:
        # Simple policy: propose an action if stock is low
        res = state.get_resource("stock_wh1")
        if res.current <= 150:
            return [
                Action(
                    id=f"agent_action_{context.step}",
                    type="custom_dispatch",
                    parameters={"qty": 25.0},
                )
            ]
        return []

    adapter = CallableActorAdapter(actor_id="langgraph_agent", act_fn=custom_agent_fn)
    assert adapter.actor_id == "langgraph_agent"

    rng = np.random.default_rng(42)
    ctx = ActorContext(step=0, timestamp=0.0, rng=rng)

    actions = adapter.act(sample_world_state, ctx)
    assert len(actions) == 1
    assert actions[0].type == "custom_dispatch"
    assert actions[0].get("qty") == 25.0

    # Test cloning
    cloned = adapter.clone()
    assert cloned.actor_id == adapter.actor_id
    assert cloned is not adapter


def test_z3_constraint_adapter_without_z3(sample_world_state: WorldState) -> None:
    """Verify Z3ConstraintAdapter handles missing or mock Z3 solver gracefully."""

    def dummy_solver(
        state: WorldState, action: Action | None, z3_mod: object
    ) -> tuple[bool, str, dict[str, object]]:
        return True, "Z3 satisfied", {}

    adapter = Z3ConstraintAdapter(constraint_id="z3_capacity", solver_fn=dummy_solver)

    if adapter._z3_module is None:
        with pytest.raises(SimulationConfigurationError, match="z3-solver is not installed"):
            adapter.evaluate(sample_world_state)
    else:
        res = adapter.evaluate(sample_world_state)
        assert res.satisfied
