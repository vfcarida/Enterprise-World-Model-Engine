"""Unit tests for rule-based and stochastic decision actors."""

from __future__ import annotations

import numpy as np

from ewm_engine.actors.base import ActorContext
from ewm_engine.actors.rule_based import ThresholdReplenishmentActor
from ewm_engine.actors.stochastic import StochasticActor
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState


def test_threshold_replenishment_actor(sample_world_state: WorldState) -> None:
    """Test that ThresholdReplenishmentActor triggers an action only below threshold."""
    actor = ThresholdReplenishmentActor(
        actor_id="actor_replenish",
        source_resource="stock_wh1",
        target_resource="stock_wh2",
        reorder_point=40.0,
        order_quantity=50.0,
    )
    rng = np.random.default_rng(42)

    # Initial state: stock_wh2 is 50.0 > 40.0 -> no action
    ctx0 = ActorContext(step=0, timestamp=0.0, rng=rng)
    actions0 = actor.act(sample_world_state, ctx0)
    assert len(actions0) == 0

    # Depleted state: stock_wh2 is 30.0 <= 40.0 -> trigger replenishment
    depleted_state = sample_world_state.update_resource("stock_wh2", new_value=30.0)
    ctx1 = ActorContext(step=1, timestamp=1.0, rng=rng)
    actions1 = actor.act(depleted_state, ctx1)
    assert len(actions1) == 1
    assert actions1[0].actor_id == "actor_replenish"
    assert actions1[0].get("quantity") == 50.0


def test_stochastic_actor_choice(sample_world_state: WorldState) -> None:
    """Test that StochasticActor chooses action candidates according to distribution."""
    c1 = Action(id="choice_a", type="act_a")
    c2 = Action(id="choice_b", type="act_b")

    actor = StochasticActor(
        actor_id="stochastic_agent",
        action_candidates=[c1, c2],
        probabilities=[1.0, 0.0],  # Deterministic selection of c1
    )
    rng = np.random.default_rng(42)
    ctx = ActorContext(step=0, timestamp=0.0, rng=rng)

    actions = actor.act(sample_world_state, ctx)
    assert len(actions) == 1
    assert actions[0].type == "act_a"
    assert actions[0].actor_id == "stochastic_agent"
