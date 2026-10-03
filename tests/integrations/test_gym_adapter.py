"""Unit and integration tests for EnterpriseGymEnv adapter."""

from __future__ import annotations

import numpy as np

from ewm_engine.constraints.standard import (
    ActionTransferAvailabilityConstraint,
    ResourceCapacityConstraint,
)
from ewm_engine.core.actions import Action
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.integrations.gym import EnterpriseGymEnv


def _make_test_world() -> World:
    state = WorldState(
        resources=[
            Resource(id="stock_a", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="stock_b", current=20.0, min_value=0.0, max_value=200.0),
        ]
    )
    return World(
        state=state,
        dynamics=DeterministicTransferDynamics(),
        constraints=[
            ResourceCapacityConstraint(resource_id="stock_a"),
            ResourceCapacityConstraint(resource_id="stock_b"),
            ActionTransferAvailabilityConstraint(),
        ],
    )


def test_enterprise_gym_env_reset_and_step() -> None:
    """Ensure gym environment resets cleanly and steps world dynamics."""
    world = _make_test_world()
    transfer_action = Action(
        id="act_transfer_20",
        type="transfer_resource",
        parameters={
            "source_resource": "stock_a",
            "target_resource": "stock_b",
            "quantity": 20.0,
        },
    )

    env = EnterpriseGymEnv(
        world=world,
        max_steps=3,
        action_mapping=[transfer_action],
    )

    # 1. Reset
    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2,)
    assert obs[0] == 100.0
    assert obs[1] == 20.0
    assert info["step"] == 0

    # 2. Step 1 (valid transfer action 0)
    obs, _reward, terminated, truncated, info = env.step(0)
    assert obs[0] == 80.0
    assert obs[1] == 40.0
    assert not terminated
    assert not truncated
    assert info["actions_accepted"] == 1
    assert info["violations_count"] == 0

    # 3. Step 2 & 3 (reaching max_steps=3 -> truncated)
    env.step(0)
    obs, _reward, _terminated, truncated, info = env.step(0)
    assert truncated is True
    assert obs[0] == 40.0
    assert obs[1] == 80.0


def test_enterprise_gym_env_constraint_violation_penalty() -> None:
    """Ensure invalid actions trigger penalty and termination if configured."""
    world = _make_test_world()
    # Action exceeding stock_a (150 > 100)
    excess_action = Action(
        id="act_overflow",
        type="transfer_resource",
        parameters={
            "source_resource": "stock_a",
            "target_resource": "stock_b",
            "quantity": 150.0,
        },
    )

    env = EnterpriseGymEnv(
        world=world,
        max_steps=5,
        action_mapping=[excess_action],
        violation_penalty=50.0,
        terminate_on_hard_violation=True,
    )

    env.reset()
    _obs, reward, terminated, _truncated, info = env.step(0)

    # Action rejected, penalty applied
    assert info["actions_proposed"] == 1
    assert info["actions_accepted"] == 0
    assert info["hard_violations"] >= 1
    assert reward <= -50.0
    assert terminated is True
