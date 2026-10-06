"""Unit and integration tests for EnterpriseGymEnv adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
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


class _SoftWarningConstraint:
    """Simple soft constraint for testing reward penalties without termination."""

    def __init__(self, resource_id: str, threshold: float) -> None:
        self.resource_id = resource_id
        self.threshold = threshold
        self.constraint_id = f"warn_{resource_id}"
        self.version = "1.0.0"
        self.severity = ConstraintSeverity.SOFT
        self.description = f"Soft warning for {resource_id} > {threshold}"

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase,
    ) -> ConstraintResult:
        if phase != ConstraintPhase.POST_TRANSITION:
            return ConstraintResult(
                constraint_id=self.constraint_id,
                satisfied=True,
                severity=self.severity,
                phase=phase,
                message="Skipped (phase mismatch)",
            )
        res = state.get_resource(self.resource_id)
        if res is not None and res.current > self.threshold:
            return ConstraintResult(
                constraint_id=self.constraint_id,
                satisfied=False,
                severity=self.severity,
                phase=phase,
                message=f"Resource {self.resource_id} exceeds soft threshold {self.threshold}",
            )
        return ConstraintResult(
            constraint_id=self.constraint_id,
            satisfied=True,
            severity=self.severity,
            phase=phase,
            message="OK",
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


def test_enterprise_gym_env_spaces_and_subclass() -> None:
    """Ensure observation and action spaces match bounded resources and Gymnasium types."""
    try:
        import gymnasium as gym
        from gymnasium.spaces import Box, Discrete

        has_gym = True
    except ImportError:
        has_gym = False

    world = _make_test_world()
    transfer_action = Action(
        id="act_transfer",
        type="transfer_resource",
        parameters={"source_resource": "stock_a", "target_resource": "stock_b", "quantity": 10.0},
    )

    env = EnterpriseGymEnv(
        world=world,
        action_mapping=[transfer_action],
        render_mode="ansi",
    )

    if has_gym:
        assert isinstance(env, gym.Env)
        assert isinstance(env.observation_space, Box)
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 1
        # Bounds should match Resource min_value=0.0 and max_value=200.0
        np.testing.assert_allclose(env.observation_space.low, [0.0, 0.0], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(env.observation_space.high, [200.0, 200.0], rtol=1e-5, atol=1e-6)

    assert env.render_mode == "ansi"
    ansi_output = env.render()
    assert ansi_output is not None
    assert "stock_a" in ansi_output
    assert "stock_b" in ansi_output


def test_enterprise_gym_env_render_modes(capsys: pytest.CaptureFixture[str]) -> None:
    """Ensure render handles ansi, human, and html modes cleanly."""
    world = _make_test_world()
    env = EnterpriseGymEnv(world=world, render_mode="human")
    env.reset()
    res = env.render()
    assert res is None
    captured = capsys.readouterr()
    assert "--- EWM Engine Gym Step 0" in captured.out
    assert "stock_a: 100.00 [0.0..200.0]" in captured.out

    # HTML render mode
    env_html = EnterpriseGymEnv(world=world, render_mode="html")
    env_html.reset()
    html_output = env_html.render()
    assert html_output is not None
    assert "<!DOCTYPE html>" in html_output

    # Invalid render mode
    with pytest.raises(ValueError, match="Unsupported render_mode"):
        EnterpriseGymEnv(world=world, render_mode="invalid_mode")


def test_enterprise_gym_env_callable_action_mapping_and_soft_violations() -> None:
    """Ensure callable action mapping and soft constraint penalties work as expected."""
    world = _make_test_world()
    # Add a soft warning if stock_b exceeds 50.0
    world.add_constraint(_SoftWarningConstraint(resource_id="stock_b", threshold=50.0))

    def dynamic_action_builder(action_input: Any) -> Action | None:
        qty = float(action_input)
        return Action(
            id=f"dyn_transfer_{qty}",
            type="transfer_resource",
            parameters={
                "source_resource": "stock_a",
                "target_resource": "stock_b",
                "quantity": qty,
            },
        )

    env = EnterpriseGymEnv(
        world=world,
        max_steps=5,
        action_mapping=dynamic_action_builder,
        soft_violation_penalty=15.0,
        terminate_on_hard_violation=True,
    )

    env.reset(seed=123)
    # Step 1: transfer 40 -> stock_b becomes 60 (exceeds 50 threshold)
    _obs, reward, terminated, _truncated, info = env.step(40.0)

    assert not terminated
    assert info["soft_violations"] == 1
    assert info["hard_violations"] == 0
    assert reward == -15.0  # soft penalty applied without terminating episode

    env.close()


def test_enterprise_gym_env_custom_spaces_and_observation_fn() -> None:
    """Test custom observation function and explicit spaces."""
    try:
        from gymnasium.spaces import Box

        has_gym = True
    except ImportError:
        has_gym = False

    world = _make_test_world()

    def custom_obs_fn(state: WorldState) -> np.ndarray:
        res_a = state.get_resource("stock_a")
        res_b = state.get_resource("stock_b")
        diff = (res_a.current if res_a else 0.0) - (res_b.current if res_b else 0.0)
        return np.array([diff], dtype=np.float32)

    custom_obs_space = (
        Box(low=-200.0, high=200.0, shape=(1,), dtype=np.float32) if has_gym else None
    )
    custom_act_space = Box(low=0.0, high=50.0, shape=(1,), dtype=np.float32) if has_gym else None

    env = EnterpriseGymEnv(
        world=world,
        observation_fn=custom_obs_fn,
        observation_space=custom_obs_space,
        action_space=custom_act_space,
    )

    obs, _info = env.reset()
    assert obs.shape == (1,)
    assert obs[0] == 80.0  # 100 - 20 = 80
    assert env.current_state is not None
