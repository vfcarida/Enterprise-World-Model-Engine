"""Gymnasium (OpenAI Gym) reinforcement learning environment adapter for EWM Engine."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import numpy as np

from ewm_engine.constraints.results import ConstraintResult, ConstraintSeverity
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.provenance.trace import SystemicTrace
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario, ScheduledAction

try:
    _gym_mod: Any = importlib.import_module("gymnasium")
    _spaces_mod: Any = importlib.import_module("gymnasium.spaces")
    _HAS_GYMNASIUM = True
except ImportError:
    _gym_mod = None
    _spaces_mod = None
    _HAS_GYMNASIUM = False

if _HAS_GYMNASIUM and hasattr(_gym_mod, "Env"):
    _EnvBase: type[Any] = _gym_mod.Env
else:
    _EnvBase = object


class EnterpriseGymEnv(_EnvBase):  # type: ignore[misc]
    """Reinforcement learning environment adapter wrapping an EWM Engine World.

    Complies with the Gymnasium Env interface:
    `reset() -> (observation, info)`
    `step(action) -> (observation, reward, terminated, truncated, info)`
    `render() -> str | None`
    `close() -> None`

    Enables standard RL libraries (Stable-Baselines3, CleanRL, Ray RLlib, etc.)
    to train policies directly inside enterprise simulation worlds with explicit
    pre-action and post-transition constraint verification.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["ansi", "human", "html"],
        "render_fps": 10,
    }

    def __init__(
        self,
        world: World,
        max_steps: int = 50,
        observation_resources: Sequence[str] | None = None,
        observation_fn: Callable[[WorldState], np.ndarray] | None = None,
        action_mapping: Sequence[Action] | Callable[[Any], Action | None] | None = None,
        reward_fn: (
            Callable[[WorldState, Action | None, WorldState, Sequence[ConstraintResult]], float]
            | None
        ) = None,
        violation_penalty: float = 100.0,
        soft_violation_penalty: float = 10.0,
        terminate_on_hard_violation: bool = True,
        strict_gymnasium: bool = False,
        render_mode: str | None = None,
        observation_space: Any = None,
        action_space: Any = None,
        seed: int = 42,
    ) -> None:
        """Initialize the enterprise gym environment.

        Args:
            world: World specification.
            max_steps: Maximum simulation steps per episode before truncation.
            observation_resources: Specific resource IDs to vectorize as observation.
            observation_fn: Optional custom callable translating WorldState into a 1D float32 numpy array.
            action_mapping: Either a discrete list of candidate Actions (for Discrete action space),
                or a callable converting raw agent action (int or array) to an EWM Action.
            reward_fn: Custom reward function (prev_state, action, next_state, violations) -> float.
            violation_penalty: Penalty subtracted from reward for each hard constraint violation.
            soft_violation_penalty: Penalty subtracted from reward for each soft constraint violation.
            terminate_on_hard_violation: If True, halts episode when a hard invariant is breached.
            strict_gymnasium: If True, raises SimulationConfigurationError immediately if gymnasium is missing.
            render_mode: Render mode ("ansi", "human", "html", or None).
            observation_space: Optional explicit Gymnasium space overriding auto-inference.
            action_space: Optional explicit Gymnasium space overriding auto-inference.
            seed: Initial pseudo-random number generator seed.
        """
        if strict_gymnasium and not _HAS_GYMNASIUM:
            raise SimulationConfigurationError(
                "Gymnasium is required for EnterpriseGymEnv when strict_gymnasium=True. "
                "Install with `pip install gymnasium` or `pip install ewm-engine[ml]`."
            )

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode '{render_mode}'. Supported modes: {self.metadata['render_modes']}"
            )

        if _HAS_GYMNASIUM and _EnvBase is not object:
            super().__init__()

        self._world = world
        self._current_state: WorldState = world.initial_state
        self._max_steps = max_steps
        self._step_idx = 0
        self._violation_penalty = violation_penalty
        self._soft_violation_penalty = soft_violation_penalty
        self._terminate_on_hard_violation = terminate_on_hard_violation
        self.render_mode = render_mode
        self._rng = np.random.default_rng(seed)
        self._trace = SystemicTrace()
        self._engine = SimulationEngine()

        # Observation setup
        if observation_fn is not None:
            self._obs_fn = observation_fn
            sample_obs = self._obs_fn(self._current_state)
            obs_dim = sample_obs.shape[0]
            self._res_ids: list[str] = []
        else:
            self._res_ids = list(
                observation_resources
                if observation_resources is not None
                else list(self._world.initial_state.resources.keys())
            )
            self._obs_fn = self._default_observation_fn
            obs_dim = len(self._res_ids)

        # Action setup
        self._action_mapping = action_mapping

        # Reward setup
        self._reward_fn = reward_fn or self._default_reward_fn

        # Setup spaces if gymnasium is installed
        if observation_space is not None:
            self.observation_space: Any = observation_space
        elif _HAS_GYMNASIUM and _spaces_mod is not None:
            if observation_fn is not None:
                self.observation_space = _spaces_mod.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(obs_dim,),
                    dtype=np.float32,
                )
            else:
                lows = []
                highs = []
                for rid in self._res_ids:
                    res = self._world.initial_state.get_resource(rid)
                    min_v = float(res.min_value) if res is not None else -np.inf
                    max_v = float(res.max_value) if res is not None else np.inf
                    lows.append(min_v)
                    highs.append(max_v)
                self.observation_space = _spaces_mod.Box(
                    low=np.array(lows, dtype=np.float32),
                    high=np.array(highs, dtype=np.float32),
                    dtype=np.float32,
                )
        else:
            self.observation_space = None

        if action_space is not None:
            self.action_space: Any = action_space
        elif _HAS_GYMNASIUM and _spaces_mod is not None:
            if isinstance(action_mapping, (list, tuple)):
                self.action_space = _spaces_mod.Discrete(len(action_mapping))
            else:
                self.action_space = _spaces_mod.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                )
        else:
            self.action_space = None

    def _default_observation_fn(self, state: WorldState) -> np.ndarray:
        values = []
        for rid in self._res_ids:
            res = state.get_resource(rid)
            values.append(res.current if res is not None else 0.0)
        return np.array(values, dtype=np.float32)

    def _default_reward_fn(
        self,
        prev_state: WorldState,
        action: Action | None,
        next_state: WorldState,
        violations: Sequence[ConstraintResult],
    ) -> float:
        reward = 0.0
        # Penalize hard constraint breaches
        hard_count = sum(1 for v in violations if v.severity == ConstraintSeverity.HARD)
        soft_count = sum(1 for v in violations if v.severity == ConstraintSeverity.SOFT)
        reward -= hard_count * self._violation_penalty
        reward -= soft_count * self._soft_violation_penalty
        return reward

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to the world's initial state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._current_state = self._world.initial_state
        self._step_idx = 0
        self._trace = SystemicTrace()

        obs = self._obs_fn(self._current_state)
        info = {
            "step": self._step_idx,
            "fingerprint": self._current_state.fingerprint,
        }
        return obs, info

    def step(self, action_input: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the simulation world by executing candidate action."""
        curr_step = self._step_idx
        prev_state = self._current_state

        # Resolve Action instance
        action: Action | None = None
        if self._action_mapping is not None:
            if isinstance(self._action_mapping, (list, tuple)):
                idx = int(action_input)
                if 0 <= idx < len(self._action_mapping):
                    action = self._action_mapping[idx]
            elif callable(self._action_mapping):
                action = self._action_mapping(action_input)
        elif isinstance(action_input, Action):
            action = action_input

        # Schedule action for current step
        scheduled_actions = (
            [ScheduledAction(step=curr_step, action=action)] if action is not None else []
        )
        step_scenario = Scenario(
            name=f"GymStep_{curr_step}",
            horizon=curr_step + 1,
            samples=1,
            scheduled_actions=tuple(scheduled_actions),
        )

        step_record, next_state, is_invalid = self._engine._execute_step(
            world=self._world,
            state=prev_state,
            step_idx=curr_step,
            rng=self._rng,
            trace=self._trace,
            scenario=step_scenario,
        )

        self._current_state = next_state
        self._step_idx += 1

        violations = step_record.constraint_violations
        reward = self._reward_fn(prev_state, action, next_state, violations)

        has_hard_violation = any(v.severity == ConstraintSeverity.HARD for v in violations)
        terminated = bool(self._terminate_on_hard_violation and (is_invalid or has_hard_violation))
        truncated = bool(self._step_idx >= self._max_steps)

        obs = self._obs_fn(next_state)
        info = {
            "step": self._step_idx,
            "actions_proposed": len(step_record.actions_proposed),
            "actions_accepted": len(step_record.actions_accepted),
            "violations_count": len(violations),
            "hard_violations": sum(1 for v in violations if v.severity == ConstraintSeverity.HARD),
            "soft_violations": sum(1 for v in violations if v.severity == ConstraintSeverity.SOFT),
            "fingerprint": next_state.fingerprint,
            "is_invalid": is_invalid,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> str | None:
        """Render the environment state according to configured render_mode."""
        if self.render_mode is None:
            return None

        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            sys.stdout.write(self._render_ansi() + "\n")
            sys.stdout.flush()
            return None
        elif self.render_mode == "html":
            world_name = getattr(self._world, "name", "EnterpriseWorld")
            return self._trace.to_html(title=f"{world_name} Trace")
        else:
            raise NotImplementedError(
                f"Unsupported render_mode '{self.render_mode}'. Supported modes: {self.metadata['render_modes']}"
            )

    def _render_ansi(self) -> str:
        lines = [
            f"--- EWM Engine Gym Step {self._step_idx}/{self._max_steps} ---",
            f"State Fingerprint: {self._current_state.fingerprint[:16]}...",
            "Resources:",
        ]
        for rid in self._res_ids:
            res = self._current_state.get_resource(rid)
            val = f"{res.current:.2f}" if res is not None else "N/A"
            bounds = f" [{res.min_value}..{res.max_value}]" if res is not None else ""
            lines.append(f"  {rid}: {val}{bounds}")
        return "\n".join(lines)

    def close(self) -> None:
        """Clean up any active simulation resources or visualizers."""
        pass

    @property
    def current_state(self) -> WorldState:
        """Access the current underlying immutable world state."""
        return self._current_state
