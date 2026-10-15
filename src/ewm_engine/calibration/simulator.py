"""Forward simulation wrapper for Simulation-Based Inference (SBI).

Wraps deterministic, seeded EWM Engine forward rollouts to map parameter vectors
theta in R^D to summary statistic feature vectors x in R^P.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ewm_engine.calibration.priors import PriorSet
from ewm_engine.core.spec import WorldFactory, WorldSpec
from ewm_engine.core.types import ResourceId
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario


class ForwardSimulatorWrapper:
    """Wraps an EWM WorldSpec and Scenario as an SBI forward model: theta -> x."""

    def __init__(
        self,
        spec: WorldSpec,
        scenario: Scenario,
        prior_set: PriorSet,
        tracked_resources: Sequence[ResourceId],
        summary_statistic_mode: str = "trajectory",
    ) -> None:
        self.spec = spec
        self.scenario = scenario
        self.prior_set = prior_set
        self.tracked_resources = tuple(tracked_resources)
        self.summary_statistic_mode = summary_statistic_mode
        self._engine = SimulationEngine()
        self._factory = WorldFactory()

    def apply_parameters_to_spec(self, theta_dict: dict[str, float]) -> WorldSpec:
        """Create a new WorldSpec with parameters from theta_dict applied."""
        new_resources = list(self.spec.resources)
        new_dynamics = (
            self.spec.dynamics.model_copy(deep=True) if self.spec.dynamics is not None else None
        )
        new_world_meta = self.spec.world.model_copy(deep=True)

        for prior in self.prior_set.priors:
            if prior.name not in theta_dict:
                continue
            val = float(theta_dict[prior.name])

            if prior.target == "resource":
                # Update matching resource initial level
                updated_resources = []
                for r in new_resources:
                    if r.id == prior.target_id:
                        updated_resources.append(r.model_copy(update={"current": val}))
                    else:
                        updated_resources.append(r)
                new_resources = updated_resources
            elif prior.target in ("rule", "parameter"):
                if new_dynamics is not None:
                    dyn_params = dict(new_dynamics.parameters)
                    dyn_params[prior.target_id] = val
                    new_dynamics = new_dynamics.model_copy(update={"parameters": dyn_params})
            elif prior.target == "context":
                ctx = dict(new_world_meta.context)
                ctx[prior.target_id] = val
                new_world_meta = new_world_meta.model_copy(update={"context": ctx})

        update_dict: dict[str, Any] = {
            "resources": new_resources,
            "world": new_world_meta,
        }
        if new_dynamics is not None:
            update_dict["dynamics"] = new_dynamics

        return self.spec.model_copy(update=update_dict)

    def extract_summary_statistics(
        self,
        trajectory: Any,
    ) -> np.ndarray:
        """Extract summary feature vector from a completed trajectory."""
        features: list[float] = []
        states = [trajectory.initial_state] + [
            step.transition_result.next_state for step in trajectory.steps
        ]

        if self.summary_statistic_mode == "moments":
            # Extract mean, variance, min, max, final level for each tracked resource
            for rid in self.tracked_resources:
                vals = [float(s.resources[rid].current) for s in states if rid in s.resources]
                if not vals:
                    features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                    continue
                arr = np.array(vals, dtype=np.float64)
                features.extend(
                    [
                        float(np.mean(arr)),
                        float(np.std(arr)),
                        float(np.min(arr)),
                        float(np.max(arr)),
                        float(arr[-1]),
                    ]
                )
        else:
            # Trajectory mode: full sequence of resource values across steps
            for rid in self.tracked_resources:
                for s in states:
                    if rid in s.resources:
                        features.append(float(s.resources[rid].current))
                    else:
                        features.append(0.0)

        return np.array(features, dtype=np.float64)

    def simulate_single(
        self,
        theta_dict: dict[str, float],
        seed: int = 42,
    ) -> tuple[np.ndarray, dict[str, list[float]]]:
        """Simulate single parameter realization theta, returning summary stats and full trajectory series."""
        param_spec = self.apply_parameters_to_spec(theta_dict)
        world = self._factory.create_world(param_spec)

        scen = self.scenario.model_copy(update={"seed": seed, "samples": 1})
        result = self._engine.run(world, scen)
        traj = result.trajectories[0]

        summary_stats = self.extract_summary_statistics(traj)

        # Collect full series for plotting/predictive overlay
        series: dict[str, list[float]] = {}
        states = [traj.initial_state] + [step.transition_result.next_state for step in traj.steps]
        for rid in self.tracked_resources:
            series[rid] = [float(s.resources[rid].current) for s in states if rid in s.resources]

        return summary_stats, series

    def simulate_batch(
        self,
        thetas: np.ndarray,
        base_seed: int = 42,
    ) -> np.ndarray:
        """Simulate a batch of parameter vectors thetas of shape (N, D) -> returns X of shape (N, P)."""
        theta_dicts = self.prior_set.to_dict_list(thetas)
        outputs: list[np.ndarray] = []

        for i, t_dict in enumerate(theta_dicts):
            # Seed changes per simulation to capture environmental variability if stochastic
            seed_i = (base_seed + (i * 31)) % (2**31 - 1)
            stats, _ = self.simulate_single(t_dict, seed=seed_i)
            outputs.append(stats)

        return np.array(outputs, dtype=np.float64)
