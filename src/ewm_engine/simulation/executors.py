"""Pluggable execution backends for parallel and distributed Monte Carlo rollouts.

Provides the RolloutExecutor protocol and concrete backends:
- SerialExecutor: In-process sequential reference execution (default).
- MultiprocessingExecutor: Multi-core CPU execution via standard library concurrent.futures.
- RayExecutor: Distributed execution across cores or clusters via Ray (optional extra).

All backends preserve the strict determinism invariant (AC-P03-1):
    distributed_result.logical == serial_result.logical
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from ewm_engine.constraints.results import ConstraintPhase, ConstraintResult
from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.simulation.rollout import RolloutResult, execute_single_rollout
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import StepRecord

if TYPE_CHECKING:
    from ewm_engine.core.world import World


@runtime_checkable
class RolloutExecutor(Protocol):
    """Protocol defining the interface for simulation rollout execution backends."""

    def execute(
        self,
        world: World,
        scenario: Scenario,
        child_seeds: Sequence[np.random.SeedSequence],
        *,
        on_constraint_eval: Callable[[ConstraintResult, ConstraintPhase, int, int], None]
        | None = None,
        on_step_completed: Callable[[int, int, str, StepRecord, bool], None] | None = None,
    ) -> list[RolloutResult]:
        """Execute rollouts across child seeds and return sorted rollout results.

        Results must be returned in strict ascending order of sample_id.
        """
        ...


class SerialExecutor:
    """Sequential, in-process rollout executor (default reference implementation).

    Preserves zero-overhead sequential execution and emits fine-grained lifecycle
    events synchronously as steps evolve.
    """

    def execute(
        self,
        world: World,
        scenario: Scenario,
        child_seeds: Sequence[np.random.SeedSequence],
        *,
        on_constraint_eval: Callable[[ConstraintResult, ConstraintPhase, int, int], None]
        | None = None,
        on_step_completed: Callable[[int, int, str, StepRecord, bool], None] | None = None,
    ) -> list[RolloutResult]:
        """Execute all sample rollouts sequentially in the current process."""
        return [
            execute_single_rollout(
                world=world,
                scenario=scenario,
                sample_idx=idx,
                child_seed=seed,
                on_constraint_eval=on_constraint_eval,
                on_step_completed=on_step_completed,
            )
            for idx, seed in enumerate(child_seeds)
        ]


def _mp_worker_entrypoint(
    payload: tuple[World, Scenario, int, np.random.SeedSequence],
) -> RolloutResult:
    """Worker task entrypoint for multiprocessing execution.

    Must remain a top-level module function to support 'spawn' on Windows/macOS.
    """
    world, scenario, sample_idx, child_seed = payload
    return execute_single_rollout(
        world=world,
        scenario=scenario,
        sample_idx=sample_idx,
        child_seed=child_seed,
    )


class MultiprocessingExecutor:
    """Parallel process pool rollout executor using Python stdlib concurrent.futures.

    Distributes rollout tasks across local CPU cores with zero third-party dependencies.
    Preserves bitwise determinism by passing pre-spawned child seeds and reassembling
    completed rollouts in deterministic sample_id order.
    """

    def __init__(
        self,
        max_workers: int | None = None,
        chunksize: int = 1,
    ) -> None:
        """Initialize MultiprocessingExecutor.

        Args:
            max_workers: Maximum number of worker processes (defaults to os.cpu_count()).
            chunksize: Task chunk size passed to ProcessPoolExecutor.map.
        """
        self.max_workers = max_workers
        self.chunksize = chunksize

    def execute(
        self,
        world: World,
        scenario: Scenario,
        child_seeds: Sequence[np.random.SeedSequence],
        *,
        on_constraint_eval: Callable[[ConstraintResult, ConstraintPhase, int, int], None]
        | None = None,
        on_step_completed: Callable[[int, int, str, StepRecord, bool], None] | None = None,
    ) -> list[RolloutResult]:
        """Execute rollouts in parallel across process workers and reassemble deterministically."""
        from concurrent.futures import ProcessPoolExecutor

        tasks = [(world, scenario, idx, child_seeds[idx]) for idx in range(len(child_seeds))]

        with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(pool.map(_mp_worker_entrypoint, tasks, chunksize=self.chunksize))

        results.sort(key=lambda r: r.sample_id)
        return results


def _ray_worker_entrypoint(
    world: World,
    scenario: Scenario,
    sample_idx: int,
    child_seed: np.random.SeedSequence,
) -> RolloutResult:
    """Ray remote worker task entrypoint."""
    return execute_single_rollout(
        world=world,
        scenario=scenario,
        sample_idx=sample_idx,
        child_seed=child_seed,
    )


class RayExecutor:
    """Distributed rollout executor scaling across multi-core nodes or Ray clusters.

    Requires the optional 'distributed' extra: `pip install ewm-engine[distributed]`.
    Places World and Scenario into Plasma shared-memory store once and dispatches
    stateless rollout tasks across the Ray cluster.
    """

    def __init__(
        self,
        address: str | None = None,
        num_cpus: int | None = None,
        ray_remote_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize RayExecutor.

        Args:
            address: Optional Ray cluster address (e.g., 'auto' or 'ray://<head>:10001').
            num_cpus: Number of CPUs to allocate for local Ray instance if initializing.
            ray_remote_kwargs: Additional options passed to ray.remote(worker).options(...).
        """
        self.address = address
        self.num_cpus = num_cpus
        self.ray_remote_kwargs = ray_remote_kwargs or {}

    def execute(
        self,
        world: World,
        scenario: Scenario,
        child_seeds: Sequence[np.random.SeedSequence],
        *,
        on_constraint_eval: Callable[[ConstraintResult, ConstraintPhase, int, int], None]
        | None = None,
        on_step_completed: Callable[[int, int, str, StepRecord, bool], None] | None = None,
    ) -> list[RolloutResult]:
        """Execute rollouts across a Ray cluster and reassemble deterministically."""
        try:
            import ray
        except ImportError as exc:
            raise ImportError(
                "The 'ray' package is required to use RayExecutor. "
                "Install it via 'pip install ewm-engine[distributed]' or 'pip install ray>=2.9.0'."
            ) from exc

        if not ray.is_initialized():
            init_kwargs: dict[str, Any] = {"ignore_reinit_error": True}
            if self.address is not None:
                init_kwargs["address"] = self.address
            if self.num_cpus is not None:
                init_kwargs["num_cpus"] = self.num_cpus
            ray.init(**init_kwargs)

        world_ref = ray.put(world)
        scenario_ref = ray.put(scenario)

        worker_task = ray.remote(_ray_worker_entrypoint)
        if self.ray_remote_kwargs:
            worker_task = worker_task.options(**self.ray_remote_kwargs)

        futures = [
            worker_task.remote(world_ref, scenario_ref, idx, child_seeds[idx])
            for idx in range(len(child_seeds))
        ]

        results: list[RolloutResult] = ray.get(futures)
        results.sort(key=lambda r: r.sample_id)
        return results


def resolve_executor(executor: RolloutExecutor | str | None) -> RolloutExecutor:
    """Resolve an executor parameter to a validated RolloutExecutor instance.

    Args:
        executor: An instance of RolloutExecutor, a string shorthand ('serial',
            'multiprocessing', 'ray'), or None (defaulting to SerialExecutor).

    Returns:
        A concrete RolloutExecutor instance.

    Raises:
        SimulationConfigurationError: If an invalid string or unsupported object is provided.
    """
    if executor is None or executor == "serial":
        return SerialExecutor()
    if executor == "multiprocessing":
        return MultiprocessingExecutor()
    if executor == "ray":
        return RayExecutor()
    if isinstance(executor, RolloutExecutor):
        return executor
    raise SimulationConfigurationError(
        f"Unknown executor specification: {executor!r}. "
        "Expected 'serial', 'multiprocessing', 'ray', or an instance of RolloutExecutor."
    )
