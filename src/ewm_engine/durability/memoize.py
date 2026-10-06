"""Simulation rollout memoization keyed on canonical SHA-256 fingerprints.

Conforms to Track T2: Fingerprint-keyed ResultStore & Memoization.
Ensures that a cache hit produces an outcome identical to a fresh execution,
avoiding redundant expensive simulation rollouts across identical scenario configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.durability.protocol import ResultStore
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import SimulationResult

if TYPE_CHECKING:
    from ewm_engine.core.world import World


def compute_simulation_fingerprint(world: World, scenario: Scenario) -> str:
    """Compute the deterministic canonical SHA-256 fingerprint for a simulation run.

    Combines:
    - world.initial_state.fingerprint
    - dynamics and component identities / versions
    - scenario_id, horizon, samples, and RNG seed entropy
    - interventions and scheduled actions
    """
    components: list[dict[str, str]] = []
    if world.dynamics is not None:
        dyn_name = getattr(world.dynamics, "name", world.dynamics.__class__.__name__)
        dyn_ver = getattr(world.dynamics, "version", "1.0.0")
        components.append({"id": dyn_name, "version": dyn_ver})

    for src in world.event_sources:
        src_id = getattr(src, "name", src.__class__.__name__)
        src_ver = getattr(src, "version", "1.0.0")
        components.append({"id": src_id, "version": src_ver})

    for actor in world.actors:
        actor_id = getattr(actor, "actor_id", actor.__class__.__name__)
        actor_ver = getattr(actor, "version", "1.0.0")
        components.append({"id": actor_id, "version": actor_ver})

    components.sort(key=lambda x: x["id"])

    # Scheduled actions canonical list
    sched_actions = [
        {
            "step": sa.step,
            "action_type": sa.action.type,
            "parameters": sa.action.parameters,
        }
        for sa in scenario.scheduled_actions
    ]
    sched_actions.sort(key=lambda x: (x["step"], x["action_type"]))

    # Intervention canonical dict
    intervention_dict = (
        scenario.intervention.model_dump(mode="json") if scenario.intervention is not None else None
    )

    canonical_payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "initial_state_fingerprint": world.initial_state.fingerprint,
        "scenario_id": scenario.scenario_id,
        "horizon": scenario.horizon,
        "samples": scenario.samples,
        "seed": scenario.seed,
        "components": components,
        "intervention": intervention_dict,
        "scheduled_actions": sched_actions,
    }

    return canonical_sha256(canonical_payload)


class MemoizedSimulationRunner:
    """Wrapper that caches and retrieves SimulationResults using a ResultStore.

    Guarantees that a cache hit returns results identical to a fresh run.
    """

    def __init__(
        self,
        engine: SimulationEngine | None = None,
        store: ResultStore | None = None,
    ) -> None:
        """Initialize runner with simulation engine and result store."""
        from ewm_engine.durability.backends.in_memory import InMemoryResultStore

        self.engine = engine or SimulationEngine()
        self.store = store or InMemoryResultStore()
        self._last_cache_hit: bool = False
        self._last_fingerprint: str = ""

    @property
    def last_cache_hit(self) -> bool:
        """Whether the most recent call to run() was served from cache."""
        return self._last_cache_hit

    @property
    def last_fingerprint(self) -> str:
        """The canonical fingerprint computed for the most recent run."""
        return self._last_fingerprint

    def run(
        self,
        world: World,
        scenario: Scenario,
        force_refresh: bool = False,
        **kwargs: Any,
    ) -> SimulationResult:
        """Execute or retrieve cached simulation rollouts."""
        fp = compute_simulation_fingerprint(world=world, scenario=scenario)
        self._last_fingerprint = fp

        if not force_refresh:
            cached = self.store.get(fp)
            if cached is not None:
                self._last_cache_hit = True
                return cached

        self._last_cache_hit = False
        result = self.engine.run(world=world, scenario=scenario, **kwargs)
        self.store.put(fp, result, metadata={"scenario_id": scenario.scenario_id})
        return result


def run_memoized(
    engine: SimulationEngine,
    world: World,
    scenario: Scenario,
    store: ResultStore,
    force_refresh: bool = False,
    **kwargs: Any,
) -> tuple[SimulationResult, bool]:
    """Execute a simulation with ResultStore memoization.

    Returns:
        tuple of (SimulationResult, cache_hit_boolean)
    """
    runner = MemoizedSimulationRunner(engine=engine, store=store)
    result = runner.run(world=world, scenario=scenario, force_refresh=force_refresh, **kwargs)
    return result, runner.last_cache_hit
