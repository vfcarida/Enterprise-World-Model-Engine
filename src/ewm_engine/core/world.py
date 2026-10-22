"""World abstraction coordinating state, dynamics, constraints, actors, and simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ewm_engine.core.actions import Action, Intervention
from ewm_engine.core.context import ContextSnapshot
from ewm_engine.core.events import ExogenousEventSource
from ewm_engine.core.observation import ObservationFilter, ObservationSpec
from ewm_engine.core.state import WorldState

if TYPE_CHECKING:
    from ewm_engine.actors.base import Actor
    from ewm_engine.constraints.base import Constraint
    from ewm_engine.constraints.registry import ConstraintRegistry
    from ewm_engine.dynamics.base import DynamicsModel
    from ewm_engine.simulation.scenario import Scenario, ScheduledAction
    from ewm_engine.simulation.trajectory import SimulationResult


class World:
    """The root container orchestrating an enterprise world specification.

    Binds together the initial state, dynamics models, operational constraints,
    exogenous event sources, participating actors, typed runtime context K_t,
    and partial observability specifications O_t.
    """

    def __init__(
        self,
        state: WorldState | None = None,
        dynamics: DynamicsModel | None = None,
        constraints: ConstraintRegistry | Sequence[Constraint] | None = None,
        event_sources: Sequence[ExogenousEventSource] | None = None,
        actors: Sequence[Actor] | None = None,
        *,
        initial_state: WorldState | None = None,
        exogenous_events: Sequence[ExogenousEventSource] | None = None,
        runtime_context: ContextSnapshot | None = None,
        observation_spec: ObservationSpec | Mapping[str, ObservationSpec] | None = None,
    ) -> None:
        from ewm_engine.constraints.registry import ConstraintRegistry
        from ewm_engine.exceptions import InvalidWorldStateError

        resolved_state = state if state is not None else initial_state
        if resolved_state is None:
            raise InvalidWorldStateError(
                "World requires a valid initial state (pass 'state' or 'initial_state')."
            )

        self.initial_state = resolved_state
        self.dynamics = dynamics

        if constraints is None:
            self.constraints = ConstraintRegistry()
        elif isinstance(constraints, ConstraintRegistry):
            self.constraints = constraints
        else:
            self.constraints = ConstraintRegistry(list(constraints))

        resolved_events = event_sources if event_sources is not None else exogenous_events
        self.event_sources: list[ExogenousEventSource] = list(resolved_events or [])
        self.actors: list[Actor] = list(actors or [])
        self.runtime_context: ContextSnapshot | None = runtime_context
        self.observation_spec: ObservationSpec | Mapping[str, ObservationSpec] | None = (
            observation_spec
        )
        self.observation_filter: ObservationFilter = ObservationFilter(observation_spec)

    def add_actor(self, actor: Actor) -> World:
        """Register an autonomous actor or policy agent."""
        self.actors.append(actor)
        return self

    def add_constraint(self, constraint: Constraint) -> World:
        """Register an operational or physical constraint."""
        self.constraints.register(constraint)
        return self

    def add_event_source(self, event_source: ExogenousEventSource) -> World:
        """Register an exogenous shock generator."""
        self.event_sources.append(event_source)
        return self

    def snapshot(self) -> WorldState:
        """Retrieve an immutable snapshot of the baseline world state."""
        return self.initial_state

    @property
    def current_state(self) -> WorldState:
        """Retrieve the current baseline state of the world."""
        return self.initial_state

    def with_runtime_context(self, runtime_context: ContextSnapshot) -> World:
        """Return a branched World with an updated ContextSnapshot without mutating self."""
        return self.branch(runtime_context=runtime_context)

    def with_observation_spec(
        self,
        observation_spec: ObservationSpec | Mapping[str, ObservationSpec] | None,
    ) -> World:
        """Return a branched World with an updated ObservationSpec without mutating self."""
        return self.branch(observation_spec=observation_spec)

    def branch(
        self,
        state: WorldState | None = None,
        *,
        runtime_context: ContextSnapshot | None = None,
        observation_spec: ObservationSpec | Mapping[str, ObservationSpec] | None = None,
    ) -> World:
        """Create an independent counterfactual branch from a world state snapshot.

        Modifications in the branched world do not affect the origin world.
        """
        import copy

        base_state = state if state is not None else self.initial_state
        resolved_ctx = (
            runtime_context if runtime_context is not None else copy.deepcopy(self.runtime_context)
        )
        resolved_obs = (
            observation_spec
            if observation_spec is not None
            else copy.deepcopy(self.observation_spec)
        )

        cloned_actors: list[Actor] = []
        for a in self.actors:
            if hasattr(a, "clone") and callable(a.clone):
                cloned_actors.append(a.clone())
            else:
                try:
                    cloned_actors.append(copy.deepcopy(a))
                except Exception:
                    cloned_actors.append(a)

        cloned_event_sources: list[ExogenousEventSource] = []
        for es in self.event_sources:
            if hasattr(es, "clone") and callable(es.clone):
                cloned_event_sources.append(es.clone())
            else:
                try:
                    cloned_event_sources.append(copy.deepcopy(es))
                except Exception:
                    cloned_event_sources.append(es)

        return World(
            state=base_state,
            dynamics=self.dynamics,
            constraints=self.constraints.clone(),
            event_sources=cloned_event_sources,
            actors=cloned_actors,
            runtime_context=resolved_ctx,
            observation_spec=resolved_obs,
        )

    def simulate(
        self,
        scenario: Scenario | None = None,
        intervention: Intervention | None = None,
        actions: Sequence[Action | ScheduledAction] | Action | ScheduledAction | None = None,
        horizon: int = 10,
        samples: int = 1,
        seed: int = 42,
        name: str | None = None,
        executor: Any = None,
    ) -> SimulationResult:
        """Execute a forward Monte Carlo simulation of this world under a given scenario.

        If scenario is not provided, a default Scenario is created with the given
        horizon, samples, seed, and optional actions. Actions provided without an explicit
        step schedule are scheduled at step 0 (immediate execution).
        """
        from ewm_engine.simulation.engine import SimulationEngine
        from ewm_engine.simulation.scenario import Scenario, ScheduledAction

        scheduled: list[ScheduledAction] = []
        if actions is not None:
            if isinstance(actions, (Action, ScheduledAction)):
                action_items = [actions]
            else:
                action_items = list(actions)
            for item in action_items:
                if isinstance(item, ScheduledAction):
                    scheduled.append(item)
                else:
                    scheduled.append(ScheduledAction(step=0, action=item))

        if scenario is None:
            is_cf = bool(scheduled or intervention)
            default_name = "Counterfactual" if is_cf else "Baseline"
            default_id = "counterfactual" if is_cf else "baseline"
            scenario = Scenario(
                name=name or default_name,
                scenario_id=(name.lower().replace(" ", "_") if name else default_id),
                horizon=horizon,
                samples=samples,
                seed=seed,
                intervention=intervention,
                scheduled_actions=tuple(scheduled),
            )
        else:
            if intervention is not None and scenario.intervention is None:
                scenario = scenario.with_intervention(intervention)
            if scheduled:
                scenario = scenario.branch(
                    scenario_id=scenario.scenario_id,
                    scheduled_actions=(*scenario.scheduled_actions, *scheduled),
                )
            if name is not None:
                scenario = scenario.branch(scenario_id=name.lower().replace(" ", "_"), name=name)

        engine = SimulationEngine(executor=executor)
        return engine.run(world=self, scenario=scenario)

    def as_scm(self, horizon: int = 1) -> Any:
        """Compile an additive Structural Causal Model (SCM) view from this world."""
        from ewm_engine.core.scm import StructuralCausalModel

        return StructuralCausalModel.from_world(self, horizon=horizon)

    def counterfactual(
        self,
        observed: Any,
        intervention: Any,
        horizon: int | None = None,
        seed: int | None = None,
        locality_tolerance: float = 1e-5,
    ) -> Any:
        """Evaluate a native Pearl three-step counterfactual rollout under Twin-Rollout noise coupling.

        Implements:
            1. Abduction: Recover posterior over exogenous noise U given observed trajectory.
            2. Action: Apply causal surgery do(X = x) to replace structural assignments.
            3. Prediction: Re-run the modified model using the abduced, coupled exogenous noise.

        References:
            - Richens & Everitt (ICLR 2024 oral, arXiv:2402.10877): Robust agents learn causal world models.
            - Pearl (2009): Causality: Models, Reasoning, and Inference.
        """
        from ewm_engine.simulation.counterfactual import CounterfactualEngine

        cf_engine = CounterfactualEngine(locality_tolerance=locality_tolerance)
        return cf_engine.run(
            world=self,
            observed=observed,
            intervention=intervention,
            horizon=horizon,
            seed=seed,
        )
