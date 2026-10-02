"""World abstraction coordinating state, dynamics, constraints, actors, and simulation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ewm_engine.core.actions import Intervention
from ewm_engine.core.events import ExogenousEventSource
from ewm_engine.core.state import WorldState

if TYPE_CHECKING:
    from ewm_engine.actors.base import Actor
    from ewm_engine.constraints.base import Constraint
    from ewm_engine.constraints.registry import ConstraintRegistry
    from ewm_engine.dynamics.base import DynamicsModel
    from ewm_engine.simulation.scenario import Scenario
    from ewm_engine.simulation.trajectory import SimulationResult


class World:
    """The root container orchestrating an enterprise world specification.

    Binds together the initial state, dynamics models, operational constraints,
    exogenous event sources, and participating actors.
    """

    def __init__(
        self,
        state: WorldState,
        dynamics: DynamicsModel | None = None,
        constraints: ConstraintRegistry | Sequence[Constraint] | None = None,
        event_sources: Sequence[ExogenousEventSource] | None = None,
        actors: Sequence[Actor] | None = None,
    ) -> None:
        from ewm_engine.constraints.registry import ConstraintRegistry

        self.initial_state = state
        self.dynamics = dynamics

        if constraints is None:
            self.constraints = ConstraintRegistry()
        elif isinstance(constraints, ConstraintRegistry):
            self.constraints = constraints
        else:
            self.constraints = ConstraintRegistry(list(constraints))

        self.event_sources: list[ExogenousEventSource] = list(event_sources or [])
        self.actors: list[Actor] = list(actors or [])

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

    def branch(self, state: WorldState | None = None) -> World:
        """Create an independent counterfactual branch from a world state snapshot.

        Modifications in the branched world do not affect the origin world.
        """
        base_state = state if state is not None else self.initial_state
        return World(
            state=base_state,
            dynamics=self.dynamics,
            constraints=self.constraints.clone(),
            event_sources=list(self.event_sources),
            actors=list(self.actors),
        )

    def simulate(
        self,
        scenario: Scenario | None = None,
        intervention: Intervention | None = None,
        horizon: int = 10,
        samples: int = 1,
        seed: int = 42,
    ) -> SimulationResult:
        """Execute a forward Monte Carlo simulation of this world under a given scenario.

        If scenario is not provided, a default Scenario is created with the given
        horizon, samples, and seed.
        """
        from ewm_engine.simulation.engine import SimulationEngine
        from ewm_engine.simulation.scenario import Scenario

        if scenario is None:
            scenario = Scenario(
                name="AdHocSimulation",
                horizon=horizon,
                samples=samples,
                seed=seed,
                intervention=intervention,
            )
        elif intervention is not None and scenario.intervention is None:
            scenario = scenario.with_intervention(intervention)

        engine = SimulationEngine()
        return engine.run(world=self, scenario=scenario)
