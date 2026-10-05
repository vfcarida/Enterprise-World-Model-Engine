"""Candidate decision agents for evaluation in the hospital supply world model."""

from __future__ import annotations

from collections.abc import Sequence

from ewm_engine.actors.base import Actor, ActorContext
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import ActorId
from ewm_engine.integrations.ortools import CPSATAllocationPlanner


class GreedyMyopicAgent(Actor):
    """Myopic agent that attempts to satisfy the single largest demand center in one transfer.

    Fails to consider corridor link bandwidth constraints (max 60 units), resulting
    in hard constraint violations and rejected actions by the engine.
    """

    def __init__(self, actor_id: ActorId = "greedy_agent") -> None:
        self._actor_id = actor_id

    @property
    def actor_id(self) -> ActorId:
        return self._actor_id

    def act(self, state: WorldState, context: ActorContext) -> Sequence[Action]:
        # Identifies hospital with lowest stock (highest perceived vulnerability)
        hospitals = ["hospital_north", "hospital_central", "hospital_south"]
        stocks = {h: state.get_resource(h).current for h in hospitals}
        neediest = min(stocks, key=lambda k: stocks[k])

        # Attempts to dump 100 units through a corridor with a 60-unit capacity limit
        return [
            Action(
                id=f"{self.actor_id}_burst_{context.step}",
                type="transfer_resource",
                parameters={
                    "source_resource": "hub_stock",
                    "target_resource": neediest,
                    "quantity": 100.0,
                },
            )
        ]

    def clone(self) -> GreedyMyopicAgent:
        return GreedyMyopicAgent(actor_id=self.actor_id)


class ProportionalHeuristicAgent(Actor):
    """Conservative heuristic agent that splits available budget equally across all nodes.

    Respects corridor capacity limits (20 units < 60 limit), but fails to adapt to
    asymmetric demand distributions across hospitals.
    """

    def __init__(
        self,
        actor_id: ActorId = "proportional_agent",
        transfer_per_node: float = 20.0,
    ) -> None:
        self._actor_id = actor_id
        self.transfer_per_node = transfer_per_node

    @property
    def actor_id(self) -> ActorId:
        return self._actor_id

    def act(self, state: WorldState, context: ActorContext) -> Sequence[Action]:
        hub = state.get_resource("hub_stock")
        if hub is None or hub.current < (self.transfer_per_node * 3):
            return []

        hospitals = ["hospital_north", "hospital_central", "hospital_south"]
        actions = []
        for idx, hosp in enumerate(hospitals):
            actions.append(
                Action(
                    id=f"{self.actor_id}_transfer_{idx}_{context.step}",
                    type="transfer_resource",
                    parameters={
                        "source_resource": "hub_stock",
                        "target_resource": hosp,
                        "quantity": self.transfer_per_node,
                    },
                )
            )
        return actions

    def clone(self) -> ProportionalHeuristicAgent:
        return ProportionalHeuristicAgent(
            actor_id=self.actor_id,
            transfer_per_node=self.transfer_per_node,
        )


class CPSATOptimizationAgent(Actor):
    """Operations Research decision agent powered by Google OR-Tools CP-SAT.

    Solves joint multi-commodity flow under corridor capacities and regional demand targets,
    minimizing unserved demand while preserving constraint feasibility.
    """

    def __init__(
        self,
        actor_id: ActorId = "cpsat_agent",
        time_limit_seconds: float = 5.0,
    ) -> None:
        self._actor_id = actor_id
        self.time_limit_seconds = time_limit_seconds
        # Target replenishment demands per hospital
        self.targets = {
            "hospital_north": 30,
            "hospital_central": 45,
            "hospital_south": 25,
        }
        self.planner = CPSATAllocationPlanner(
            actor_id=actor_id,
            sources=["hub_stock"],
            destinations=["hospital_north", "hospital_central", "hospital_south"],
            demands=self._calculate_demands,
            capacities={
                ("hub_stock", "hospital_north"): 60,
                ("hub_stock", "hospital_central"): 60,
                ("hub_stock", "hospital_south"): 60,
            },
            allow_partial=True,
            time_limit_seconds=time_limit_seconds,
        )

    def _calculate_demands(self, state: WorldState) -> dict[str, int]:
        """Compute replenishment required to restore each hospital to target level."""
        demands = {}
        for hosp, target in self.targets.items():
            current = int(state.get_resource(hosp).current)
            deficit = max(0, target - current)
            demands[hosp] = deficit
        return demands

    @property
    def actor_id(self) -> ActorId:
        return self._actor_id

    def act(self, state: WorldState, context: ActorContext) -> Sequence[Action]:
        return self.planner.act(state, context)

    def clone(self) -> CPSATOptimizationAgent:
        return CPSATOptimizationAgent(
            actor_id=self.actor_id,
            time_limit_seconds=self.time_limit_seconds,
        )
