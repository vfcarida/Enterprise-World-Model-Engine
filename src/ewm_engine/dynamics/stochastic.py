"""Stochastic dynamics models incorporating probabilistic transitions and random distributions."""

from __future__ import annotations

from collections.abc import Sequence

from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import RandomGenerator, ResourceId
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.provenance.evidence import EvidenceLevel


class StochasticDemandDynamics:
    """Simulates stochastic demand draining resources from entities.

    Samples consumption from a Gaussian distribution N(mean, std) truncated at 0.
    Tracks fulfilled vs unmet demand in state memory or return metrics.
    """

    def __init__(
        self,
        resource_id: ResourceId,
        mean_demand: float,
        std_demand: float,
        name: str = "StochasticDemandDynamics",
        evidence_level: EvidenceLevel = EvidenceLevel.PREDICTIVE,
    ) -> None:
        self.resource_id = resource_id
        self.mean_demand = max(0.0, mean_demand)
        self.std_demand = max(0.0, std_demand)
        self.name = name
        self.evidence_level = evidence_level

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
    ) -> TransitionResult:
        # Check if exogenous events amplify demand
        demand_multiplier = 1.0
        for event in exogenous_events:
            if event.type == "demand_surge":
                demand_multiplier *= float(event.get("surge_factor", 1.5))

        # Sample demand
        raw_demand = rng.normal(self.mean_demand, self.std_demand) * demand_multiplier
        sampled_demand = max(0.0, float(raw_demand))

        current_res = state.get_resource(self.resource_id)
        available_stock = max(0.0, current_res.current - current_res.min_value)

        fulfilled_demand = min(sampled_demand, available_stock)
        unmet_demand = sampled_demand - fulfilled_demand

        next_state = state.update_resource(self.resource_id, delta=-fulfilled_demand)

        # Update running unmet demand memory
        prior_unmet = float(state.memory.get(f"unmet_{self.resource_id}", 0.0))
        next_state = next_state.with_memory(f"unmet_{self.resource_id}", prior_unmet + unmet_demand)
        next_state = next_state.with_memory(f"last_demand_{self.resource_id}", sampled_demand)

        return TransitionResult(
            next_state=next_state,
            applied_changes={
                "resource_id": self.resource_id,
                "sampled_demand": sampled_demand,
                "fulfilled_demand": fulfilled_demand,
                "unmet_demand": unmet_demand,
            },
            evidence_level=self.evidence_level,
            diagnostics={"demand_multiplier": demand_multiplier},
            model_name=self.name,
        )
