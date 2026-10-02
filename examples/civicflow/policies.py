"""CivicFlow: Resource-allocation policy actors."""

from __future__ import annotations

from collections.abc import Sequence

from ewm_engine.actors.base import ActorContext
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import ActorId


class MyopicNearestFirstActor:
    """Policy A: Myopic nearest-shelter-first allocation.

    Dispatches supplies exclusively to the closest shelter (S1), neglecting
    vulnerable coastal populations (S2) and ignoring rising river stage forecasts.
    """

    def __init__(self, actor_id: ActorId = "myopic_coordinator") -> None:
        self._actor_id = actor_id

    @property
    def actor_id(self) -> ActorId:
        return self._actor_id

    def act(self, state: WorldState, context: ActorContext) -> Sequence[Action]:
        actions: list[Action] = []
        # Checks Shelter S1 stock first
        s1_rat = state.get_resource("rations_shelter_s1").current
        if s1_rat < 80.0:
            actions.append(
                Action(
                    id=f"dispatch_s1_step_{context.step}",
                    actor_id=self.actor_id,
                    type="dispatch_relief",
                    parameters={
                        "source_depot": "depot_valley",
                        "target_shelter": "shelter_s1",
                        "rations": 70.0,
                        "water": 140.0,
                    },
                    timestamp=context.timestamp,
                )
            )
        else:
            # If S1 full, dispatch to S2
            actions.append(
                Action(
                    id=f"dispatch_s2_step_{context.step}",
                    actor_id=self.actor_id,
                    type="dispatch_relief",
                    parameters={
                        "source_depot": "depot_valley",
                        "target_shelter": "shelter_s2",
                        "rations": 60.0,
                        "water": 120.0,
                    },
                    timestamp=context.timestamp,
                )
            )
        return actions


class CapacityAwareRegionalActor:
    """Policy B: Capacity-aware & inundation-preemptive allocation.

    Preemptively pushes high reserves into coastal Shelter S2 before Causeway C1 floods,
    while concurrently routing supplies from Central Depot to Shelter S3 based on occupancy.
    """

    def __init__(self, actor_id: ActorId = "proactive_regional_coordinator") -> None:
        self._actor_id = actor_id

    @property
    def actor_id(self) -> ActorId:
        return self._actor_id

    def act(self, state: WorldState, context: ActorContext) -> Sequence[Action]:
        actions: list[Action] = []
        river_stage = float(state.memory.get("river_stage_meters", 3.0))
        c1_status = state.memory.get("road_C1_status", "open")

        # 1. Preemptive Surge to S2 before C1 closes
        if c1_status == "open":
            # If river is rising towards inundation, prioritize buffering coastal S2
            s2_water = state.get_resource("water_shelter_s2").current
            if river_stage > 3.8 or s2_water < 200.0:
                actions.append(
                    Action(
                        id=f"preemptive_s2_step_{context.step}",
                        actor_id=self.actor_id,
                        type="dispatch_relief",
                        parameters={
                            "source_depot": "depot_valley",
                            "target_shelter": "shelter_s2",
                            "rations": 100.0,
                            "water": 200.0,
                        },
                        timestamp=context.timestamp,
                    )
                )

        # 2. Concurrently supply Shelter S3 from Depot Central
        s3_rat = state.get_resource("rations_shelter_s3").current
        if s3_rat < 80.0:
            actions.append(
                Action(
                    id=f"direct_s3_step_{context.step}",
                    actor_id=self.actor_id,
                    type="dispatch_relief",
                    parameters={
                        "source_depot": "depot_central",
                        "target_shelter": "shelter_s3",
                        "rations": 60.0,
                        "water": 120.0,
                    },
                    timestamp=context.timestamp,
                )
            )

        # 3. Supply S1 as needed
        s1_rat = state.get_resource("rations_shelter_s1").current
        if s1_rat < 70.0:
            actions.append(
                Action(
                    id=f"supply_s1_step_{context.step}",
                    actor_id=self.actor_id,
                    type="dispatch_relief",
                    parameters={
                        "source_depot": "depot_valley",
                        "target_shelter": "shelter_s1",
                        "rations": 50.0,
                        "water": 100.0,
                    },
                    timestamp=context.timestamp,
                )
            )

        return actions
