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
        avail_rat = state.get_resource("rations_depot_valley").current
        avail_wat = state.get_resource("water_depot_valley").current

        # Checks Shelter S1 stock first
        s1_rat = state.get_resource("rations_shelter_s1").current
        if s1_rat < 80.0 and (avail_rat > 0 or avail_wat > 0):
            rat = min(70.0, avail_rat)
            wat = min(140.0, avail_wat)
            if rat > 0 or wat > 0:
                actions.append(
                    Action(
                        id=f"dispatch_s1_step_{context.step}",
                        actor_id=self.actor_id,
                        type="dispatch_relief",
                        parameters={
                            "source_depot": "depot_valley",
                            "target_shelter": "shelter_s1",
                            "rations": rat,
                            "water": wat,
                        },
                        timestamp=context.timestamp,
                    )
                )
        else:
            # If S1 full, dispatch to S2 ONLY if C1 is open (otherwise myopically wait)
            c1_status = state.memory.get("road_C1_status", "open")
            if c1_status == "open" and (avail_rat > 0 or avail_wat > 0):
                rat = min(40.0, avail_rat)
                wat = min(80.0, avail_wat)
                if rat > 0 or wat > 0:
                    actions.append(
                        Action(
                            id=f"dispatch_s2_step_{context.step}",
                            actor_id=self.actor_id,
                            type="dispatch_relief",
                            parameters={
                                "source_depot": "depot_valley",
                                "target_shelter": "shelter_s2",
                                "rations": rat,
                                "water": wat,
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

        avail_v_rat = state.get_resource("rations_depot_valley").current
        avail_v_wat = state.get_resource("water_depot_valley").current
        avail_c_rat = state.get_resource("rations_depot_central").current
        avail_c_wat = state.get_resource("water_depot_central").current

        # 1. Preemptive Surge to S2 before C1 closes
        if c1_status == "open":
            # If river is rising towards inundation, prioritize buffering coastal S2
            s2_water = state.get_resource("water_shelter_s2").current
            if (river_stage > 3.8 or s2_water < 200.0) and (avail_v_rat > 0 or avail_v_wat > 0):
                rat = min(100.0, avail_v_rat)
                wat = min(200.0, avail_v_wat)
                if rat > 0 or wat > 0:
                    actions.append(
                        Action(
                            id=f"preemptive_s2_step_{context.step}",
                            actor_id=self.actor_id,
                            type="dispatch_relief",
                            parameters={
                                "source_depot": "depot_valley",
                                "target_shelter": "shelter_s2",
                                "rations": rat,
                                "water": wat,
                            },
                            timestamp=context.timestamp,
                        )
                    )
                    avail_v_rat -= rat
                    avail_v_wat -= wat
        else:
            # 2. When C1 is inundated, actively reroute supplies to secondary shelter hubs
            if avail_c_rat > 0 or avail_c_wat > 0:
                rat = min(70.0, avail_c_rat)
                wat = min(140.0, avail_c_wat)
                if rat > 0 or wat > 0:
                    actions.append(
                        Action(
                            id=f"reroute_delivery_s3_step_{context.step}",
                            actor_id=self.actor_id,
                            type="reroute_delivery",
                            parameters={
                                "source_depot": "depot_central",
                                "target_shelter": "shelter_s3",
                                "rations": rat,
                                "water": wat,
                                "trace_relation": "reroutes",
                            },
                            timestamp=context.timestamp,
                        )
                    )
                    avail_c_rat -= rat
                    avail_c_wat -= wat

        # 3. Supply Shelter S3 from Depot Central via high-ground route R3
        s3_rat = state.get_resource("rations_shelter_s3").current
        if s3_rat < 80.0 and (avail_c_rat > 0 or avail_c_wat > 0):
            rat = min(60.0, avail_c_rat)
            wat = min(120.0, avail_c_wat)
            if rat > 0 or wat > 0:
                actions.append(
                    Action(
                        id=f"direct_s3_step_{context.step}",
                        actor_id=self.actor_id,
                        type="dispatch_relief",
                        parameters={
                            "source_depot": "depot_central",
                            "target_shelter": "shelter_s3",
                            "rations": rat,
                            "water": wat,
                        },
                        timestamp=context.timestamp,
                    )
                )

        # 4. Supply S1 as needed
        s1_rat = state.get_resource("rations_shelter_s1").current
        if s1_rat < 70.0 and (avail_v_rat > 0 or avail_v_wat > 0):
            rat = min(50.0, avail_v_rat)
            wat = min(100.0, avail_v_wat)
            if rat > 0 or wat > 0:
                actions.append(
                    Action(
                        id=f"supply_s1_step_{context.step}",
                        actor_id=self.actor_id,
                        type="dispatch_relief",
                        parameters={
                            "source_depot": "depot_valley",
                            "target_shelter": "shelter_s1",
                            "rations": rat,
                            "water": wat,
                        },
                        timestamp=context.timestamp,
                    )
                )

        return actions


# Canonical Policy Names
NearestShelterFirstPolicy = MyopicNearestFirstActor
CapacityAwareAllocationPolicy = CapacityAwareRegionalActor
