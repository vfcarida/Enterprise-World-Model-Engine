"""CivicFlow: Hydrological and disaster relief logistics dynamics."""

from __future__ import annotations

from collections.abc import Sequence

from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import RandomGenerator
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.provenance.evidence import EvidenceLevel


class FloodHydrologyDynamics:
    """Models rainfall accumulation, river stage rise, and road inundation thresholds."""

    def __init__(self, name: str = "FloodHydrologyDynamics") -> None:
        self.name = name

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
    ) -> TransitionResult:
        current_rain = float(state.memory.get("rainfall_rate_mm_h", 15.0))
        current_river = float(state.memory.get("river_stage_meters", 3.2))

        # 1. Process exogenous weather shocks
        for ev in exogenous_events:
            if ev.type == "rainfall_surge":
                surge_delta = float(ev.get("rain_increase_mm_h", 10.0))
                current_rain += surge_delta

        # Stochastic fluctuation around rainfall rate
        actual_rain = max(0.0, current_rain + float(rng.normal(0.0, 2.0)))
        # Hydrological runoff relation: delta river = 0.04 * rain - base_discharge
        river_delta = (0.035 * actual_rain) - 0.2
        next_river = max(1.5, current_river + river_delta)

        # Inundation check
        inundation_threshold = float(state.active_rules.get("flood_inundation_threshold_m", 5.5))
        c1_closed = next_river >= inundation_threshold
        road_status = "closed" if c1_closed else "open"

        next_state = state.with_memory("rainfall_rate_mm_h", actual_rain)
        next_state = next_state.with_memory("river_stage_meters", next_river)
        next_state = next_state.with_memory("road_C1_status", road_status)

        return TransitionResult(
            next_state=next_state,
            applied_changes={
                "rainfall_mm_h": actual_rain,
                "river_stage_m": next_river,
                "road_C1_status": road_status,
            },
            evidence_level=EvidenceLevel.STRUCTURAL,
            diagnostics={"inundated": c1_closed},
            model_name=self.name,
        )


class DisasterReliefLogisticsDynamics:
    """Models supply transfers, road connectivity, evacuee consumption, and unmet demand."""

    def __init__(self, name: str = "DisasterReliefLogisticsDynamics") -> None:
        self.name = name

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
    ) -> TransitionResult:
        current_state = state
        dispatches_executed = 0
        total_rations_moved = 0.0
        total_water_moved = 0.0
        road_c1_status = current_state.memory.get("road_C1_status", "open")

        # 1. Process supply dispatch actions
        for act in actions:
            if act.type != "dispatch_relief":
                continue

            depot = str(act.get("source_depot"))
            shelter = str(act.get("target_shelter"))
            rations = max(0.0, float(act.get("rations", 0.0)))
            water = max(0.0, float(act.get("water", 0.0)))

            # If targeting shelter_s2 via coastal route and road C1 is closed, dispatch fails
            if shelter == "shelter_s2" and road_c1_status == "closed":
                continue

            depot_rat_id = f"rations_{depot}"
            depot_wat_id = f"water_{depot}"
            shlt_rat_id = f"rations_{shelter}"
            shlt_wat_id = f"water_{shelter}"

            depot_rat = current_state.get_resource(depot_rat_id)
            depot_wat = current_state.get_resource(depot_wat_id)
            shlt_rat = current_state.get_resource(shlt_rat_id)
            shlt_wat = current_state.get_resource(shlt_wat_id)

            # Conservation constraints
            movable_rat = min(rations, depot_rat.current, shlt_rat.available_capacity)
            movable_wat = min(water, depot_wat.current, shlt_wat.available_capacity)

            if movable_rat > 0.0 or movable_wat > 0.0:
                current_state = current_state.update_resource(depot_rat_id, delta=-movable_rat)
                current_state = current_state.update_resource(shlt_rat_id, delta=movable_rat)
                current_state = current_state.update_resource(depot_wat_id, delta=-movable_wat)
                current_state = current_state.update_resource(shlt_wat_id, delta=movable_wat)
                dispatches_executed += 1
                total_rations_moved += movable_rat
                total_water_moved += movable_wat

        # 2. Simulate evacuee consumption and unmet demand across all shelters
        total_step_unmet_rations = 0.0
        total_step_unmet_water = 0.0

        for s_idx in ["s1", "s2", "s3"]:
            occupancy = current_state.get_resource(f"occupancy_shelter_{s_idx}").current
            dem_rations = occupancy * float(
                current_state.active_rules.get("ration_per_evacuee_per_step", 1.0)
            )
            dem_water = occupancy * float(
                current_state.active_rules.get("water_liters_per_evacuee_per_step", 2.0)
            )

            cur_rat = current_state.get_resource(f"rations_shelter_{s_idx}")
            cur_wat = current_state.get_resource(f"water_shelter_{s_idx}")

            consumed_rat = min(cur_rat.current, dem_rations)
            unmet_rat = dem_rations - consumed_rat

            consumed_wat = min(cur_wat.current, dem_water)
            unmet_wat = dem_water - consumed_wat

            current_state = current_state.update_resource(
                f"rations_shelter_{s_idx}", delta=-consumed_rat
            )
            current_state = current_state.update_resource(
                f"water_shelter_{s_idx}", delta=-consumed_wat
            )

            total_step_unmet_rations += unmet_rat
            total_step_unmet_water += unmet_wat

        prior_unmet_rat = float(current_state.memory.get("cumulative_unserved_rations", 0.0))
        prior_unmet_wat = float(current_state.memory.get("cumulative_unserved_water", 0.0))

        current_state = current_state.with_memory(
            "cumulative_unserved_rations", prior_unmet_rat + total_step_unmet_rations
        )
        current_state = current_state.with_memory(
            "cumulative_unserved_water", prior_unmet_wat + total_step_unmet_water
        )

        return TransitionResult(
            next_state=current_state,
            applied_changes={
                "dispatches_executed": dispatches_executed,
                "rations_moved": total_rations_moved,
                "water_moved": total_water_moved,
                "step_unserved_rations": total_step_unmet_rations,
                "step_unserved_water": total_step_unmet_water,
            },
            evidence_level=EvidenceLevel.STRUCTURAL,
            model_name=self.name,
        )
