"""CivicFlow: World state and topology for disaster logistics research simulation."""

from __future__ import annotations

from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState


def create_civicflow_world_state() -> WorldState:
    """Builds the canonical initial world state for the CivicFlow flood response simulation.

    RESEARCH DISCLAIMER:
    This simulation is designed strictly for academic, research, and algorithmic evaluation
    purposes. It is NOT calibrated or certified for operational emergency-management deployment.
    """
    # 1. Geographic Regions, Shelters, Warehouses, and Roads
    entities = [
        # Regional Administrative Zones
        Entity(
            id="region_upland", type="region", attributes={"elevation_m": 120.0, "risk_tier": "low"}
        ),
        Entity(
            id="region_valley", type="region", attributes={"elevation_m": 45.0, "risk_tier": "high"}
        ),
        Entity(
            id="region_coastal",
            type="region",
            attributes={"elevation_m": 8.0, "risk_tier": "critical"},
        ),
        # Central and Regional Logistics Depots
        Entity(
            id="depot_central",
            type="warehouse",
            attributes={"zone": "region_upland", "throughput_max": 200},
        ),
        Entity(
            id="depot_valley",
            type="warehouse",
            attributes={"zone": "region_valley", "throughput_max": 100},
        ),
        # Emergency Shelters
        Entity(
            id="shelter_s1", type="shelter", attributes={"zone": "region_valley", "max_beds": 150}
        ),
        Entity(
            id="shelter_s2", type="shelter", attributes={"zone": "region_coastal", "max_beds": 200}
        ),
        Entity(
            id="shelter_s3", type="shelter", attributes={"zone": "region_coastal", "max_beds": 100}
        ),
        # Evacuation Transport Fleet
        Entity(id="fleet_alpha", type="transport_fleet", attributes={"truck_capacity_tons": 5.0}),
    ]

    # 2. Road Network and Logistical Corridors
    relationships = [
        # Depot Central connects to Depot Valley via primary Highway H1
        Relationship(
            source="depot_central",
            target="depot_valley",
            type="corridor",
            attributes={"road_id": "H1", "distance_km": 35.0, "status": "open"},
        ),
        # Depot Valley connects to Shelter S1 via Arterial A1
        Relationship(
            source="depot_valley",
            target="shelter_s1",
            type="corridor",
            attributes={"road_id": "A1", "distance_km": 10.0, "status": "open"},
        ),
        # Depot Valley connects to Shelter S2 via Coastal Causeway C1 (flood-prone)
        Relationship(
            source="depot_valley",
            target="shelter_s2",
            type="corridor",
            attributes={"road_id": "C1", "distance_km": 25.0, "status": "open"},
        ),
        # Depot Central connects directly to Shelter S3 via Hill Route R3
        Relationship(
            source="depot_central",
            target="shelter_s3",
            type="corridor",
            attributes={"road_id": "R3", "distance_km": 60.0, "status": "open"},
        ),
    ]

    # 3. Finite Critical Relief Resources
    resources = [
        # Depot Inventories (Rations and Potable Water Packs)
        Resource(
            id="rations_depot_central",
            entity_id="depot_central",
            current=1000.0,
            min_value=0.0,
            max_value=2000.0,
            unit="ration_packs",
        ),
        Resource(
            id="water_depot_central",
            entity_id="depot_central",
            current=1500.0,
            min_value=0.0,
            max_value=3000.0,
            unit="water_liters",
        ),
        Resource(
            id="rations_depot_valley",
            entity_id="depot_valley",
            current=300.0,
            min_value=0.0,
            max_value=600.0,
            unit="ration_packs",
        ),
        Resource(
            id="water_depot_valley",
            entity_id="depot_valley",
            current=400.0,
            min_value=0.0,
            max_value=800.0,
            unit="water_liters",
        ),
        # Shelter On-Hand Stocks
        Resource(
            id="rations_shelter_s1",
            entity_id="shelter_s1",
            current=50.0,
            min_value=0.0,
            max_value=500.0,
            unit="ration_packs",
        ),
        Resource(
            id="water_shelter_s1",
            entity_id="shelter_s1",
            current=80.0,
            min_value=0.0,
            max_value=1000.0,
            unit="water_liters",
        ),
        Resource(
            id="occupancy_shelter_s1",
            entity_id="shelter_s1",
            current=60.0,
            min_value=0.0,
            max_value=150.0,
            unit="evacuees",
        ),
        Resource(
            id="rations_shelter_s2",
            entity_id="shelter_s2",
            current=40.0,
            min_value=0.0,
            max_value=600.0,
            unit="ration_packs",
        ),
        Resource(
            id="water_shelter_s2",
            entity_id="shelter_s2",
            current=60.0,
            min_value=0.0,
            max_value=1200.0,
            unit="water_liters",
        ),
        Resource(
            id="occupancy_shelter_s2",
            entity_id="shelter_s2",
            current=110.0,
            min_value=0.0,
            max_value=200.0,
            unit="evacuees",
        ),
        Resource(
            id="rations_shelter_s3",
            entity_id="shelter_s3",
            current=20.0,
            min_value=0.0,
            max_value=300.0,
            unit="ration_packs",
        ),
        Resource(
            id="water_shelter_s3",
            entity_id="shelter_s3",
            current=30.0,
            min_value=0.0,
            max_value=600.0,
            unit="water_liters",
        ),
        Resource(
            id="occupancy_shelter_s3",
            entity_id="shelter_s3",
            current=50.0,
            min_value=0.0,
            max_value=100.0,
            unit="evacuees",
        ),
        # Transport Capacity
        Resource(
            id="available_trucks",
            entity_id="fleet_alpha",
            current=12.0,
            min_value=0.0,
            max_value=12.0,
            unit="trucks",
        ),
    ]

    return WorldState(
        entities=entities,
        relationships=relationships,
        resources=resources,
        memory={
            "cumulative_unserved_rations": 0.0,
            "cumulative_unserved_water": 0.0,
            "river_stage_meters": 3.2,
            "rainfall_rate_mm_h": 15.0,
            "road_C1_status": "open",
        },
        active_rules={
            "flood_inundation_threshold_m": 5.5,
            "ration_per_evacuee_per_step": 1.0,
            "water_liters_per_evacuee_per_step": 2.0,
        },
        context={
            "weather_warning": "level_3_flood_watch",
            "evacuation_phase": "active_relocation",
        },
        timestamp=0.0,
        step=0,
    )
