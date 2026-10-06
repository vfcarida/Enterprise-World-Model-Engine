"""Unit tests for HeterogeneousGraphView and graph projection (P10)."""

from __future__ import annotations

import numpy as np
import pytest

from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.graph import HeterogeneousGraphView
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState


def _build_test_world_state() -> WorldState:
    """Build a multi-entity, multi-relational test WorldState."""
    entities = [
        Entity(
            id="wh_north",
            type="warehouse",
            attributes={"sqft": 50000.0, "temporal_valid_from": 0.0, "temporal_valid_until": 100.0},
            tags=("hub", "tier_1"),
        ),
        Entity(
            id="wh_south",
            type="warehouse",
            attributes={"sqft": 30000.0, "temporal_valid_from": 0.0, "temporal_valid_until": 100.0},
            tags=("spoke",),
        ),
        Entity(
            id="store_east",
            type="retail_store",
            attributes={
                "foot_traffic": 1200.0,
                "temporal_valid_from": 5.0,
                "temporal_valid_until": 50.0,
            },
            tags=("flagship",),
        ),
        Entity(
            id="truck_alpha",
            type="transport_vehicle",
            attributes={"payload_tons": 25.0},
            tags=("fleet",),
        ),
    ]

    resources = [
        Resource(
            id="inventory_north",
            entity_id="wh_north",
            current=500.0,
            min_value=0.0,
            max_value=2000.0,
        ),
        Resource(
            id="inventory_south",
            entity_id="wh_south",
            current=250.0,
            min_value=0.0,
            max_value=1000.0,
        ),
        Resource(
            id="shelf_stock", entity_id="store_east", current=40.0, min_value=0.0, max_value=200.0
        ),
        Resource(id="fuel", entity_id="truck_alpha", current=80.0, min_value=0.0, max_value=100.0),
    ]

    relationships = [
        Relationship(
            source="wh_north",
            target="wh_south",
            type="inter_warehouse_transfer",
            attributes={"distance_km": 120.0, "weight": 1.0},
        ),
        Relationship(
            source="wh_south",
            target="store_east",
            type="last_mile_delivery",
            attributes={
                "distance_km": 25.0,
                "weight": 0.8,
                "temporal_valid_from": 5.0,
                "temporal_valid_until": 50.0,
            },
        ),
        Relationship(
            source="truck_alpha",
            target="wh_north",
            type="assigned_to_hub",
            attributes={"weight": 1.0},
        ),
    ]

    return WorldState(
        step=10,
        timestamp=10.0,
        schema_version="2.0.0",
        entities=entities,
        resources=resources,
        relationships=relationships,
        memory={"last_event": "none"},
        active_rules={"surge_pricing": False},
        context={"season": "Q4"},
    )


def test_heterogeneous_graph_view_projection() -> None:
    """Assert WorldState projects into a valid HeterogeneousGraphView."""
    state = _build_test_world_state()
    graph = state.as_graph()

    assert isinstance(graph, HeterogeneousGraphView)
    assert graph.step == 10
    assert graph.timestamp == 10.0

    # Node types partitioning
    assert sorted(graph.node_types) == ["retail_store", "transport_vehicle", "warehouse"]
    assert graph.num_nodes("warehouse") == 2
    assert graph.num_nodes("retail_store") == 1
    assert graph.num_nodes("transport_vehicle") == 1
    assert graph.num_nodes() == 4

    # Edge types partitioning
    expected_edge_types = [
        "transport_vehicle__assigned_to_hub__warehouse",
        "warehouse__inter_warehouse_transfer__warehouse",
        "warehouse__last_mile_delivery__retail_store",
    ]
    assert sorted(graph.edge_types) == sorted(expected_edge_types)
    assert graph.num_edges() == 3


def test_heterogeneous_graph_features_and_indices() -> None:
    """Assert node feature matrices and edge index extraction are correct."""
    state = _build_test_world_state()
    graph = state.as_graph()

    # Extract node features for warehouses
    wh_features = graph.get_node_features(
        "warehouse", resource_keys=["inventory_north", "inventory_south"]
    )
    assert wh_features.shape == (2, 2)
    # Both warehouses present, non-zero
    assert np.all(wh_features >= 0.0)

    # Edge index for last_mile_delivery
    src_idx, dst_idx = graph.get_edge_index("warehouse__last_mile_delivery__retail_store")
    assert len(src_idx) == 1
    assert len(dst_idx) == 1
    # Warehouse south (index 1 in sorted ['wh_north', 'wh_south']) -> Store east (index 0)
    assert src_idx[0] == 1
    assert dst_idx[0] == 0

    # Edge features
    edge_features = graph.get_edge_features(
        "warehouse__last_mile_delivery__retail_store", attribute_keys=["distance_km"]
    )
    assert edge_features.shape == (1, 2)  # [weight, distance_km]
    assert edge_features[0, 0] == pytest.approx(0.8)
    assert edge_features[0, 1] == pytest.approx(25.0)


def test_heterogeneous_graph_temporal_filtering() -> None:
    """Assert temporal filtering excludes expired and not-yet-active entities and edges."""
    state = _build_test_world_state()
    graph = state.as_graph()

    # store_east and last_mile_delivery are valid in [5.0, 50.0)
    # At t=2.0, store_east should NOT be active
    graph_t2 = graph.filter_temporal(current_time=2.0)
    assert "retail_store" not in graph_t2.nodes_by_type
    assert "warehouse__last_mile_delivery__retail_store" not in graph_t2.edges_by_type

    # At t=20.0, store_east and last_mile_delivery MUST be active
    graph_t20 = graph.filter_temporal(current_time=20.0)
    assert "retail_store" in graph_t20.nodes_by_type
    assert "warehouse__last_mile_delivery__retail_store" in graph_t20.edges_by_type

    # At t=60.0, store_east has expired
    graph_t60 = graph.filter_temporal(current_time=60.0)
    assert "retail_store" not in graph_t60.nodes_by_type


def test_heterogeneous_graph_roundtrip_to_world_state() -> None:
    """Assert HeterogeneousGraphView reconstructs WorldState losslessly."""
    state = _build_test_world_state()
    graph = state.as_graph()

    reconstructed = graph.to_world_state(base_state=state)
    assert reconstructed.step == state.step
    assert reconstructed.timestamp == state.timestamp
    assert set(reconstructed.entities.keys()) == set(state.entities.keys())
    assert set(reconstructed.resources.keys()) == set(state.resources.keys())
    assert len(reconstructed.relationships) == len(state.relationships)

    for r_id, r in state.resources.items():
        assert reconstructed.resources[r_id].current == pytest.approx(r.current)
