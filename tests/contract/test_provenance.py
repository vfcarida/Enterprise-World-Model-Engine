"""Contract tests for AC-010: Provenance configuration, canonical serialization, and fingerprint stability."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.exceptions import InvalidWorldStateError
from ewm_engine.provenance.metadata import ComponentVersion, Provenance, SimulationMetadata
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario


@pytest.mark.contract
def test_provenance_minimal_required_configuration() -> None:
    """AC-010: Provenance model contains all spec-mandated fields and component versions."""
    prov = Provenance(
        scenario_fingerprint="f" * 64,
        initial_state_fingerprint="0" * 64,
        seed=42,
        horizon=10,
        samples=5,
        components=[
            ComponentVersion(
                component_id="DeterministicTransferDynamics", component_version="1.0.0"
            ),
            ComponentVersion(component_id="WeatherEventSource", component_version="0.2.1"),
        ],
        constraint_versions=[
            ComponentVersion(
                component_id="ActionTransferAvailabilityConstraint", component_version="1.0.0"
            ),
        ],
        runtime_metadata={"host": "worker-1", "scenario_id": "test_run"},
    )

    # Required fields must exist and conform
    assert prov.scenario_fingerprint == "f" * 64
    assert prov.initial_state_fingerprint == "0" * 64
    assert prov.seed == 42
    assert prov.horizon == 10
    assert prov.samples == 5
    assert len(prov.components) == 2
    assert prov.components[0].component_id == "DeterministicTransferDynamics"
    assert len(prov.constraint_versions) == 1
    assert prov.constraint_versions[0].component_id == "ActionTransferAvailabilityConstraint"
    assert prov.runtime_metadata["host"] == "worker-1"
    assert len(prov.fingerprint) == 64
    assert prov.engine_version != ""

    # Backward-compatible property aliases
    assert prov.world_hash == prov.initial_state_fingerprint
    assert prov.random_seed == prov.seed
    assert prov.scenario_id == "test_run"
    assert prov.custom_metadata == prov.runtime_metadata


@pytest.mark.contract
def test_simulation_metadata_backward_compatibility() -> None:
    """AC-010: SimulationMetadata continues to support legacy kwargs and behaves as a Provenance subclass."""
    metadata = SimulationMetadata(
        scenario_id="scenario_alpha",
        world_hash="a" * 64,
        dynamics_name="DeterministicTransferDynamics",
        constraint_versions={"capacity_limit": "1.0.0"},
        random_seed=42,
        horizon=10,
        samples=5,
    )

    assert metadata.scenario_id == "scenario_alpha"
    assert metadata.world_hash == "a" * 64
    assert metadata.initial_state_fingerprint == "a" * 64
    assert metadata.random_seed == 42
    assert metadata.seed == 42
    assert metadata.horizon == 10
    assert metadata.samples == 5
    assert len(metadata.fingerprint) == 64
    assert len(metadata.constraint_versions) == 1
    assert metadata.constraint_versions[0].component_id == "capacity_limit"
    assert isinstance(metadata, Provenance)


@pytest.mark.contract
def test_provenance_components_and_constraints_order_independence() -> None:
    """AC-010: Provenance fingerprint is invariant to the ordering of components and constraint versions."""
    c1 = ComponentVersion(component_id="DynamicsA", component_version="1.0.0")
    c2 = ComponentVersion(component_id="DynamicsB", component_version="2.0.0")

    k1 = ComponentVersion(component_id="ConstraintA", component_version="1.0.0")
    k2 = ComponentVersion(component_id="ConstraintB", component_version="1.0.0")

    prov1 = Provenance(
        scenario_fingerprint="1" * 64,
        initial_state_fingerprint="2" * 64,
        seed=10,
        horizon=5,
        samples=1,
        components=[c1, c2],
        constraint_versions=[k1, k2],
    )
    prov2 = Provenance(
        scenario_fingerprint="1" * 64,
        initial_state_fingerprint="2" * 64,
        seed=10,
        horizon=5,
        samples=1,
        components=[c2, c1],
        constraint_versions=[k2, k1],
    )

    assert prov1.fingerprint == prov2.fingerprint


@pytest.mark.contract
def test_branch_initial_state_fingerprint_sharing(sample_world_state: WorldState) -> None:
    """AC-010 & AC-005: Equal initial states across simulation branches share initial_state_fingerprint."""
    world_base = sample_world_state
    world_branch1 = sample_world_state.model_copy()
    world_branch2 = WorldState(
        entities=list(sample_world_state.entities.values()),
        resources=list(sample_world_state.resources.values()),
        relationships=list(sample_world_state.relationships),
        memory=sample_world_state.memory,
        active_rules=sample_world_state.active_rules,
        context=sample_world_state.context,
    )

    # Identical state content produces identical initial_state_fingerprint
    assert world_base.fingerprint == world_branch1.fingerprint
    assert world_base.fingerprint == world_branch2.fingerprint

    # Modifying state produces different fingerprint
    altered_state = world_base.update_resource("stock_wh1", delta=50.0)
    assert altered_state.fingerprint != world_base.fingerprint


@pytest.mark.contract
def test_simulation_result_exposes_provenance(sample_world_state: WorldState) -> None:
    """SimulationResult exposes .provenance conforming to the new Provenance model."""
    from ewm_engine.core.world import World
    from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics

    world = World(state=sample_world_state, dynamics=DeterministicTransferDynamics())
    scenario = Scenario(name="ProvenanceCheck", horizon=2, samples=1, seed=99)
    engine = SimulationEngine()
    result = engine.run(world=world, scenario=scenario)

    assert hasattr(result, "provenance")
    assert isinstance(result.provenance, Provenance)
    assert result.provenance.scenario_fingerprint == scenario.fingerprint
    assert result.provenance.initial_state_fingerprint == sample_world_state.fingerprint
    assert result.provenance.seed == 99
    assert len(result.provenance.components) >= 1
    assert result.provenance.components[0].component_id == "TransferDynamics"


@pytest.mark.contract
def test_fingerprint_insertion_order_independence() -> None:
    """AC-010: State fingerprints are independent of dictionary insertion order."""
    dict_order_1 = {"z_first": 1, "a_second": 2, "m_third": {"nested_z": 10, "nested_a": 20}}
    dict_order_2 = {"a_second": 2, "m_third": {"nested_a": 20, "nested_z": 10}, "z_first": 1}

    state1 = WorldState(
        memory=dict_order_1,
        context={"temp": 25.0, "humidity": 60.0},
        active_rules={"rule_b": False, "rule_a": True},
    )
    state2 = WorldState(
        memory=dict_order_2,
        context={"humidity": 60.0, "temp": 25.0},
        active_rules={"rule_a": True, "rule_b": False},
    )

    assert state1.fingerprint == state2.fingerprint
    assert state1.state_hash == state2.state_hash


@pytest.mark.contract
def test_entities_and_resources_order_independence() -> None:
    """Entities, relationships, and resources passed in different orders produce identical fingerprints."""
    e1 = Entity(id="ent_1", type="node", attributes={"region": "east"})
    e2 = Entity(id="ent_2", type="node", attributes={"region": "west"})

    r1 = Resource(id="res_1", current=10.0, min_value=0.0, max_value=50.0)
    r2 = Resource(id="res_2", current=20.0, min_value=0.0, max_value=100.0)

    rel1 = Relationship(source="ent_1", target="ent_2", type="link")
    rel2 = Relationship(source="ent_2", target="ent_1", type="link")

    state_forward = WorldState(
        entities=[e1, e2],
        resources=[r1, r2],
        relationships=[rel1, rel2],
    )
    state_reversed = WorldState(
        entities=[e2, e1],
        resources=[r2, r1],
        relationships=[rel2, rel1],
    )

    assert state_forward.fingerprint == state_reversed.fingerprint
    assert state_forward.state_hash == state_reversed.state_hash


@pytest.mark.contract
def test_non_finite_float_rejection() -> None:
    """AC-010: WorldState and canonical serialization reject NaN and Infinity with InvalidWorldStateError."""
    # NaN in memory
    state_nan_mem = WorldState(memory={"bad": float("nan")})
    with pytest.raises(InvalidWorldStateError, match="Canonical serialization rejects non-finite"):
        _ = state_nan_mem.fingerprint

    # Inf in context
    state_inf_ctx = WorldState(context={"diverged": float("inf")})
    with pytest.raises(InvalidWorldStateError, match="Canonical serialization rejects non-finite"):
        _ = state_inf_ctx.fingerprint

    # -Inf in active_rules
    state_neginf_rules = WorldState(active_rules={"bound": float("-inf")})
    with pytest.raises(InvalidWorldStateError, match="Canonical serialization rejects non-finite"):
        _ = state_neginf_rules.state_hash

    # NaN in entity attributes
    state_nan_entity = WorldState(
        entities=[Entity(id="e1", type="node", attributes={"loss": float("nan")})]
    )
    with pytest.raises(InvalidWorldStateError, match="Canonical serialization rejects non-finite"):
        _ = state_nan_entity.fingerprint

    # Inf in resource current
    state_inf_res = WorldState(
        resources=[Resource(id="r1", current=float("inf"), min_value=0.0, max_value=100.0)]
    )
    with pytest.raises(InvalidWorldStateError, match="Canonical serialization rejects non-finite"):
        _ = state_inf_res.fingerprint

    # NaN in timestamp
    state_nan_ts = WorldState(timestamp=float("nan"))
    with pytest.raises(InvalidWorldStateError, match="Canonical serialization rejects non-finite"):
        _ = state_nan_ts.fingerprint


@pytest.mark.contract
def test_float_precision_determinism() -> None:
    """Canonical serialization rounds floats to 6 decimal places deterministically."""
    state_a = WorldState(context={"rate": 1.2345671})
    state_b = WorldState(context={"rate": 1.2345674})
    assert state_a.fingerprint == state_b.fingerprint

    state_c = WorldState(context={"rate": 1.234568})
    assert state_a.fingerprint != state_c.fingerprint


@pytest.mark.contract
def test_utc_timestamp_normalization() -> None:
    """Timestamps and datetimes are normalized to UTC so timezone-shifted equivalents match."""
    dt_utc = datetime(2026, 10, 2, 12, 0, 0, tzinfo=UTC)
    dt_offset = datetime(2026, 10, 2, 15, 0, 0, tzinfo=timezone(timedelta(hours=3)))
    dt_naive = datetime(2026, 10, 2, 12, 0, 0)

    hash_utc = canonical_sha256({"time": dt_utc})
    hash_offset = canonical_sha256({"time": dt_offset})
    hash_naive = canonical_sha256({"time": dt_naive})

    assert hash_utc == hash_offset
    assert hash_utc == hash_naive


@pytest.mark.contract
def test_state_fingerprint_determinism_across_constructions() -> None:
    """Same logical state constructed separately produces identical fingerprint."""
    s1 = WorldState(
        entities=[Entity(id="a", type="wh", attributes={"cap": 100})],
        resources=[Resource(id="r", current=50.0, min_value=0.0, max_value=100.0)],
        memory={"step_counter": 5},
        context={"temp": 21.0},
        active_rules={"mode": "standard"},
        timestamp=5.0,
        step=5,
    )
    s2 = WorldState(
        entities=[Entity(id="a", type="wh", attributes={"cap": 100})],
        resources=[Resource(id="r", current=50.0, min_value=0.0, max_value=100.0)],
        memory={"step_counter": 5},
        context={"temp": 21.0},
        active_rules={"mode": "standard"},
        timestamp=5.0,
        step=5,
    )

    assert s1.fingerprint == s2.fingerprint
    assert s1.state_hash == s2.state_hash
