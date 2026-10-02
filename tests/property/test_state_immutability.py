"""Property-based tests verifying deep immutability and alias-isolation for WorldState."""

from __future__ import annotations

import copy
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from ewm_engine.core.actions import Action, Intervention
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState

# Strategy for safe, finite JSON-like primitive values
safe_primitives = st.one_of(
    st.booleans(),
    st.integers(min_value=-10000, max_value=10000),
    st.floats(min_value=-10000.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
    st.text(min_size=1, max_size=20),
)

# Strategy for nested dictionaries
safe_nested_dicts = st.dictionaries(
    keys=st.text(min_size=1, max_size=10),
    values=st.one_of(
        safe_primitives,
        st.lists(safe_primitives, max_size=5),
        st.dictionaries(keys=st.text(min_size=1, max_size=10), values=safe_primitives, max_size=3),
    ),
    max_size=5,
)


@settings(max_examples=40)
@given(
    initial_memory=safe_nested_dicts,
    initial_context=safe_nested_dicts,
    initial_active_rules=safe_nested_dicts,
    entity_attrs=safe_nested_dicts,
)
def test_world_state_construction_defensive_copying(
    initial_memory: dict[str, Any],
    initial_context: dict[str, Any],
    initial_active_rules: dict[str, Any],
    entity_attrs: dict[str, Any],
) -> None:
    """Mutating source dictionaries after constructing WorldState must not alter stored state."""
    # Retain deep copies to verify against
    expected_memory = copy.deepcopy(initial_memory)
    expected_context = copy.deepcopy(initial_context)
    expected_active_rules = copy.deepcopy(initial_active_rules)
    expected_entity_attrs = copy.deepcopy(entity_attrs)

    entity = Entity(id="ent_1", type="node", attributes=entity_attrs)
    resource = Resource(id="res_1", current=100.0, min_value=0.0, max_value=200.0)

    state = WorldState(
        entities=[entity],
        resources=[resource],
        memory=initial_memory,
        context=initial_context,
        active_rules=initial_active_rules,
    )
    baseline_fingerprint = state.fingerprint

    # Mutate source objects externally
    initial_memory["__mutated_key__"] = "contaminated"
    for v in initial_memory.values():
        if isinstance(v, dict):
            v["__nested_mutated__"] = 999
        elif isinstance(v, list):
            v.append("contaminated_element")

    initial_context["__mutated_context__"] = True
    initial_active_rules["__mutated_rules__"] = 12345
    entity_attrs["__mutated_attr__"] = "dirty"

    # Invariants: internal state and fingerprint must remain perfectly intact
    assert state.memory == expected_memory
    assert state.context == expected_context
    assert state.active_rules == expected_active_rules
    assert state.get_entity("ent_1").attributes == expected_entity_attrs
    assert state.fingerprint == baseline_fingerprint
    assert state.state_hash == baseline_fingerprint


@settings(max_examples=40)
@given(
    mem=safe_nested_dicts,
    ctx=safe_nested_dicts,
    rules=safe_nested_dicts,
)
def test_world_state_accessor_defensive_copying(
    mem: dict[str, Any],
    ctx: dict[str, Any],
    rules: dict[str, Any],
) -> None:
    """Mutating values retrieved from state accessors must not alter stored state or fingerprint."""
    state = WorldState(
        entities=[Entity(id="e1", type="server", attributes={"load": 0.5})],
        resources=[Resource(id="r1", current=50.0, min_value=0.0, max_value=100.0)],
        memory=mem,
        context=ctx,
        active_rules=rules,
    )
    baseline_fp = state.fingerprint

    # Retrieve and mutate memory
    retrieved_mem = state.memory
    retrieved_mem["_injected_"] = "hacked"
    for val in retrieved_mem.values():
        if isinstance(val, dict):
            val["_injected_"] = -1
        elif isinstance(val, list):
            val.clear()

    # Retrieve and mutate context
    retrieved_ctx = state.context
    retrieved_ctx["_injected_ctx_"] = "corrupted"

    # Retrieve and mutate active_rules
    retrieved_rules = state.active_rules
    retrieved_rules["_injected_rules_"] = 999999

    # Retrieve and mutate entities map
    retrieved_entities = state.entities
    retrieved_entities["e2"] = Entity(id="e2", type="rogue")

    # Retrieve and mutate entity attributes
    ent = state.get_entity("e1")
    ent.attributes["load"] = 999.0
    ent.attributes["extra"] = "forbidden"

    # Invariants: state remains strictly unaffected
    assert "_injected_" not in state.memory
    assert "_injected_ctx_" not in state.context
    assert "_injected_rules_" not in state.active_rules
    assert "e2" not in state.entities
    assert state.get_entity("e1").attributes.get("load") == 0.5
    assert "extra" not in state.get_entity("e1").attributes
    assert state.fingerprint == baseline_fp
    assert state.state_hash == baseline_fp


def test_domain_primitives_defensive_copying() -> None:
    """Confirm Entity, Relationship, Action, Event, and Intervention protect internal mapping fields."""
    # Entity
    e_attrs: dict[str, Any] = {"tags": ["a", "b"], "meta": {"k": "v"}}
    e = Entity(id="e1", type="unit", attributes=e_attrs)
    list(e_attrs["tags"]).append("c")
    e.attributes["meta"]["k"] = "mutated"
    assert e.attributes["tags"] == ["a", "b"]
    assert e.attributes["meta"]["k"] == "v"

    # Relationship
    r_attrs: dict[str, Any] = {"cost": 10.0, "details": {"type": "fiber"}}
    r = Relationship(source="e1", target="e1", type="loop", attributes=r_attrs)
    r_attrs["cost"] = 99.0
    r.attributes["details"]["type"] = "copper"
    assert r.attributes["cost"] == 10.0
    assert r.attributes["details"]["type"] == "fiber"

    # Action
    a_params: dict[str, Any] = {"qty": 100, "meta": {"route": "fast"}}
    a = Action(id="a1", type="move", actor_id="agt", parameters=a_params)
    a_params["qty"] = 0
    a.parameters["meta"]["route"] = "slow"
    assert a.parameters["qty"] == 100
    assert a.parameters["meta"]["route"] == "fast"

    # ExogenousEvent
    ev_params: dict[str, Any] = {"scale": 2.5, "flags": ["alert"]}
    ev = ExogenousEvent(id="ev1", type="storm", severity=2.5, parameters=ev_params)
    ev_params["scale"] = 9.9
    flags_val = ev.parameters["flags"]
    if isinstance(flags_val, list):
        flags_val.clear()
    assert ev.parameters["scale"] == 2.5
    assert ev.parameters["flags"] == ["alert"]

    # Intervention
    iv_params: dict[str, Any] = {"rate": 0.05, "nested": {"active": True}}
    iv = Intervention(id="iv1", description="tax", parameters=iv_params)
    iv_params["rate"] = 0.50
    iv.parameters["nested"]["active"] = False
    assert iv.parameters["rate"] == 0.05
    assert iv.parameters["nested"]["active"] is True
