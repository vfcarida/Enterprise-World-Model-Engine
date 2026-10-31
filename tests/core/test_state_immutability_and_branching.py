"""Tests for EWM-001: Strict Immutability, Monotonic EventLog, and Leak-Proof BranchManager."""

from __future__ import annotations

import tracemalloc
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from ewm_engine.core.action import Action, Intervention, apply_action
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import State, WorldState
from ewm_engine.core.time import (
    BranchManager,
    EventLog,
    TemporalMonotonicityError,
    collect_mutable_container_ids,
)
from ewm_engine.exceptions import InvalidWorldStateError

safe_primitives = st.one_of(
    st.booleans(),
    st.integers(min_value=-5000, max_value=5000),
    st.floats(min_value=-5000.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
    st.text(min_size=1, max_size=12),
)

nested_dict_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=8).filter(lambda k: not k.startswith("_epistemic_")),
    values=st.one_of(
        safe_primitives,
        st.lists(safe_primitives, max_size=4),
        st.dictionaries(keys=st.text(min_size=1, max_size=8), values=safe_primitives, max_size=3),
    ),
    max_size=4,
)


@settings(max_examples=30)
@given(
    nested_mem=nested_dict_strategy,
    nested_ctx=nested_dict_strategy,
)
def test_hypothesis_nested_dictionary_mutation_fails_to_corrupt_state(
    nested_mem: dict[str, Any],
    nested_ctx: dict[str, Any],
) -> None:
    """Property-based test proving external and post-transition nested dict mutations cannot corrupt State."""
    state_0 = State(
        entities=[Entity(id="node_1", type="service", attributes={"nested": {"v": 1}})],
        resources=[Resource(id="cash", current=100.0, min_value=0.0, max_value=500.0)],
        memory=nested_mem,
        context=nested_ctx,
        step=0,
    )
    fp_0 = state_0.fingerprint

    # 1. Direct attribute assignment on frozen State must fail with ValidationError
    with pytest.raises(ValidationError):
        state_0.step = 99

    # 2. Mutating caller's nested dicts after passing to with_memory / with_context must not affect state_1
    external_payload: dict[str, Any] = {"deep": {"list": [1, 2, 3], "flag": True}}
    state_1 = state_0.with_memory("payload", external_payload).with_context(
        "ctx_payload", external_payload
    )
    fp_1 = state_1.fingerprint

    external_payload["deep"]["list"].append(999)
    external_payload["deep"]["flag"] = False
    retrieved = state_1.memory["payload"]
    retrieved["deep"]["list"].clear()

    assert state_1.memory["payload"]["deep"]["list"] == [1, 2, 3]
    assert state_1.context["ctx_payload"]["deep"]["flag"] is True
    assert state_1.fingerprint == fp_1
    assert state_0.fingerprint == fp_0


def test_world_state_validators_and_query_methods_full_coverage() -> None:
    """Verify coercion validators, ontic filtering, queries, HTML rendering, and deprecated update helpers."""
    # Coerce from dicts as well as non-dict/non-list fallbacks
    state_from_dicts = WorldState(
        entities={"e1": {"id": "e1", "type": "warehouse"}, "e2": Entity(id="e2", type="store")},  # type: ignore[dict-item]
        resources={
            "r1": {"id": "r1", "current": 80.0, "min_value": 0.0, "max_value": 100.0},  # type: ignore[dict-item]
            "r2": Resource(id="r2", current=96.0, min_value=0.0, max_value=100.0),
        },
        relationships=[
            {"source": "e1", "target": "e2", "type": "supplies"},  # type: ignore[list-item]
            Relationship(source="e2", target="e1", type="returns"),
        ],
        memory={"normal": 1, "_epistemic_var": 0.25},
        context={"regime": "A", "_epistemic_note": "uncertain"},
    )
    assert len(state_from_dicts.entities) == 2
    assert len(state_from_dicts.resources) == 2
    assert len(state_from_dicts.relationships) == 2

    # Fallback branches in validators when non-collection passed
    assert WorldState._coerce_entities(None) == {}
    assert WorldState._coerce_resources(None) == {}
    assert WorldState._coerce_relationships(None) == ()

    # Ontic state stripping vs no-op clone
    ontic = state_from_dicts.to_ontic_state()
    assert "_epistemic_var" not in ontic.memory
    assert "_epistemic_note" not in ontic.context
    assert ontic.to_ontic_state().fingerprint == ontic.fingerprint
    assert state_from_dicts.ontic_fingerprint == ontic.state_hash

    # Entity, resource, and relationship lookups
    assert state_from_dicts.get_entity("e1").type == "warehouse"
    with pytest.raises(InvalidWorldStateError, match="Entity 'missing'"):
        state_from_dicts.get_entity("missing")

    assert state_from_dicts.get_resource("r1").current == 80.0
    with pytest.raises(InvalidWorldStateError, match="Resource 'missing'"):
        state_from_dicts.get_resource("missing")

    assert len(state_from_dicts.get_relationships(source="e1")) == 1
    assert len(state_from_dicts.get_relationships(target="e1")) == 1
    assert len(state_from_dicts.get_relationships(type="supplies")) == 1
    assert len(state_from_dicts.get_relationships(source="e1", target="e1")) == 0

    # Resource updates (delta, new_value, missing resource)
    s_delta = state_from_dicts.update_resource("r1", delta=10.0)
    assert s_delta.get_resource("r1").current == 90.0
    s_val = state_from_dicts.update_resource("r1", new_value=25.0)
    assert s_val.get_resource("r1").current == 25.0
    with pytest.raises(InvalidWorldStateError, match="Resource 'unknown'"):
        state_from_dicts.update_resource("unknown", delta=1.0)

    # Entity & relationship copy-on-write helpers
    s_ent = s_val.with_entity(Entity(id="e3", type="depot")).with_relationship(
        Relationship(source="e1", target="e3", type="routes")
    )
    assert "e3" in s_ent.entities
    assert len(s_ent.relationships) == 3

    # advance_time validation
    with pytest.raises(ValueError, match="strictly monotonic"):
        s_ent.advance_time(delta_t=-1.0)
    with pytest.raises(ValueError, match="strictly monotonic"):
        s_ent.advance_time(steps=0)

    # Deprecated update() and legacy_update()
    with pytest.deprecated_call():
        s_leg = s_ent.legacy_update(step=s_ent.step + 1)
    assert s_leg.step == s_ent.step + 1

    with pytest.deprecated_call():
        s_upd = s_leg.update(step=s_leg.step + 1)
    assert s_upd.step == s_leg.step + 1

    with pytest.deprecated_call(), pytest.raises(ValueError, match="retroactively decrease"):
        s_upd.legacy_update(step=0)

    # Serialization & projections & HTML rendering (empty and non-empty resources across 3 color tiers)
    dumped = s_upd.to_dict()
    restored = WorldState.from_dict(dumped)
    assert restored.fingerprint == s_upd.fingerprint
    assert s_upd.as_graph() is not None
    assert s_upd.as_scm() is not None
    html_full = s_delta._repr_html_()
    assert "WorldState" in html_full and "#f59e0b" in html_full and "#ef4444" in html_full
    html_low = s_val._repr_html_()
    assert "#3b82f6" in html_low
    html_empty = WorldState()._repr_html_()
    assert "No active resources" in html_empty


def test_action_transition_contract_and_error_guards() -> None:
    """Verify `Action + State(t) -> State(t+1)` transition contract in `core/action.py`."""
    s0 = WorldState(
        resources=[Resource(id="stock", current=50.0, min_value=0.0, max_value=100.0)],
        step=0,
        timestamp=0.0,
    )
    act = Action(id="act_rebalance", type="rebalance", parameters={"delta": 15.0})

    with pytest.raises(ValueError, match="non-negative"):
        apply_action(s0, act, delta_t=-0.5)

    # Default transition_fn=None records action id and advances step by 1
    s1_default = apply_action(s0, act)
    assert s1_default.step == 1
    assert s1_default.memory["_last_action_id"] == "act_rebalance"
    assert s0.step == 0

    # Custom transition_fn that leaves step unchanged is automatically advanced to t+1
    s1_auto = apply_action(
        s0,
        act,
        lambda st_in, a_in: st_in.update_resource("stock", delta=float(a_in.parameters["delta"])),
    )
    assert s1_auto.step == 1
    assert s1_auto.get_resource("stock").current == 65.0

    # Custom transition_fn that already advances step is preserved as-is
    s1_explicit = apply_action(
        s0,
        act,
        lambda st_in, _a: st_in.advance_time(delta_t=2.0, steps=2),
    )
    assert s1_explicit.step == 2

    # Custom transition_fn attempting retroactive step < s0.step raises ValueError
    with pytest.raises(ValueError, match="Retroactive transition forbidden"):
        apply_action(
            s1_explicit,
            act,
            lambda st_in, _a: st_in.model_copy(update={"step": 0}),
        )


def test_temporal_branch_isolation_and_monotonic_event_log() -> None:
    """Verify Temporal DAG branching, zero shared mutable `id()`s, and historical immutability."""
    # Negative initial step rejected if bypassed via model_construct
    bad_state = WorldState.model_construct(
        schema_version="1.0.0",
        entities={},
        relationships=(),
        resources={},
        memory={},
        active_rules={},
        context={},
        timestamp=0.0,
        step=-1,
    )
    with pytest.raises(TemporalMonotonicityError, match="step >= 0"):
        EventLog(bad_state)

    s0 = WorldState(
        entities=[
            Entity(id="bank", type="institution", attributes={"tier": 1}),
            Entity(id="clearing", type="hub", attributes={"tier": 0}),
        ],
        relationships=[Relationship(source="bank", target="clearing", type="settles_via")],
        resources=[Resource(id="liquidity", current=1000.0, min_value=0.0, max_value=5000.0)],
        memory={"nested": {"audit_trail": ["init"]}},
        active_rules={"threshold": 0.5},
        context={"regime": "normal"},
        step=0,
        timestamp=0.0,
    )

    manager = BranchManager(s0, root_branch_id="main", full_canonical_hashes=True)
    assert manager.root_branch_id == "main"
    assert manager.branch_ids == ("main",)

    # Advance main branch to step 2
    manager.step_branch(
        "main",
        Action(id="a1", type="deposit", parameters={"amount": 100.0}),
        lambda st_in, a: st_in.update_resource("liquidity", delta=a.parameters["amount"]),
    )
    manager.step_branch(
        "main",
        Action(id="a2", type="deposit", parameters={"amount": 200.0}),
        lambda st_in, a: st_in.update_resource("liquidity", delta=a.parameters["amount"]),
    )
    assert manager.get_head_state("main").step == 2
    assert manager.get_head_state("main").get_resource("liquidity").current == 1300.0

    # Spawn Branch A at historical step 1 with an Intervention, and Branch B at step 2 with an Action
    interv = Intervention(
        id="tighten_policy",
        description="Raise reserve rule",
        parameters={"threshold": 0.8},
    )
    branch_a_root = manager.spawn_branch(
        "branch_a",
        parent_branch_id="main",
        at_step=1,
        intervention=interv,
    )
    assert branch_a_root.step == 1
    assert branch_a_root.active_rules["tighten_policy"] == {"threshold": 0.8}

    branch_b_root = manager.spawn_branch(
        "branch_b",
        parent_branch_id="main",
        at_step=2,
        intervention=Action(id="fork_act", type="shock"),
    )
    assert branch_b_root.step == 2
    assert branch_b_root.memory["_fork_action_id"] == "fork_act"

    branch_c_root = manager.spawn_branch(
        "branch_c", parent_branch_id="main", at_step=1, intervention=None
    )
    assert branch_c_root.step == 1

    # Mutate Branch A via transition and mutate returned dictionary objects
    s_a2 = manager.step_branch(
        "branch_a",
        Action(id="a_drain", type="withdraw"),
        lambda st_in, _a: st_in.update_resource("liquidity", delta=-500.0).with_memory(
            "nested", {"audit_trail": ["branch_a_mutated"]}
        ),
    )
    s_a2.memory["nested"]["audit_trail"].append("external_tamper")

    # Assert Branch B and main historical states (t=0, t=1, t=2) are 100% unaffected
    assert manager.get_state_at("main", 0).get_resource("liquidity").current == 1000.0
    assert manager.get_state_at("main", 1).get_resource("liquidity").current == 1100.0
    assert manager.get_state_at("main", 2).get_resource("liquidity").current == 1300.0
    assert manager.get_state_at("branch_a", 0).get_resource("liquidity").current == 1000.0
    assert manager.get_head_state("branch_a").get_resource("liquidity").current == 600.0
    assert manager.get_head_state("branch_b").get_resource("liquidity").current == 1300.0
    assert manager.get_head_state("main").memory["nested"]["audit_trail"] == ["init"]

    # Prove zero shared mutable container id()s across main, branch_a, and branch_b
    assert manager.verify_branch_isolation("main", "branch_a") is True
    assert manager.verify_branch_isolation("main", "branch_b") is True
    assert manager.verify_branch_isolation("branch_a", "branch_b") is True
    assert collect_mutable_container_ids(manager.get_head_state("branch_a")).isdisjoint(
        collect_mutable_container_ids(manager.get_head_state("branch_b"))
    )

    # Verify DAG metadata & error guards
    dag = manager.get_dag()
    assert set(dag.keys()) == {"main", "branch_a", "branch_b", "branch_c"}
    assert dag["main"].child_branch_ids == ("branch_a", "branch_b", "branch_c")
    assert dag["branch_a"].parent_branch_id == "main"
    assert dag["branch_a"].fork_step == 1

    log_main = manager.get_log("main")
    assert log_main.branch_id == "main"
    assert log_main.initial_step == 0
    assert len(log_main) == 3
    assert log_main.steps == (0, 1, 2)
    assert len(log_main.events) == 2

    with pytest.raises(TemporalMonotonicityError, match="strictly greater"):
        log_main.append(s0)
    with pytest.raises(TemporalMonotonicityError, match="Retrograde timestamp"):
        log_main.append(log_main.head_state.model_copy(update={"step": 3, "timestamp": -5.0}))
    with pytest.raises(KeyError, match="Step 99"):
        log_main.get_state(99)
    with pytest.raises(KeyError, match="Unknown simulation branch"):
        manager.get_log("nonexistent")
    with pytest.raises(KeyError, match="Unknown simulation branch"):
        manager.get_branch_metadata("nonexistent")
    with pytest.raises(ValueError, match="already exists"):
        manager.spawn_branch("branch_a")
    with pytest.raises(TemporalMonotonicityError, match="future step"):
        manager.spawn_branch("branch_future", at_step=99)
    with pytest.raises(ValueError, match="compare a branch with itself"):
        manager.verify_branch_isolation("main", "main")

    # Test ancestor lookup before root step raises KeyError
    branch_from_1 = EventLog(s0.advance_time(steps=2), branch_id="late_root")
    assert branch_from_1.initial_step == 2
    late_mgr = BranchManager(s0.advance_time(steps=2), root_branch_id="late")
    with pytest.raises(KeyError, match="precedes root initial step"):
        late_mgr.get_state_at("late", 0)


def test_ten_thousand_step_rollout_memory_safeguard() -> None:
    """Profile 10,000+ step rollout in BranchManager to verify OOM safeguard (< 50 MB peak)."""
    s0 = WorldState(
        entities=[Entity(id="srv_1", type="compute", attributes={"region": "us-east"})],
        resources=[Resource(id="cpu_credits", current=10000.0, min_value=0.0, max_value=50000.0)],
        memory={"window": [1, 2, 3]},
        step=0,
        timestamp=0.0,
    )
    tracemalloc.start()
    mgr = BranchManager(s0, root_branch_id="long_horizon", full_canonical_hashes=False)
    noop = Action(id="act_tick", type="tick")
    for _ in range(10_050):
        mgr.step_branch(
            "long_horizon",
            noop,
            lambda st_in, _a: st_in.update_resource("cpu_credits", delta=1.0),
            retain_intermediate_states=False,
        )
    _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_bytes / (1024 * 1024)
    assert mgr.get_head_state("long_horizon").step == 10_050
    assert mgr.get_head_state("long_horizon").get_resource("cpu_credits").current == 20_050.0
    assert len(mgr.get_log("long_horizon").events) == 10_050
    assert peak_mb < 50.0, f"10,050-step rollout exceeded 50 MB memory budget: {peak_mb:.2f} MB"
