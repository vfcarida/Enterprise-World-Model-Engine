"""Unit tests for constraints, registry validation, and provenance tracking."""

from __future__ import annotations

from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.constraints.standard import (
    ActionTransferAvailabilityConstraint,
    ResourceCapacityConstraint,
    ResourceNonNegativeConstraint,
)
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState


def test_capacity_constraint(sample_world_state: WorldState) -> None:
    """Test ResourceCapacityConstraint evaluation under normal and overflowing states."""
    constraint = ResourceCapacityConstraint(resource_id="stock_wh1")

    # Within bounds (current=100, max=200)
    res_ok = constraint.evaluate(sample_world_state, phase=ConstraintPhase.POST_TRANSITION)
    assert res_ok.satisfied
    assert res_ok.penalty == 0.0
    assert res_ok.phase == ConstraintPhase.POST_TRANSITION

    # Over capacity (current=250, max=200)
    overflow_state = sample_world_state.update_resource("stock_wh1", new_value=250.0, clamp=False)
    res_overflow = constraint.evaluate(overflow_state, phase=ConstraintPhase.POST_TRANSITION)
    assert not res_overflow.satisfied
    assert res_overflow.severity == ConstraintSeverity.HARD
    assert res_overflow.penalty == 50.0
    assert "stock_wh1" in res_overflow.violating_resources
    assert res_overflow.violating_values["current"] == 250.0


def test_non_negative_constraint(sample_world_state: WorldState) -> None:
    """Test ResourceNonNegativeConstraint under negative levels."""
    constraint = ResourceNonNegativeConstraint(resource_id="stock_wh1")

    negative_state = sample_world_state.update_resource("stock_wh1", new_value=-15.0, clamp=False)
    res = constraint.evaluate(negative_state, phase=ConstraintPhase.POST_TRANSITION)
    assert not res.satisfied
    assert res.penalty == 15.0
    assert res.violating_values["current"] == -15.0
    assert res.phase == ConstraintPhase.POST_TRANSITION


def test_transfer_availability_phase_awareness(sample_world_state: WorldState) -> None:
    """ActionTransferAvailabilityConstraint operates strictly during PRE_ACTION phase."""
    constraint = ActionTransferAvailabilityConstraint(action_type="transfer_resource")
    action = Action(
        id="act_transfer",
        type="transfer_resource",
        parameters={"source_resource": "stock_wh1", "quantity": 999.0},
    )

    # In PRE_ACTION phase, excessive transfer is caught and rejected
    res_pre = constraint.evaluate(
        sample_world_state, actions=[action], phase=ConstraintPhase.PRE_ACTION
    )
    assert not res_pre.satisfied
    assert res_pre.severity == ConstraintSeverity.HARD
    assert res_pre.phase == ConstraintPhase.PRE_ACTION

    # In POST_TRANSITION phase, action constraint is inactive (returns satisfied)
    res_post = constraint.evaluate(
        sample_world_state, actions=[action], phase=ConstraintPhase.POST_TRANSITION
    )
    assert res_post.satisfied
    assert res_post.phase == ConstraintPhase.POST_TRANSITION


def test_constraint_result_entity_ids_sync() -> None:
    """ConstraintResult maintains bidirectional synchronization between entity_ids and violating_entities."""
    res1 = ConstraintResult(
        satisfied=False,
        constraint_id="c1",
        entity_ids=("warehouse_a", "warehouse_b"),
    )
    assert res1.violating_entities == ("warehouse_a", "warehouse_b")
    assert res1.entity_ids == ("warehouse_a", "warehouse_b")

    res2 = ConstraintResult(
        satisfied=False,
        constraint_id="c2",
        violating_entities=("hospital_1",),
    )
    assert res2.entity_ids == ("hospital_1",)
    assert res2.violating_entities == ("hospital_1",)


def test_action_pre_validation_in_registry(sample_world_state: WorldState) -> None:
    """Test that ConstraintRegistry filters out actions breaching HARD constraints."""
    registry = ConstraintRegistry(
        [
            ActionTransferAvailabilityConstraint(action_type="transfer_resource"),
        ]
    )

    valid_action = Action(
        id="act_valid",
        type="transfer_resource",
        parameters={"source_resource": "stock_wh1", "quantity": 50.0},
    )
    excessive_action = Action(
        id="act_excessive",
        type="transfer_resource",
        parameters={
            "source_resource": "stock_wh1",
            "quantity": 999.0,
        },  # Exceeds available stock (100)
    )

    accepted, results = registry.validate_actions(
        state=sample_world_state,
        actions=[valid_action, excessive_action],
    )

    assert len(accepted) == 1
    assert accepted[0].id == "act_valid"
    assert len(results) == 1
    assert not results[0].satisfied
    assert results[0].phase == ConstraintPhase.PRE_ACTION
    assert results[0].preceding_action_id == "act_excessive"


def test_validate_state_multi_action_attribution(sample_world_state: WorldState) -> None:
    """Verify that validate_state attributes violations to the specific action touching the resource."""
    registry = ConstraintRegistry(
        [
            ResourceCapacityConstraint(resource_id="stock_wh2"),
        ]
    )

    action_1 = Action(
        id="act_wh1",
        type="transfer_resource",
        parameters={"target_resource": "stock_wh1", "quantity": 10.0},
    )
    action_2 = Action(
        id="act_wh2",
        type="transfer_resource",
        parameters={"target_resource": "stock_wh2", "quantity": 300.0},
    )

    # State where stock_wh2 exceeds capacity
    overflow_state = sample_world_state.update_resource("stock_wh2", new_value=250.0, clamp=False)
    results = registry.validate_state(overflow_state, preceding_actions=[action_1, action_2])

    assert len(results) == 1
    assert not results[0].satisfied
    assert results[0].phase == ConstraintPhase.POST_TRANSITION
    # Must be attributed to action_2 which touched stock_wh2, not action_1
    assert results[0].preceding_action_id == "act_wh2"
    assert results[0].metadata["preceding_action_ids"] == ["act_wh1", "act_wh2"]
