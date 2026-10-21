"""Unit tests for EWM Concordia Game-Master bridge.

Tests intent-to-action parsing, constraint arbitration, state advancement,
canonical fingerprint transition, and scientific grounding citations.
"""

from __future__ import annotations

from collections.abc import Sequence

from ewm_engine.constraints.base import Constraint
from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.core.actions import Action
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.integrations.concordia import ArbiterVerdict, EWMWorldArbiter
from ewm_engine.provenance.evidence import EvidenceLevel


class MockViolatingConstraint:
    """Mock constraint that always flags a hard violation."""

    id = "mock_conservation_invariant"
    constraint_id = "mock_conservation_invariant"
    version = "1.0.0"
    severity = ConstraintSeverity.HARD
    description = "Mock conservation invariant"

    def check(self, state: WorldState) -> ConstraintResult:
        return ConstraintResult(
            satisfied=False,
            constraint_id="mock_conservation_invariant",
            severity=ConstraintSeverity.HARD,
            message="Resource balance conservation violated: illegal deficit.",
        )

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase = ConstraintPhase.POST_TRANSITION,
    ) -> ConstraintResult:
        return self.check(state)


def _make_test_world(with_violation: bool = False) -> World:
    state = WorldState(
        step=0,
        timestamp=0.0,
        entities=[],
        resources=[
            Resource(id="stock_a", current=100.0, min_value=0.0, max_value=500.0),
            Resource(id="stock_b", current=50.0, min_value=0.0, max_value=500.0),
        ],
    )
    constraints: list[Constraint] = [MockViolatingConstraint()] if with_violation else []
    return World(state=state, constraints=constraints)


def test_intent_parsing_natural_language() -> None:
    """Verify natural language heuristic extraction parses resource transfers."""
    world = _make_test_world()
    arbiter = EWMWorldArbiter(world)

    # 1. Natural language transfer regex
    act = arbiter.parse_intent_to_action(
        agent_id="agent_alice",
        intent="Please transfer 42.5 from stock_a to stock_b now",
    )
    assert act.actor_id == "agent_alice"
    assert act.type == "transfer_resource"
    assert act.parameters["quantity"] == 42.5
    assert act.parameters["source_resource"] == "stock_a"
    assert act.parameters["target_resource"] == "stock_b"

    # 2. Structured dict intent
    dict_intent = {
        "id": "act_custom",
        "type": "negotiate_contract",
        "parameters": {"price": 100},
    }
    act_dict = arbiter.parse_intent_to_action("agent_bob", dict_intent)
    assert act_dict.actor_id == "agent_bob"
    assert act_dict.type == "negotiate_contract"
    assert act_dict.parameters["price"] == 100

    # 3. Free text fallback intent
    act_free = arbiter.parse_intent_to_action("agent_carol", "Let us inspect the quarterly report")
    assert act_free.type == "generic_intent"
    assert act_free.parameters["raw_intent"] == "Let us inspect the quarterly report"


def test_arbitration_approval_advances_world() -> None:
    """Verify approved action steps world, computes observation, and transitions fingerprint."""
    world = _make_test_world(with_violation=False)
    fp_start = world.initial_state.fingerprint
    arbiter = EWMWorldArbiter(world)

    verdict = arbiter.arbitrate_intent(
        agent_id="agent_1",
        intent="transfer 20 from stock_a to stock_b",
    )
    assert isinstance(verdict, ArbiterVerdict)
    assert verdict.status == "APPROVED"
    assert verdict.is_valid is True
    assert len(verdict.violations) == 0
    assert verdict.state_fingerprint_before == fp_start
    assert verdict.state_fingerprint_after != fp_start
    assert verdict.evidence_level == EvidenceLevel.STRUCTURAL
    assert len(verdict.trace_id) == 64
    assert "Action 'transfer_resource' executed successfully" in verdict.grounded_observation
    assert world.initial_state.step == 1


def test_arbitration_rejection_preserves_world_state() -> None:
    """Verify rejected action halts transition and leaves world state untouched."""
    world = _make_test_world(with_violation=True)
    fp_start = world.initial_state.fingerprint
    arbiter = EWMWorldArbiter(world)

    verdict = arbiter.arbitrate_intent(
        agent_id="agent_malicious",
        intent="transfer 9999 from stock_a to stock_b",
    )
    assert verdict.status == "REJECTED"
    assert verdict.is_valid is False
    assert len(verdict.violations) == 1
    assert verdict.state_fingerprint_before == fp_start
    assert verdict.state_fingerprint_after == fp_start
    assert "rejected by World Arbiter" in verdict.grounded_observation
    assert world.initial_state.step == 0


def test_scientific_citations_grounding() -> None:
    """Verify required scientific literature citations are present in module docstring."""
    from ewm_engine.integrations import concordia

    doc = concordia.__doc__ or ""
    assert "2406.03689" in doc  # Vafa et al.
    assert "2411.02385" in doc  # Kang et al.
    assert "2410.19923" in doc  # Gkountouras et al.
