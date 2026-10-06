"""Unit tests for multi-agent observation views, action proposals, and deterministic mediation (T6)."""

from __future__ import annotations

from collections.abc import Sequence

from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.core.actions import Action
from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import ConstraintId
from ewm_engine.multiagent.mediator import ConstraintMediator
from ewm_engine.multiagent.views import (
    ActorActionProposal,
    ActorObservationView,
    AdjudicationResult,
)


class MockActionConstraint:
    """Hard constraint rejecting action with name 'forbidden'."""

    def __init__(self, id: str = "no_forbidden") -> None:
        self._id = id

    @property
    def constraint_id(self) -> ConstraintId:
        return self._id

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def severity(self) -> ConstraintSeverity:
        return ConstraintSeverity.HARD

    @property
    def description(self) -> str:
        return "Rejects forbidden actions"

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase = ConstraintPhase.PRE_ACTION,
    ) -> ConstraintResult:
        for act in actions:
            if "forbidden" in act.id:
                return ConstraintResult(
                    satisfied=False,
                    constraint_id=self.constraint_id,
                    constraint_version=self.version,
                    severity=self.severity,
                    message="Forbidden action requested",
                )
        return ConstraintResult(
            satisfied=True,
            constraint_id=self.constraint_id,
            constraint_version=self.version,
            severity=self.severity,
        )


def test_actor_observation_view_filtering() -> None:
    """Test ActorObservationView filters entities, resources, and memory based on access."""
    state = WorldState(
        step=3,
        timestamp=3.0,
        entities={"e1": Entity(id="e1", type="node"), "e2": Entity(id="e2", type="secret")},
        resources={
            "public_gold": Resource(id="public_gold", current=100.0),
            "vault": Resource(id="vault", current=999.0),
        },
        memory={"public_info": "hello", "secret_key": "xyz"},
    )

    view = ActorObservationView.from_world_state(
        state,
        actor_id="agent_1",
        entity_filter=["e1"],
        resource_filter=["public_gold"],
        memory_filter=["public_info"],
    )

    assert view.actor_id == "agent_1"
    assert view.step == 3
    assert len(view.visible_entities) == 1
    assert view.visible_entities[0].id == "e1"
    assert "public_gold" in view.visible_resources
    assert "vault" not in view.visible_resources
    assert view.visible_memory == {"public_info": "hello"}


def test_constraint_mediator_priorities_and_rejections() -> None:
    """Test ConstraintMediator orders by priority, rejects hard constraint violations, and enforces resource locks."""
    state = WorldState(
        step=1,
        resources={"budget": Resource(id="budget", current=50.0)},
    )

    constraint = MockActionConstraint()
    mediator = ConstraintMediator(resource_locks=True)

    prop1 = ActorActionProposal(
        actor_id="agent_A",
        action=Action(id="forbidden_act", type="custom", actor_id="agent_A"),
        priority=10,
    )
    prop2 = ActorActionProposal(
        actor_id="agent_B",
        action=Action(
            id="spend_30",
            type="custom",
            actor_id="agent_B",
            parameters={"cost": {"budget": 30.0}},
        ),
        priority=5,
    )
    prop3 = ActorActionProposal(
        actor_id="agent_C",
        action=Action(
            id="spend_40",
            type="custom",
            actor_id="agent_C",
            parameters={"cost": {"budget": 40.0}},
        ),
        priority=2,
    )

    result: AdjudicationResult = mediator.adjudicate(
        proposals=[prop3, prop1, prop2],  # un-ordered input
        state=state,
        constraints=[constraint],
    )

    # prop1 (priority 10) evaluated first, but rejected because forbidden
    # prop2 (priority 5) evaluated second, accepted (spending 30 <= 50, remaining 20)
    # prop3 (priority 2) evaluated third, rejected because 30 + 40 > 50 (insufficient resource)
    assert len(result.accepted_actions) == 1
    assert result.accepted_actions[0].id == "spend_30"

    assert len(result.rejected_actions) == 2
    rej_ids = [a.id for a, _ in result.rejected_actions]
    assert "forbidden_act" in rej_ids
    assert "spend_40" in rej_ids

    assert len(result.adjudication_fingerprint) == 64
