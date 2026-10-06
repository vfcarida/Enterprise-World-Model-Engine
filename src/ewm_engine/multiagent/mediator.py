"""Mediator / Game Master protocols and deterministic constraint-based implementation.

Adjudicates competing multi-actor proposals deterministically without requiring an LLM.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ewm_engine.constraints.base import Constraint
from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.multiagent.views import (
    ActorActionProposal,
    AdjudicationResult,
)


@runtime_checkable
class Mediator(Protocol):
    """Protocol for an adjudicator resolving competing multi-actor proposals."""

    def adjudicate(
        self,
        proposals: Sequence[ActorActionProposal],
        state: WorldState,
        constraints: Sequence[Constraint] | None = None,
    ) -> AdjudicationResult:
        """Resolve proposed actions into accepted and rejected actions."""
        ...


class ConstraintMediator(Mediator):
    """Deterministic default mediator reusing the hard/soft constraint engine.

    Resolves multi-actor conflicts deterministically:
    1. Sorts proposals by priority (descending), actor_id (lexicographic), action.id (lexicographic).
    2. Validates each candidate action against action-level hard constraints.
    3. Rejects actions that violate hard constraints with an explicit violation reason.
    4. Evaluates soft constraints and tallies penalties.
    """

    def __init__(self, resource_locks: bool = True) -> None:
        self.resource_locks = resource_locks

    def adjudicate(
        self,
        proposals: Sequence[ActorActionProposal],
        state: WorldState,
        constraints: Sequence[Constraint] | None = None,
    ) -> AdjudicationResult:
        all_constraints = list(constraints or [])

        # Deterministic sorting: priority DESC, actor_id ASC, action.id ASC
        sorted_proposals = sorted(
            proposals,
            key=lambda p: (-p.priority, p.actor_id, p.action.id),
        )

        accepted: list[Action] = []
        rejected: list[tuple[Action, str]] = []
        penalties: dict[str, float] = {}

        # Track allocated resources during this adjudication step if resource locks enabled
        allocated_resources: dict[str, float] = {}

        for prop in sorted_proposals:
            act = prop.action
            is_valid = True
            rejection_reason = ""

            # 1. Action constraint evaluation
            for c in all_constraints:
                verdict: ConstraintResult = c.evaluate(
                    state, (act,), phase=ConstraintPhase.PRE_ACTION
                )
                if not verdict.satisfied:
                    if verdict.severity == ConstraintSeverity.HARD:
                        is_valid = False
                        rejection_reason = (
                            verdict.message or f"Violates hard constraint '{c.constraint_id}'"
                        )
                        break
                    else:
                        penalties[f"{act.id}_{c.constraint_id}"] = verdict.penalty or 0.0

            # 2. Scarcity / resource lock check
            cost_dict = act.parameters.get("cost", {}) if isinstance(act.parameters, dict) else {}
            if is_valid and self.resource_locks and cost_dict:
                for res_id, req_amount in cost_dict.items():
                    available = (
                        state.resources[res_id].current if res_id in state.resources else 0.0
                    )
                    already_allocated = allocated_resources.get(res_id, 0.0)
                    if already_allocated + req_amount > available:
                        is_valid = False
                        rejection_reason = (
                            f"Insufficient resource '{res_id}': required {req_amount}, "
                            f"remaining {available - already_allocated}"
                        )
                        break

            if is_valid:
                accepted.append(act)
                if self.resource_locks and cost_dict:
                    for res_id, req_amount in cost_dict.items():
                        allocated_resources[res_id] = (
                            allocated_resources.get(res_id, 0.0) + req_amount
                        )
            else:
                rejected.append((act, rejection_reason))

        return AdjudicationResult(
            accepted_actions=tuple(accepted),
            rejected_actions=tuple(rejected),
            penalties=penalties,
            metadata={
                "mediator_type": "ConstraintMediator",
                "proposals_count": len(proposals),
                "accepted_count": len(accepted),
                "rejected_count": len(rejected),
            },
        )
