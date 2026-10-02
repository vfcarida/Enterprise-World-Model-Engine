"""Constraint registry coordinating action pre-validation and state post-validation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from ewm_engine.constraints.base import Constraint
from ewm_engine.constraints.results import ConstraintResult, ConstraintSeverity
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import ConstraintId


class ConstraintRegistry:
    """Registry maintaining active constraints for an enterprise world model.

    Evaluates both pre-transition action compatibility and post-transition state validity.
    """

    def __init__(self, constraints: Sequence[Constraint] | None = None) -> None:
        self._constraints: dict[ConstraintId, Constraint] = {}
        if constraints:
            for c in constraints:
                self.register(c)

    def register(self, constraint: Constraint) -> None:
        """Register a new constraint into the world registry."""
        self._constraints[constraint.constraint_id] = constraint

    def get(self, constraint_id: ConstraintId) -> Constraint | None:
        """Fetch a registered constraint by ID."""
        return self._constraints.get(constraint_id)

    def __len__(self) -> int:
        return len(self._constraints)

    def __iter__(self) -> Iterator[Constraint]:
        return iter(self._constraints.values())

    def validate_actions(
        self,
        state: WorldState,
        actions: Sequence[Action],
    ) -> tuple[list[Action], list[ConstraintResult]]:
        """Pre-transition validation of proposed actions.

        Returns:
            A tuple of (accepted_actions, constraint_results).
            If an action violates a HARD constraint, it is excluded from accepted_actions.
        """
        accepted_actions: list[Action] = []
        all_results: list[ConstraintResult] = []

        for action in actions:
            action_valid = True
            for constraint in self._constraints.values():
                res = constraint.evaluate(state=state, action=action)
                if not res.satisfied:
                    all_results.append(
                        res.model_copy(
                            update={
                                "step": state.step,
                                "preceding_action_id": action.id,
                            }
                        )
                    )
                    if res.severity == ConstraintSeverity.HARD:
                        action_valid = False
            if action_valid:
                accepted_actions.append(action)

        return accepted_actions, all_results

    def validate_state(
        self,
        state: WorldState,
        preceding_actions: Sequence[Action] | None = None,
    ) -> list[ConstraintResult]:
        """Post-transition validation of the evolved world state."""
        results: list[ConstraintResult] = []
        actions = list(preceding_actions or [])
        all_action_ids = [a.id for a in actions]

        for constraint in self._constraints.values():
            res = constraint.evaluate(state=state, action=None)
            if not res.satisfied:
                matched_action_id: str | None = None

                # 1. Match action touching violating resources
                if res.violating_resources and actions:
                    for a in reversed(actions):
                        params_str = str(list(a.parameters.values()))
                        if any(r_id in params_str for r_id in res.violating_resources):
                            matched_action_id = a.id
                            break

                # 2. Match action touching violating entities
                if matched_action_id is None and res.violating_entities and actions:
                    for a in reversed(actions):
                        params_str = str(list(a.parameters.values()))
                        if any(e_id in params_str for e_id in res.violating_entities):
                            matched_action_id = a.id
                            break

                # 3. Fallback to first action if actions exist
                if matched_action_id is None and actions:
                    matched_action_id = actions[0].id

                metadata = dict(res.metadata)
                if all_action_ids:
                    metadata["preceding_action_ids"] = all_action_ids

                results.append(
                    res.model_copy(
                        update={
                            "step": state.step,
                            "preceding_action_id": matched_action_id,
                            "metadata": metadata,
                        }
                    )
                )
        return results

    @staticmethod
    def has_hard_violations(results: Sequence[ConstraintResult]) -> bool:
        """Check if any evaluated constraint result contains a HARD violation."""
        return any(not r.satisfied and r.severity == ConstraintSeverity.HARD for r in results)

    @staticmethod
    def total_penalty(results: Sequence[ConstraintResult]) -> float:
        """Compute the sum of penalties from soft constraint violations."""
        return sum(r.penalty for r in results if not r.satisfied)

    def clone(self) -> ConstraintRegistry:
        """Create an independent copy of the registry."""
        return ConstraintRegistry(list(self._constraints.values()))
