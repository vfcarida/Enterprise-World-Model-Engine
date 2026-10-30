"""Double-Entry Accounting & Temporal Liquidity Structural Constraints (`EWM-004`).

Implements `DoubleEntryConstraint` in the `EWM-002` structural rules layer, enforcing:
1. Exact transaction conservation:
   $$\\sum \\Delta \\text{Credits} - \\sum \\Delta \\text{Debits} = \\text{Decimal('0.00')}$$
2. Fundamental balance-sheet identity at every step:
   $$\\sum \\text{Assets} = \\sum \\text{Liabilities} + \\sum \\text{Equity}$$
3. Temporal liquidity floor compliance:
   $$\\sum_{a \\in \\text{Liquid}} a.\\text{balance} \\ge \\text{min\\_total\\_liquidity}, \\quad a.\\text{balance} \\ge a.\\text{min\\_balance}$$
4. Strict rejection of `float` parameters with `TypeError` (zero tolerance for floating-point drift).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.domains.finance.state import (
    ZERO_DECIMAL,
    FinancialState,
    JournalEntry,
    JournalLine,
    deprecated_float_api_guard,
    ensure_corporate_decimal_context,
    to_exact_decimal,
)


def _reject_nested_floats(obj: Any, *, context_label: str) -> None:
    """Recursively inspect action parameters and raise `TypeError` if any `float` is present."""
    if isinstance(obj, float):
        deprecated_float_api_guard(obj, api_name=context_label)
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            _reject_nested_floats(v, context_label=f"{context_label}.{k}")
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        for idx, item in enumerate(obj):
            _reject_nested_floats(item, context_label=f"{context_label}[{idx}]")


def parse_journal_entry_from_action(action: Action, *, step: int = 0) -> JournalEntry:
    """Parse an enterprise `Action` into an exact `Decimal`-backed `JournalEntry`.

    Supports both:
    - `action.parameters["lines"]`: list of `{"account_id": ..., "debit": ..., "credit": ...}`
    - `action.parameters["debits"]` & `action.parameters["credits"]`: mappings `{account_id: amount}`

    Raises:
        TypeError: If any monetary amount in `action.parameters` is a `float`.
    """
    ensure_corporate_decimal_context()
    _reject_nested_floats(action.parameters, context_label=f"Action({action.id}).parameters")

    lines: list[JournalLine] = []
    raw_lines = action.parameters.get("lines")
    if isinstance(raw_lines, Sequence) and not isinstance(raw_lines, (str, bytes)):
        for idx, item in enumerate(raw_lines):
            if not isinstance(item, Mapping):
                raise TypeError(
                    f"Journal line {idx} in Action({action.id}) must be a mapping, got {type(item).__name__}"
                )
            lines.append(
                JournalLine(
                    account_id=str(item["account_id"]),
                    debit=to_exact_decimal(
                        item.get("debit", "0.00"),
                        field_name=f"Action({action.id}).lines[{idx}].debit",
                    ),
                    credit=to_exact_decimal(
                        item.get("credit", "0.00"),
                        field_name=f"Action({action.id}).lines[{idx}].credit",
                    ),
                    memo=str(item.get("memo", "")),
                )
            )

    debits_map = action.parameters.get("debits")
    if isinstance(debits_map, Mapping):
        for acc_id, amt in debits_map.items():
            lines.append(
                JournalLine(
                    account_id=str(acc_id),
                    debit=to_exact_decimal(
                        amt, field_name=f"Action({action.id}).debits[{acc_id!r}]"
                    ),
                    credit=ZERO_DECIMAL,
                )
            )

    credits_map = action.parameters.get("credits")
    if isinstance(credits_map, Mapping):
        for acc_id, amt in credits_map.items():
            lines.append(
                JournalLine(
                    account_id=str(acc_id),
                    debit=ZERO_DECIMAL,
                    credit=to_exact_decimal(
                        amt, field_name=f"Action({action.id}).credits[{acc_id!r}]"
                    ),
                )
            )

    return JournalEntry(
        entry_id=action.id,
        step=step,
        description=str(action.parameters.get("description", action.type)),
        lines=tuple(lines),
    )


class DoubleEntryConstraint:
    """Structural accounting invariant enforcing exact double-entry conservation and liquidity floors.

    Registered in the `EWM-002` `HybridTransitionRouter` (`ConstraintRegistry`, `pre_rules`,
    and `post_rules`) so that:
    1. Any action where $\\sum \\Delta \\text{Credits} - \\sum \\Delta \\text{Debits} \\ne \\text{Decimal('0.00')}$
       (even by $\\$0.01$) is 100% rejected before state mutation.
    2. Every resulting state satisfies $\\sum \\text{Assets} = \\sum \\text{Liabilities} + \\sum \\text{Equity}$
       with strictly $\\text{Decimal('0.00')}$ tolerance.
    3. Temporal liquidity constraints ($\\text{total\\_liquidity} \\ge \\text{min\\_total\\_liquidity}$ and
       account minimum balances) hold at every step.
    """

    def __init__(
        self,
        *,
        constraint_id: str = "double_entry_accounting_invariant",
        version: str = "1.0.0",
        enforce_liquidity_floor: bool = True,
    ) -> None:
        self._constraint_id = constraint_id
        self._version = version
        self.enforce_liquidity_floor = bool(enforce_liquidity_floor)

    @property
    def constraint_id(self) -> str:
        return self._constraint_id

    @property
    def version(self) -> str:
        return self._version

    @property
    def severity(self) -> ConstraintSeverity:
        return ConstraintSeverity.HARD

    @property
    def description(self) -> str:
        return (
            "Enforces exact Decimal double-entry conservation (sum(Credits) - sum(Debits) == 0.00), "
            "balance-sheet identity (sum(Assets) == sum(Liabilities) + sum(Equity)), and temporal "
            "liquidity floors with 0.00 tolerance."
        )

    def validate_financial_state(
        self,
        fin_state: FinancialState,
    ) -> tuple[bool, str, dict[str, str]]:
        """Validate `FinancialState` balance-sheet identity and liquidity floors in exact Decimal."""
        ensure_corporate_decimal_context()
        discrepancy = fin_state.accounting_discrepancy
        details = {
            "total_assets": str(fin_state.total_assets),
            "total_liabilities": str(fin_state.total_liabilities),
            "total_equity": str(fin_state.total_equity),
            "accounting_discrepancy": str(discrepancy),
            "total_liquidity": str(fin_state.total_liquidity),
            "min_total_liquidity": str(fin_state.min_total_liquidity),
        }
        if discrepancy != ZERO_DECIMAL:
            return (
                False,
                f"Accounting identity violated: sum(Assets)={fin_state.total_assets} != "
                f"sum(Liabilities)={fin_state.total_liabilities} + sum(Equity)={fin_state.total_equity} "
                f"(discrepancy={discrepancy} != 0.00)",
                details,
            )

        if self.enforce_liquidity_floor:
            if fin_state.total_liquidity < fin_state.min_total_liquidity:
                return (
                    False,
                    f"Temporal liquidity floor violated: total_liquidity={fin_state.total_liquidity} < "
                    f"min_total_liquidity={fin_state.min_total_liquidity}",
                    details,
                )
            for acc in fin_state.accounts.values():
                if acc.min_balance is not None and acc.balance < acc.min_balance:
                    return (
                        False,
                        f"Account minimum balance violated for {acc.account_id!r}: "
                        f"balance={acc.balance} < min_balance={acc.min_balance}",
                        details,
                    )

        return True, "Accounting identity and liquidity constraints satisfied (0.00 discrepancy).", details

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase,
    ) -> ConstraintResult:
        """Evaluate double-entry and balance-sheet invariants at `PRE_ACTION` or `POST_TRANSITION`."""
        ensure_corporate_decimal_context()
        fin_state = FinancialState.from_world_state(state)

        if phase == ConstraintPhase.PRE_ACTION:
            candidate = fin_state
            for act in actions:
                entry = parse_journal_entry_from_action(act, step=state.step)
                imbalance = entry.net_imbalance
                if imbalance != ZERO_DECIMAL:
                    return ConstraintResult(
                        satisfied=False,
                        constraint_id=self.constraint_id,
                        constraint_version=self.version,
                        severity=ConstraintSeverity.HARD,
                        phase=phase,
                        message=(
                            f"Double-entry imbalance in action {act.id!r}: "
                            f"sum(Credits)={entry.total_credits} - sum(Debits)={entry.total_debits} "
                            f"= {imbalance} != 0.00"
                        ),
                        violating_entities=("general_ledger",),
                        violating_values={
                            "total_debits": str(entry.total_debits),
                            "total_credits": str(entry.total_credits),
                            "net_imbalance": str(imbalance),
                        },
                        step=state.step,
                        preceding_action_id=act.id,
                    )
                try:
                    candidate = candidate.apply_journal_entry(entry)
                except KeyError as exc:
                    return ConstraintResult(
                        satisfied=False,
                        constraint_id=self.constraint_id,
                        constraint_version=self.version,
                        severity=ConstraintSeverity.HARD,
                        phase=phase,
                        message=str(exc),
                        violating_entities=("general_ledger",),
                        step=state.step,
                        preceding_action_id=act.id,
                    )

                ok, msg, details = self.validate_financial_state(candidate)
                if not ok:
                    return ConstraintResult(
                        satisfied=False,
                        constraint_id=self.constraint_id,
                        constraint_version=self.version,
                        severity=ConstraintSeverity.HARD,
                        phase=phase,
                        message=f"Action {act.id!r} rejected: {msg}",
                        violating_entities=("general_ledger",),
                        violating_values=details,
                        step=state.step,
                        preceding_action_id=act.id,
                    )

            return ConstraintResult(
                satisfied=True,
                constraint_id=self.constraint_id,
                constraint_version=self.version,
                severity=ConstraintSeverity.HARD,
                phase=phase,
                message="All proposed journal actions satisfy double-entry and liquidity invariants.",
                step=state.step,
            )

        # POST_TRANSITION: Verify resulting state ledger balance and resource consistency
        ok, msg, details = self.validate_financial_state(fin_state)
        if not ok:
            return ConstraintResult(
                satisfied=False,
                constraint_id=self.constraint_id,
                constraint_version=self.version,
                severity=ConstraintSeverity.HARD,
                phase=phase,
                message=msg,
                violating_entities=("general_ledger",),
                violating_values=details,
                step=state.step,
            )

        return ConstraintResult(
            satisfied=True,
            constraint_id=self.constraint_id,
            constraint_version=self.version,
            severity=ConstraintSeverity.HARD,
            phase=phase,
            message=msg,
            step=state.step,
        )

    def as_pre_rule(self, state: WorldState, actions: Sequence[Action]) -> tuple[bool, str]:
        """Callable adapter for `HybridTransitionRouter(pre_rules=[...])`."""
        res = self.evaluate(state, actions, phase=ConstraintPhase.PRE_ACTION)
        return res.satisfied, res.message

    def as_post_rule(
        self,
        prev_state: WorldState,
        candidate_state: WorldState,
        actions: Sequence[Action],
    ) -> tuple[bool, str]:
        """Callable adapter for `HybridTransitionRouter(post_rules=[...])`."""
        res = self.evaluate(candidate_state, actions, phase=ConstraintPhase.POST_TRANSITION)
        return res.satisfied, res.message
