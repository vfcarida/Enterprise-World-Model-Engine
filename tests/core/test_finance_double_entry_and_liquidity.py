"""Tests for EWM-004: Finance Domain Pack Evolution (Exact Decimal, Double-Entry & Liquidity Stress).

Verifies:
1. `FinancialState`, `LedgerAccount`, `JournalLine`, and `build_journal_action` strictly use
   `decimal.Decimal` and raise `TypeError` (with `DeprecationWarning`) if any `float` is passed.
2. Corporate `Decimal` precision safeguard (`getcontext().prec >= 38`) prevents silent truncation
   of multi-trillion-dollar corporate values down to `$0.01`.
3. Submitting an action that unbalances the ledger by `$0.01` (`Decimal('0.01')`) is 100% rejected
   by the `EWM-002` `HybridTransitionRouter` / `TransitionEngine`.
4. Running a 1,000-step simulation preserves `sum(Assets) == sum(Liabilities) + sum(Equity)` with
   strictly `Decimal('0.00')` discrepancy, and benchmarks `Decimal` vs `float` rollout performance.
5. The corporate liquidity drain stress-test scenario executes using `engine.branch().do()`.
"""

from __future__ import annotations

from decimal import Decimal, getcontext

import pytest

from ewm_engine.constraints.results import ConstraintPhase
from ewm_engine.core.actions import Action
from ewm_engine.domains.finance import (
    MIN_CORPORATE_DECIMAL_PRECISION,
    ZERO_DECIMAL,
    AccountCategory,
    DoubleEntryConstraint,
    FinancialState,
    JournalEntry,
    JournalLine,
    LedgerAccount,
    benchmark_decimal_vs_float_rollout,
    build_corporate_treasury_engine,
    build_journal_action,
    create_default_treasury_state,
    ensure_corporate_decimal_context,
    parse_journal_entry_from_action,
    run_liquidity_drain_stress_test,
    to_exact_decimal,
)


def test_decimal_precision_safeguard_and_strict_float_rejection() -> None:
    """Verify corporate Decimal precision >= 38 and strict TypeError when floats are passed."""
    # Even if external code lowers Decimal precision, our safeguard restores prec >= 38
    getcontext().prec = 6
    restored_prec = ensure_corporate_decimal_context()
    assert restored_prec >= MIN_CORPORATE_DECIMAL_PRECISION
    assert getcontext().prec >= MIN_CORPORATE_DECIMAL_PRECISION

    # Multi-trillion corporate value + $0.01 must never truncate silently
    large_corp = to_exact_decimal("999999999999999999.99")
    cent = to_exact_decimal("0.01")
    assert large_corp + cent == Decimal("1000000000000000000.00")
    assert to_exact_decimal(42) == Decimal("42")

    # Passing float anywhere raises TypeError (and DeprecationWarning where guarded)
    with pytest.raises(TypeError, match="Float values are prohibited"):
        to_exact_decimal(100.50)

    with pytest.raises(TypeError, match="Boolean values"):
        to_exact_decimal(True)

    with pytest.raises(TypeError, match="Unsupported monetary type"):
        to_exact_decimal(object())

    with pytest.raises(ValueError, match="Non-finite"):
        to_exact_decimal(Decimal("NaN"))

    with pytest.raises(ValueError, match="Non-finite"):
        to_exact_decimal("Infinity")

    with pytest.raises(ValueError, match="Invalid Decimal string"):
        to_exact_decimal("not_a_number")

    with pytest.warns(DeprecationWarning), pytest.raises(TypeError):
        LedgerAccount(
            account_id="cash",
            name="Cash",
            category=AccountCategory.ASSET,
            balance=100.0,  # type: ignore[arg-type]
        )

    with pytest.warns(DeprecationWarning), pytest.raises(TypeError):
        JournalLine(account_id="cash", debit=0.01)  # type: ignore[arg-type]

    with pytest.warns(DeprecationWarning), pytest.raises(TypeError):
        build_journal_action(
            "bad_float_tx",
            debits={"cash": 10.0},  # type: ignore[dict-item]
            credits={"retained_earnings": Decimal("10.00")},
        )

    with pytest.warns(DeprecationWarning), pytest.raises(TypeError):
        FinancialState(min_total_liquidity=500.0)  # type: ignore[arg-type]


def test_engine_rejects_one_cent_ledger_imbalance_and_liquidity_breach() -> None:
    """Submit an action that unbalances the ledger by $0.01; assert engine rejects it."""
    engine, initial_fin = build_corporate_treasury_engine()
    assert initial_fin.is_balanced is True
    assert initial_fin.accounting_discrepancy == ZERO_DECIMAL

    # 1. Action unbalanced by +$0.01 (Credits exceed Debits by $0.01)
    unbalanced_plus_cent = build_journal_action(
        "unbalanced_by_plus_0_01",
        debits={"cash": Decimal("1000.00")},
        credits={"retained_earnings": Decimal("1000.01")},
        description="Off-by-one-cent credit imbalance ($0.01)",
    )
    outcome_plus = engine.step(action=unbalanced_plus_cent)
    assert outcome_plus.pre_rule_rejected is True
    assert outcome_plus.rolled_back is True
    assert "unbalanced_by_plus_0_01" in outcome_plus.rejected_actions
    assert any("0.01 != 0.00" in v for v in outcome_plus.pre_rule_violations)

    fin_after_plus = FinancialState.from_world_state(outcome_plus.next_state)
    assert fin_after_plus.is_balanced is True
    assert fin_after_plus.accounting_discrepancy == ZERO_DECIMAL
    assert fin_after_plus.get_balance("cash") == Decimal("10000000.00")

    # 2. Action unbalanced by -$0.01 (Debits exceed Credits by $0.01)
    unbalanced_minus_cent = build_journal_action(
        "unbalanced_by_minus_0_01",
        debits={"cash": Decimal("1000.01")},
        credits={"retained_earnings": Decimal("1000.00")},
        description="Off-by-one-cent debit imbalance (-$0.01)",
    )
    outcome_minus = engine.step(action=unbalanced_minus_cent)
    assert outcome_minus.pre_rule_rejected is True
    assert outcome_minus.rolled_back is True
    assert "unbalanced_by_minus_0_01" in outcome_minus.rejected_actions

    fin_after_minus = FinancialState.from_world_state(outcome_minus.next_state)
    assert fin_after_minus.is_balanced is True
    assert fin_after_minus.accounting_discrepancy == ZERO_DECIMAL
    assert fin_after_minus.get_balance("cash") == Decimal("10000000.00")

    # 3. Balanced double-entry action that breaches temporal liquidity floor (draining $9.8M cash)
    liquidity_breach_action = build_journal_action(
        "excessive_cash_withdrawal",
        debits={"demand_deposits": Decimal("9800000.00")},
        credits={"cash": Decimal("9800000.00")},
        description="Attempt to drain cash to $200k (< $500k min_cash_balance)",
    )
    outcome_liq = engine.step(action=liquidity_breach_action)
    assert outcome_liq.pre_rule_rejected is True
    assert outcome_liq.rolled_back is True
    assert any("min_balance" in v for v in outcome_liq.pre_rule_violations)

    # 4. Valid balanced double-entry action succeeds and preserves 0.00 discrepancy
    valid_action = build_journal_action(
        "valid_loan_repayment",
        debits={"cash": Decimal("250000.75")},
        credits={"loans_receivable": Decimal("250000.75")},
        description="Borrower repays principal into cash",
    )
    outcome_valid = engine.step(action=valid_action)
    assert outcome_valid.pre_rule_rejected is False
    assert outcome_valid.post_rule_rejected is False
    fin_valid = FinancialState.from_world_state(outcome_valid.next_state)
    assert fin_valid.is_balanced is True
    assert fin_valid.accounting_discrepancy == ZERO_DECIMAL
    assert fin_valid.get_balance("cash") == Decimal("10250000.75")
    assert fin_valid.get_balance("loans_receivable") == Decimal("24749999.25")


def test_one_thousand_step_simulation_and_decimal_vs_float_benchmark() -> None:
    """Run a 1,000-step simulation; assert accounting invariant holds with 0.00 drift."""
    report = benchmark_decimal_vs_float_rollout(steps=1000)
    assert report.steps == 1000
    assert report.decimal_final_discrepancy == ZERO_DECIMAL
    assert report.decimal_zero_drift is True
    assert report.decimal_engine_total_ms > 0.0
    assert report.decimal_pure_ledger_ms > 0.0
    assert report.float_pure_ledger_ms > 0.0
    # IEEE-754 float accumulation on $10T corporate balance exhibits non-zero drift
    assert Decimal(report.float_final_discrepancy_str) > ZERO_DECIMAL


def test_liquidity_drain_stress_test_with_branch_do_causal_api() -> None:
    """Execute the corporate liquidity drain stress test via `engine.branch().do()`."""
    report = run_liquidity_drain_stress_test(steps=5)
    assert report.steps == 5
    assert report.all_steps_balanced is True
    assert report.max_abs_discrepancy_across_all_branches == ZERO_DECIMAL

    # Inbound edge ('market_stress', 'deposit_outflow_amount') was severed by do()
    assert ("market_stress", "deposit_outflow_amount") in report.severed_edges_on_stress_branch

    # Baseline cash grew by 5 * $50,000 = $250,000 -> $10,250,000.00
    assert report.baseline_final_cash == Decimal("10250000.00")
    assert report.baseline_final_liquidity == Decimal("15250000.00")

    # Stress branch experienced $2.5M/step outflows ($12.5M total), triggering T-bill liquidation
    assert report.stress_final_tbills < Decimal("5000000.00")
    assert report.stress_final_cash >= Decimal("500000.00")
    assert report.stress_final_liquidity < report.baseline_final_liquidity

    # Recovery branch injected $1.5M/step repo liquidity via do(), maintaining higher cash & liquidity
    assert report.recovery_final_cash > report.stress_final_cash
    assert report.recovery_final_liquidity > report.stress_final_liquidity


def test_constraint_and_state_edge_cases_coverage() -> None:
    """Cover edge cases in FinancialState, DoubleEntryConstraint, and action parsing."""
    with pytest.raises(ValueError, match="violates accounting identity"):
        create_default_treasury_state(cash=Decimal("9999999.99"))

    state = create_default_treasury_state()
    with pytest.raises(KeyError):
        state.get_balance("nonexistent_account")

    bad_entry = JournalEntry(
        entry_id="bad_acc",
        lines=(JournalLine(account_id="missing_acc", debit=Decimal("10.00")),),
    )
    assert bad_entry.is_balanced is False
    assert bad_entry.to_action().type == "double_entry_journal"
    with pytest.raises(KeyError):
        state.apply_journal_entry(bad_entry)

    # Action with debits/credits dict format
    act_dc = Action(
        id="dc_act",
        type="double_entry_journal",
        parameters={
            "debits": {"cash": "100.00"},
            "credits": {"retained_earnings": "100.00"},
        },
    )
    parsed = parse_journal_entry_from_action(act_dc)
    assert parsed.is_balanced is True

    # Unknown account rejected by DoubleEntryConstraint
    constraint = DoubleEntryConstraint()
    assert constraint.version == "1.0.0"
    assert "double-entry" in constraint.description
    ws = state.to_world_state()
    bad_acc_act = build_journal_action(
        "unknown_acc_tx",
        debits={"unknown_acct": Decimal("50.00")},
        credits={"cash": Decimal("50.00")},
    )
    res_unknown = constraint.evaluate(ws, [bad_acc_act], phase=ConstraintPhase.PRE_ACTION)
    assert res_unknown.satisfied is False

    # Post-transition evaluation on corrupted balance sheet
    corrupted_accounts = dict(state.accounts)
    corrupted_accounts["cash"] = corrupted_accounts["cash"].model_copy(
        update={"balance": Decimal("9999999.99")}
    )
    corrupted_state = state.model_copy(update={"accounts": corrupted_accounts})
    assert corrupted_state.is_balanced is False
    res_post = constraint.evaluate(
        corrupted_state.to_world_state(),
        phase=ConstraintPhase.POST_TRANSITION,
    )
    assert res_post.satisfied is False
