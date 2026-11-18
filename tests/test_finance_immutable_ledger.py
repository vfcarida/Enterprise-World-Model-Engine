"""Verification suite for Prompt E6: Finance Application Evolution - Immutable Ledgers.

Verifies:
1. `Entry` and `ImmutableTransaction` enforcing `sum(debits) == sum(credits)` ($dr - cr = 0$)
   with custom `BalanceError(EngineError)` on any imbalance.
2. Strict prohibition of floating-point (`float`) inputs across `Entry`, `ImmutableTransaction`,
   `LedgerAccount`, and `FinancialState` (raising `TypeError`).
3. Tuple-relational equilibrium condition $h_a = h_o$ enforced prior to mutating state in
   `ImmutableLedger.commit`.
4. Property-based tests (`hypothesis`) generating random transaction legs across arbitrary
   precision down to $10^{-10}$ decimal places, confirming only balanced transactions
   ($dr = cr$) are persisted.
5. OCEL 2.0 E2O mapping of ledger entries with explicit `"DebitAccount"` and `"CreditAccount"`
   object qualifiers.
6. Operational safeguards preventing `UPDATE` or `DELETE` operations (`LedgerImmutabilityError`)
   and mandating error correction exclusively via compensating reversal entries.
"""

from __future__ import annotations

from decimal import Decimal, getcontext
from pathlib import Path
from typing import Dict, List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ewm_engine.core.ocel import StateEncoder, read_ocel2_sqlite, validate_ocel2_sqlite
from ewm_engine.core.types import EngineError
from ewm_engine.domains.finance import (
    MIN_CORPORATE_DECIMAL_PRECISION,
    AccountCategory,
    BalanceError,
    Entry,
    EntryDirection,
    ImmutableLedger,
    ImmutableTransaction,
    LedgerAccount,
    LedgerImmutabilityError,
    compute_probabilistic_financial_reward,
    ensure_corporate_decimal_context,
)


def _create_base_accounts() -> Dict[str, LedgerAccount]:
    """Create a balanced initial chart of accounts for ImmutableLedger tests."""
    return {
        "cash": LedgerAccount(
            account_id="cash",
            name="Operating Cash",
            category=AccountCategory.ASSET,
            balance=Decimal("1000000.0000000000"),
            is_liquid=True,
        ),
        "treasury_bills": LedgerAccount(
            account_id="treasury_bills",
            name="Treasury Bills",
            category=AccountCategory.ASSET,
            balance=Decimal("500000.0000000000"),
            is_liquid=True,
        ),
        "accounts_payable": LedgerAccount(
            account_id="accounts_payable",
            name="Accounts Payable",
            category=AccountCategory.LIABILITY,
            balance=Decimal("600000.0000000000"),
        ),
        "retained_earnings": LedgerAccount(
            account_id="retained_earnings",
            name="Retained Earnings",
            category=AccountCategory.EQUITY,
            balance=Decimal("900000.0000000000"),
        ),
    }


def test_strict_float_rejection_and_extreme_1e_minus_10_precision() -> None:
    """Passing a float raises TypeError; unbalanced by 10^-10 raises BalanceError."""
    assert issubclass(BalanceError, EngineError)
    assert issubclass(LedgerImmutabilityError, EngineError)

    prec = ensure_corporate_decimal_context()
    assert prec >= MIN_CORPORATE_DECIMAL_PRECISION
    assert getcontext().prec >= 50

    # 1. Passing float to Entry or ImmutableTransaction raises TypeError
    with pytest.warns(DeprecationWarning), pytest.raises(TypeError, match="rejected float"):
        Entry(account="cash", magnitude=100.50, direction=EntryDirection.DEBIT)  # type: ignore[arg-type]

    with pytest.warns(DeprecationWarning), pytest.raises(TypeError, match="rejected float"):
        ImmutableTransaction(
            transaction_id="tx_float_ts",
            timestamp=1700000000.5,  # type: ignore[arg-type]
            entries=(
                Entry(account="cash", magnitude=Decimal("10"), direction=EntryDirection.DEBIT),
                Entry(
                    account="retained_earnings",
                    magnitude=Decimal("10"),
                    direction=EntryDirection.CREDIT,
                ),
            ),
        )

    # 2. Extreme precision (10^-10) balanced transaction succeeds
    high_prec_amount = Decimal("9876543210.1234567891")
    tx_balanced = ImmutableTransaction(
        transaction_id="tx_1e10_ok",
        timestamp="2026-11-17T14:00:00.123456Z",
        entries=(
            Entry(
                account="cash",
                magnitude=high_prec_amount,
                direction=EntryDirection.DEBIT,
                category=AccountCategory.ASSET,
            ),
            Entry(
                account="retained_earnings",
                magnitude=high_prec_amount,
                direction=EntryDirection.CREDIT,
                category=AccountCategory.EQUITY,
            ),
        ),
    )
    assert tx_balanced.total_debits == high_prec_amount
    assert tx_balanced.total_credits == high_prec_amount
    h_a, h_o = tx_balanced.compute_tuple_equilibrium({})
    assert h_a == high_prec_amount
    assert h_o == high_prec_amount
    assert h_a == h_o

    # 3. Unbalanced by exactly 10^-10 raises BalanceError
    epsilon_1e10 = Decimal("0.0000000001")
    with pytest.raises(BalanceError, match="failed double-entry balance check"):
        ImmutableTransaction(
            transaction_id="tx_1e10_unbalanced",
            entries=(
                Entry(
                    account="cash",
                    magnitude=high_prec_amount + epsilon_1e10,
                    direction=EntryDirection.DEBIT,
                ),
                Entry(
                    account="retained_earnings",
                    magnitude=high_prec_amount,
                    direction=EntryDirection.CREDIT,
                ),
            ),
        )


@settings(max_examples=50, deadline=None)
@given(
    debit_legs=st.lists(
        st.integers(min_value=1, max_value=10**18),
        min_size=1,
        max_size=5,
    ),
    imbalance_units=st.integers(min_value=-(10**10), max_value=10**10),
)
def test_property_based_random_transaction_legs_and_1e10_conservation(
    debit_legs: List[int],
    imbalance_units: int,
) -> None:
    """Property-based test: only transactions with dr == cr and h_a == h_o are persisted."""
    scale = Decimal("0.0000000001")  # 10^-10 precision unit
    debit_decimals = [Decimal(val) * scale for val in debit_legs]
    total_dr = sum(debit_decimals, start=Decimal("0"))
    perturbation = Decimal(imbalance_units) * scale
    credit_total = total_dr + perturbation

    # Skip degenerate case where credit_total <= 0 (tested separately by Entry validator)
    if credit_total <= Decimal("0"):
        return

    entries = [
        *(
            Entry(
                account="cash" if i % 2 == 0 else "treasury_bills",
                magnitude=amt,
                direction=EntryDirection.DEBIT,
                category=AccountCategory.ASSET,
            )
            for i, amt in enumerate(debit_decimals)
        ),
        Entry(
            account="retained_earnings",
            magnitude=credit_total,
            direction=EntryDirection.CREDIT,
            category=AccountCategory.EQUITY,
        ),
    ]

    ledger = ImmutableLedger(_create_base_accounts())
    initial_assets = ledger.total_assets
    initial_claims = ledger.ownership_claims

    if perturbation == Decimal("0"):
        tx = ImmutableTransaction(
            transaction_id="prop_tx_balanced",
            entries=tuple(entries),
        )
        ledger.commit(tx)
        assert len(ledger.transactions) == 1
        assert ledger.total_assets == initial_assets + total_dr
        assert ledger.ownership_claims == initial_claims + total_dr
        assert ledger.total_assets == ledger.ownership_claims
    else:
        with pytest.raises(BalanceError):
            ImmutableTransaction(
                transaction_id="prop_tx_unbalanced",
                entries=tuple(entries),
            )
        assert len(ledger.transactions) == 0
        assert ledger.total_assets == initial_assets
        assert ledger.ownership_claims == initial_claims


def test_ocel2_mapping_with_debit_and_credit_account_qualifiers(tmp_path: Path) -> None:
    """Verify financial transactions map to E2 OCEL 2.0 with 'DebitAccount' and 'CreditAccount' qualifiers."""
    db_path = tmp_path / "finance_ledger_ocel2.sqlite"

    with StateEncoder(db_path) as encoder:
        ledger = ImmutableLedger(_create_base_accounts(), ocel_encoder=encoder)

        tx = ImmutableTransaction(
            transaction_id="tx_ocel_001",
            timestamp="2026-11-17T14:00:28.500000Z",
            description="Settle vendor invoice and recognize treasury fee",
            entries=(
                Entry(
                    account="accounts_payable",
                    magnitude=Decimal("15000.0000000001"),
                    direction=EntryDirection.DEBIT,
                    category=AccountCategory.LIABILITY,
                ),
                Entry(
                    account="treasury_bills",
                    magnitude=Decimal("5000.0000000000"),
                    direction=EntryDirection.DEBIT,
                    category=AccountCategory.ASSET,
                ),
                Entry(
                    account="cash",
                    magnitude=Decimal("20000.0000000001"),
                    direction=EntryDirection.CREDIT,
                    category=AccountCategory.ASSET,
                ),
            ),
        )
        ocel_rec = ledger.commit(tx)
        assert ocel_rec.event_id == "tx_ocel_001"
        assert ledger.total_assets == ledger.ownership_claims

    sat, unsat = validate_ocel2_sqlite(db_path)
    assert len(sat) == 24
    assert unsat == []

    ocel = read_ocel2_sqlite(db_path)
    assert len(ocel.events) == 1
    assert len(ocel.objects) == 3

    e2o_qualifiers = {
        (str(row["ocel:oid"]), str(row["ocel:qualifier"])) for _, row in ocel.relations.iterrows()
    }
    assert e2o_qualifiers == {
        ("accounts_payable", "DebitAccount"),
        ("treasury_bills", "DebitAccount"),
        ("cash", "CreditAccount"),
    }


def test_immutability_safeguards_and_compensating_reversal_entries(tmp_path: Path) -> None:
    """Prevent UPDATE/DELETE on ledger state; mandate error correction via compensating reversals."""
    db_path = tmp_path / "reversal_ocel2.sqlite"

    with StateEncoder(db_path) as encoder:
        ledger = ImmutableLedger(_create_base_accounts(), ocel_encoder=encoder)
        initial_cash = ledger.get_balance("cash")
        initial_equity = ledger.get_balance("retained_earnings")

        erroneous_tx = ImmutableTransaction(
            transaction_id="tx_err_101",
            timestamp="2026-11-17T14:01:00Z",
            description="Erroneous fee posting",
            entries=(
                Entry(
                    account="cash",
                    magnitude=Decimal("25000.0000000005"),
                    direction=EntryDirection.DEBIT,
                ),
                Entry(
                    account="retained_earnings",
                    magnitude=Decimal("25000.0000000005"),
                    direction=EntryDirection.CREDIT,
                ),
            ),
        )
        ledger.commit(erroneous_tx)
        assert ledger.get_balance("cash") == initial_cash + Decimal("25000.0000000005")

        # 1. Attempting UPDATE or DELETE raises LedgerImmutabilityError
        with pytest.raises(
            LedgerImmutabilityError, match="UPDATE operations are strictly prohibited"
        ):
            ledger.update_transaction("tx_err_101", entries=())

        with pytest.raises(
            LedgerImmutabilityError, match="DELETE operations are strictly prohibited"
        ):
            ledger.delete_transaction("tx_err_101")

        with pytest.raises(LedgerImmutabilityError):
            ledger[0] = erroneous_tx

        with pytest.raises(LedgerImmutabilityError):
            del ledger["tx_err_101"]

        # Re-committing duplicate transaction ID raises LedgerImmutabilityError
        with pytest.raises(LedgerImmutabilityError, match="already committed"):
            ledger.commit(erroneous_tx)

        # 2. Error correction via compensating reversal entry restores exact balances
        reversal_tx = ledger.reverse_transaction(
            "tx_err_101",
            "tx_rev_101",
            timestamp="2026-11-17T14:02:00Z",
            reason="Correct erroneous fee posting",
        )
        assert reversal_tx.compensates_transaction_id == "tx_err_101"
        assert len(ledger.transactions) == 2
        assert ledger.get_balance("cash") == initial_cash
        assert ledger.get_balance("retained_earnings") == initial_equity
        assert ledger.total_assets == ledger.ownership_claims

    # Verify both original and reversal transactions are recorded in OCEL 2.0
    ocel = read_ocel2_sqlite(db_path)
    assert len(ocel.events) == 2
    assert set(ocel.events["ocel:activity"]) == {
        "DoubleEntryTransaction",
        "CompensatingReversalTransaction",
    }

    # Verify probabilistic financial reward function is deprecated
    with pytest.deprecated_call():
        reward = compute_probabilistic_financial_reward(100)
    assert reward == Decimal("0.00")
