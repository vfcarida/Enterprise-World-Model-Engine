# Step-by-Step Tutorial: Processing an Immutable Double-Entry Transaction (E6 Finance Domain Pack)

**Domain Package**: `ewm_engine.domains.finance`  
**Verification Suite**: [`tests/test_finance_immutable_ledger.py`](../../tests/test_finance_immutable_ledger.py)  
**Historical Legacy Tutorial**: [`docs/legacy/finance_tutorial_legacy.md`](../legacy/finance_tutorial_legacy.md)

---

## 1. Architectural Separation: Core Engine vs. Finance Domain Pack

- **Core Engine (`ewm_engine.core`)**: Provides the domain-neutral E2 OCEL 2.0 relational event store (`StateEncoder`), E3 Continual Discovery Agent (`RetrieveAndReasonEngine`), and E4 DoWhy-GCM causal engine (`EngineCausalInterface`).
- **Finance Domain Pack (`ewm_engine.domains.finance`)**: Enforces arbitrary-precision `decimal.Decimal` arithmetic (`prec >= 50`), strict `float` prohibition (`TypeError`), double-entry balance conservation ($dr - cr = 0$), tuple-relational equilibrium ($h_a = h_o$), append-only immutability (`ImmutableLedger`), and OCEL 2.0 E2O mapping with `"DebitAccount"` and `"CreditAccount"` qualifiers.

---

## 2. Mathematical Invariants

1. **Double-Entry Balance Invariant ($dr - cr = 0$)**:
   $$\sum_{e \in \text{Debits}} dr_e - \sum_{e \in \text{Credits}} cr_e = \text{Decimal('0')}$$
   Any discrepancy (even $10^{-10}$) raises `BalanceError`.
2. **Tuple-Relational Equilibrium Condition ($h_a = h_o$)**:
   Let $h_a = \sum_{e \in \text{Assets}} (dr_e - cr_e)$ be net changes to asset classes and $h_o = \sum_{e \in \text{Liabilities} \cup \text{Equity}} (cr_e - dr_e)$ be net changes to ownership claims. Before mutating state, `ImmutableLedger.commit()` verifies:
   $$h_a = h_o$$
3. **Append-Only Immutability & Compensating Reversals**:
   `UPDATE` and `DELETE` operations are blocked with `LedgerImmutabilityError`. Error correction occurs exclusively by committing a compensating reversal transaction via `ledger.reverse_transaction(...)`.

---

## 3. Step-by-Step Executable Walkthrough

Copy and paste the following self-contained Python block directly into a Python REPL:

```python
import tempfile
from decimal import Decimal
from pathlib import Path
from ewm_engine.core.ocel import StateEncoder, read_ocel2_sqlite
from ewm_engine.domains.finance import (
    AccountCategory,
    BalanceError,
    Entry,
    EntryDirection,
    ImmutableLedger,
    ImmutableTransaction,
    LedgerAccount,
    LedgerImmutabilityError,
)

# Step 1: Initialize a balanced chart of accounts using exact Decimal values
accounts = {
    "cash": LedgerAccount(
        account_id="cash",
        name="Operating Cash",
        category=AccountCategory.ASSET,
        balance=Decimal("1000000.0000000000"),
        is_liquid=True,
    ),
    "accounts_payable": LedgerAccount(
        account_id="accounts_payable",
        name="Vendor Accounts Payable",
        category=AccountCategory.LIABILITY,
        balance=Decimal("400000.0000000000"),
    ),
    "retained_earnings": LedgerAccount(
        account_id="retained_earnings",
        name="Retained Earnings",
        category=AccountCategory.EQUITY,
        balance=Decimal("600000.0000000000"),
    ),
}

with tempfile.TemporaryDirectory() as tmp_dir:
    db_path = Path(tmp_dir) / "finance_ledger.sqlite"

    with StateEncoder(db_path) as encoder:
        ledger = ImmutableLedger(accounts, ocel_encoder=encoder)

        # Step 2: Construct and commit a balanced ImmutableTransaction (dr == cr and h_a == h_o)
        tx = ImmutableTransaction(
            transaction_id="tx_settle_invoice_001",
            timestamp="2026-11-17T14:00:00.123456Z",
            description="Pay vendor invoice with 10^-10 sub-cent precision",
            entries=(
                Entry(
                    account="accounts_payable",
                    magnitude=Decimal("25000.0000000001"),
                    direction=EntryDirection.DEBIT,
                    category=AccountCategory.LIABILITY,
                ),
                Entry(
                    account="cash",
                    magnitude=Decimal("25000.0000000001"),
                    direction=EntryDirection.CREDIT,
                    category=AccountCategory.ASSET,
                ),
            ),
        )
        ocel_event = ledger.commit(tx)
        assert ledger.total_assets == ledger.ownership_claims == Decimal("974999.9999999999")

        # Step 3: Verify that UPDATE/DELETE are blocked and reverse via a compensating entry
        try:
            ledger.delete_transaction("tx_settle_invoice_001")
        except LedgerImmutabilityError:
            pass

        reversal = ledger.reverse_transaction(
            "tx_settle_invoice_001",
            "tx_reversal_001",
            timestamp="2026-11-17T14:01:00Z",
        )
        assert ledger.get_balance("cash") == Decimal("1000000.0000000000")

    # Step 4: Verify OCEL 2.0 E2O qualifiers ("DebitAccount" and "CreditAccount")
    ocel = read_ocel2_sqlite(db_path)
    qualifiers = set(ocel.relations["ocel:qualifier"])
    assert qualifiers == {"DebitAccount", "CreditAccount"}
```
