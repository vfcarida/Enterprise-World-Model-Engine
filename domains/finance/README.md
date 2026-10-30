# Finance Domain Pack (`EWM-004`): Exact Double-Entry Ledger & Temporal Liquidity Constraints

This domain pack (`ewm_engine.domains.finance`) aligns the Enterprise World Model Engine with strict corporate accounting invariants and temporal liquidity constraints using exact Python `decimal.Decimal` arithmetic.

---

## 1. Design Principles & Mathematical Invariants

### 1.1 Exact `Decimal` Representation & `float` Prohibition

All monetary balances, minimum liquidity floors, and journal line amounts use `decimal.Decimal` with corporate precision safeguard (`getcontext().prec >= 38`, [`ensure_corporate_decimal_context`](src/ewm_engine/domains/finance/state.py)). Passing any `float` to [`FinancialState`](src/ewm_engine/domains/finance/state.py), [`LedgerAccount`](src/ewm_engine/domains/finance/state.py), [`JournalLine`](src/ewm_engine/domains/finance/state.py), or [`build_journal_action`](src/ewm_engine/domains/finance/state.py) emits a `DeprecationWarning` and raises `TypeError`.

### 1.2 Transaction Double-Entry Conservation

Every proposed [`JournalEntry`](src/ewm_engine/domains/finance/state.py) or journal `Action` evaluated by [`DoubleEntryConstraint`](src/ewm_engine/domains/finance/constraints.py) must satisfy:

$$\sum_{i} \Delta \text{Credits}_i - \sum_{i} \Delta \text{Debits}_i = \text{Decimal('0.00')}$$

Any transaction unbalanced by even `Decimal('0.01')` (\$0.01) is 100% rejected in the `EWM-002` `HybridTransitionRouter` `Pre-rules` stage before state evolution.

### 1.3 Balance-Sheet Identity & Account Sign Conventions

Accounts are partitioned into three categories ([`AccountCategory`](src/ewm_engine/domains/finance/state.py)):
- `ASSET` (debit-normal): $\Delta \text{Balance} = \Delta \text{Debit} - \Delta \text{Credit}$
- `LIABILITY` (credit-normal): $\Delta \text{Balance} = \Delta \text{Credit} - \Delta \text{Debit}$
- `EQUITY` (credit-normal): $\Delta \text{Balance} = \Delta \text{Credit} - \Delta \text{Debit}$

At every step (both `PRE_ACTION` and `POST_TRANSITION`), [`DoubleEntryConstraint`](src/ewm_engine/domains/finance/constraints.py) verifies the fundamental accounting identity with strictly `Decimal('0.00')` tolerance:

$$\sum_{a \in \text{Assets}} a.\text{balance} = \sum_{l \in \text{Liabilities}} l.\text{balance} + \sum_{e \in \text{Equity}} e.\text{balance}$$

### 1.4 Temporal Liquidity Constraints

Liquid asset accounts (`is_liquid=True`, such as `cash` and `treasury_bills`) must satisfy both individual account minimum reserve floors ($a.\text{balance} \ge a.\text{min\_balance}$) and the aggregate corporate liquidity floor:

$$\sum_{a \in \text{Liquid Assets}} a.\text{balance} \ge \text{min\_total\_liquidity}$$

---

## 2. Corporate Liquidity Drain Stress-Test Scenario

Implemented in [`scenario.py`](src/ewm_engine/domains/finance/scenario.py) ([`run_liquidity_drain_stress_test`](src/ewm_engine/domains/finance/scenario.py)), the scenario models a corporate treasury with initial balance sheet:

| Account ID | Category | Liquid? | Initial Balance (`Decimal`) | Minimum Floor (`Decimal`) |
| :--- | :---: | :---: | ---: | ---: |
| `cash` | `ASSET` | Yes | `10000000.00` (\$10.0M) | `500000.00` (\$0.5M) |
| `treasury_bills` | `ASSET` | Yes | `5000000.00` (\$5.0M) | `0.00` |
| `loans_receivable` | `ASSET` | No | `25000000.00` (\$25.0M) | `0.00` |
| `demand_deposits` | `LIABILITY` | — | `22000000.00` (\$22.0M) | `0.00` |
| `long_term_debt` | `LIABILITY` | — | `10000000.00` (\$10.0M) | `0.00` |
| `common_equity` | `EQUITY` | — | `5000000.00` (\$5.0M) | `0.00` |
| `retained_earnings` | `EQUITY` | — | `3000000.00` (\$3.0M) | `-Infinity` |

Using the `EWM-002` causal `engine.branch().do(...)` API:
1. **Baseline Branch (`treasury_main`)**: Settles recurring fee revenue (`Debit cash $50,000.00`, `Credit retained_earnings $50,000.00`).
2. **Liquidity Drain Stress Branch (`engine.branch("liquidity_drain_shock").do({"deposit_outflow_amount": 2500000})`)**: Severs inbound observational edges `market_stress -> deposit_outflow_amount` and forces a \$2.5M/step institutional deposit run (`Debit demand_deposits`, `Credit cash`), triggering automatic T-bill collateral liquidation (`Debit cash`, `Credit treasury_bills`) when cash approaches its minimum reserve floor.
3. **Emergency Repo Recovery Branch (`engine.branch("repo_facility_recovery").do({"deposit_outflow_amount": 2500000, "emergency_repo_draw": 1500000})`)**: Injects \$1.5M/step repo liquidity (`Debit cash`, `Credit long_term_debt`), preserving cash reserves while maintaining `accounting_discrepancy == Decimal('0.00')`.

---

## 3. How to Extend the Chart of Accounts and Journal Rules

### Adding New Accounts

Register additional `LedgerAccount` entries in `FinancialState.accounts` while ensuring initial $\sum \text{Assets} = \sum \text{Liabilities} + \sum \text{Equity}$:

```python
from decimal import Decimal
from ewm_engine.domains.finance import (
    AccountCategory,
    FinancialState,
    LedgerAccount,
    build_corporate_treasury_engine,
    build_journal_action,
)

custom_state = FinancialState(
    min_total_liquidity=Decimal("1000000.00"),
    accounts={
        "cash": LedgerAccount(
            account_id="cash",
            name="Operating Cash",
            category=AccountCategory.ASSET,
            balance=Decimal("5000000.00"),
            min_balance=Decimal("250000.00"),
            is_liquid=True,
        ),
        "accounts_payable": LedgerAccount(
            account_id="accounts_payable",
            name="Vendor Accounts Payable",
            category=AccountCategory.LIABILITY,
            balance=Decimal("2000000.00"),
            min_balance=Decimal("0.00"),
        ),
        "retained_earnings": LedgerAccount(
            account_id="retained_earnings",
            name="Retained Earnings",
            category=AccountCategory.EQUITY,
            balance=Decimal("3000000.00"),
        ),
    },
)

engine, _ = build_corporate_treasury_engine(custom_state)
```

### Submitting Multi-Leg Double-Entry Journal Actions

Use `build_journal_action` with `Decimal` or string amounts (never `float`):

```python
action = build_journal_action(
    "pay_vendor_invoice_001",
    debits={"accounts_payable": Decimal("125000.50")},
    credits={"cash": Decimal("125000.50")},
    description="Settle vendor invoice #001",
)
outcome = engine.step(action=action)
assert not outcome.pre_rule_rejected
```
