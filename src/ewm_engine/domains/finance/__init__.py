"""Finance Domain Pack (`EWM-004`): Exact `Decimal` Double-Entry Accounting & Liquidity Stress Engine.

Enforces:
- `FinancialState` and `LedgerAccount` using exact `decimal.Decimal` (raising `TypeError` on `float`).
- `DoubleEntryConstraint` validating $\\sum \\Delta \\text{Credits} - \\sum \\Delta \\text{Debits} = 0$
  and $\\sum \\text{Assets} = \\sum \\text{Liabilities} + \\sum \\text{Equity}$ with `0.00` tolerance
  through the `EWM-002` `HybridTransitionRouter`.
- Counterfactual corporate liquidity drain stress testing via `engine.branch().do(...)`.
"""

from __future__ import annotations

from ewm_engine.domains.finance.constraints import (
    DoubleEntryConstraint,
    parse_journal_entry_from_action,
)
from ewm_engine.domains.finance.scenario import (
    DecimalVsFloatBenchmarkReport,
    DoubleEntryLedgerDynamics,
    LiquidityStressTestReport,
    benchmark_decimal_vs_float_rollout,
    build_corporate_treasury_engine,
    create_default_treasury_state,
    run_liquidity_drain_stress_test,
)
from ewm_engine.domains.finance.state import (
    MIN_CORPORATE_DECIMAL_PRECISION,
    ZERO_DECIMAL,
    AccountCategory,
    FinancialState,
    JournalEntry,
    JournalLine,
    LedgerAccount,
    build_journal_action,
    deprecated_float_api_guard,
    ensure_corporate_decimal_context,
    to_exact_decimal,
)

__all__ = [
    "MIN_CORPORATE_DECIMAL_PRECISION",
    "ZERO_DECIMAL",
    "AccountCategory",
    "DecimalVsFloatBenchmarkReport",
    "DoubleEntryConstraint",
    "DoubleEntryLedgerDynamics",
    "FinancialState",
    "JournalEntry",
    "JournalLine",
    "LedgerAccount",
    "LiquidityStressTestReport",
    "benchmark_decimal_vs_float_rollout",
    "build_corporate_treasury_engine",
    "build_journal_action",
    "create_default_treasury_state",
    "deprecated_float_api_guard",
    "ensure_corporate_decimal_context",
    "parse_journal_entry_from_action",
    "run_liquidity_drain_stress_test",
    "to_exact_decimal",
]
