"""Exact Decimal Financial State and Double-Entry Ledger Primitives (`EWM-004`).

Enforces:
1. All monetary balances and journal deltas use Python `decimal.Decimal` (never `float`).
2. Passing any `float` to `FinancialState`, `LedgerAccount`, or `JournalEntry` raises `TypeError`.
3. Corporate-grade Decimal precision safeguard (`prec >= 38`) prevents silent truncation of
   multi-trillion-dollar corporate or sovereign balances down to sub-cent precision.
4. Exact accounting identity:
   $$\\sum \\text{Assets} = \\sum \\text{Liabilities} + \\sum \\text{Equity}$$
   and transaction double-entry identity:
   $$\\sum \\Delta \\text{Credits} - \\sum \\Delta \\text{Debits} = \\text{Decimal('0.00')}$$
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation, getcontext
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ewm_engine.core.actions import Action
from ewm_engine.core.state import Entity, Resource, WorldState

MIN_CORPORATE_DECIMAL_PRECISION: int = 38
ZERO_DECIMAL: Decimal = Decimal("0.00")


def ensure_corporate_decimal_context(min_precision: int = MIN_CORPORATE_DECIMAL_PRECISION) -> int:
    """Operational Safeguard (EWM-004 Section 11): Ensure `getcontext().prec` never truncates corporate values."""
    ctx = getcontext()
    if ctx.prec < min_precision:
        ctx.prec = min_precision
    return int(ctx.prec)


ensure_corporate_decimal_context()


def to_exact_decimal(value: Any, *, field_name: str = "monetary_value") -> Decimal:
    """Convert an exact numeric input (`Decimal`, `str`, `int`) to `Decimal`, rejecting `float`.

    Raises:
        TypeError: If `value` is a `float`, `bool`, or unsupported type (EWM-004 Section 13).
        ValueError: If `value` is a non-finite (`NaN`/`Inf`) or malformed string.
    """
    ensure_corporate_decimal_context()
    if isinstance(value, float):
        raise TypeError(
            f"Float values are prohibited in Finance domain ({field_name}={value!r}). "
            "Use exact decimal.Decimal, str, or int to guarantee zero floating-point drift."
        )
    if isinstance(value, bool):
        raise TypeError(f"Boolean values are not valid monetary amounts ({field_name}={value!r}).")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"Non-finite Decimal value is prohibited ({field_name}={value!r}).")
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, str):
        try:
            dec = Decimal(value.strip())
        except InvalidOperation as exc:
            raise ValueError(f"Invalid Decimal string for {field_name}: {value!r}") from exc
        if not dec.is_finite():
            raise ValueError(f"Non-finite Decimal string is prohibited ({field_name}={value!r}).")
        return dec
    raise TypeError(
        f"Unsupported monetary type for {field_name}: {type(value).__name__}. "
        "Expected decimal.Decimal, str, or int."
    )


def deprecated_float_api_guard(value: Any, *, api_name: str = "FinancialState") -> None:
    """Emit a deprecation warning and raise `TypeError` when legacy `float` values are passed."""
    if isinstance(value, float):
        warnings.warn(
            f"{api_name}: float-based monetary inputs are deprecated and removed in EWM-004; "
            "pass decimal.Decimal or str instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise TypeError(
            f"{api_name} rejected float input {value!r}: float-based APIs are prohibited."
        )


class AccountCategory(StrEnum):
    """Standard double-entry balance-sheet account classification."""

    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"


class LedgerAccount(BaseModel):
    """Immutable double-entry ledger account with exact `Decimal` balance and liquidity floor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    account_id: str = Field(min_length=1, description="Unique account identifier.")
    name: str = Field(min_length=1, description="Human-readable account name.")
    category: AccountCategory = Field(
        description="Balance-sheet category (asset, liability, equity)."
    )
    balance: Decimal = Field(
        default=ZERO_DECIMAL,
        description="Current exact account balance in Decimal.",
    )
    min_balance: Decimal | None = Field(
        default=None,
        description="Minimum allowed balance (e.g., 0.00 or reserve requirement for liquid assets; None if unbounded).",
    )
    is_liquid: bool = Field(
        default=False,
        description="True if this asset account counts toward immediate corporate liquidity reserves.",
    )

    @field_validator("balance", mode="before")
    @classmethod
    def _validate_balance(cls, v: Any) -> Decimal:
        deprecated_float_api_guard(v, api_name="LedgerAccount.balance")
        return to_exact_decimal(v, field_name="LedgerAccount.balance")

    @field_validator("min_balance", mode="before")
    @classmethod
    def _validate_min_balance(cls, v: Any) -> Decimal | None:
        if v is None:
            return None
        deprecated_float_api_guard(v, api_name="LedgerAccount.min_balance")
        if isinstance(v, Decimal) and v.is_infinite() and v.is_signed():
            return None
        if isinstance(v, str) and v.strip().lower() in ("none", "-inf", "-infinity"):
            return None
        return to_exact_decimal(v, field_name="LedgerAccount.min_balance")

    def apply_debit_credit(
        self,
        *,
        debit: Decimal = ZERO_DECIMAL,
        credit: Decimal = ZERO_DECIMAL,
    ) -> LedgerAccount:
        """Return a new `LedgerAccount` after applying exact debit and credit postings.

        Accounting sign conventions:
        - `ASSET` (debit-normal): $\\Delta \\text{balance} = \\text{debit} - \\text{credit}$
        - `LIABILITY` / `EQUITY` (credit-normal): $\\Delta \\text{balance} = \\text{credit} - \\text{debit}$
        """
        ensure_corporate_decimal_context()
        d_val = to_exact_decimal(debit, field_name="debit")
        c_val = to_exact_decimal(credit, field_name="credit")
        if self.category == AccountCategory.ASSET:
            new_bal = self.balance + (d_val - c_val)
        else:
            new_bal = self.balance + (c_val - d_val)
        return self.model_copy(update={"balance": new_bal})


class JournalLine(BaseModel):
    """A single debit or credit posting line in a double-entry `JournalEntry`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    account_id: str = Field(min_length=1)
    debit: Decimal = Field(default=ZERO_DECIMAL)
    credit: Decimal = Field(default=ZERO_DECIMAL)
    memo: str = Field(default="")

    @field_validator("debit", "credit", mode="before")
    @classmethod
    def _validate_non_negative_decimal(cls, v: Any) -> Decimal:
        deprecated_float_api_guard(v, api_name="JournalLine")
        dec = to_exact_decimal(v, field_name="JournalLine.amount")
        if dec < ZERO_DECIMAL:
            raise ValueError(f"JournalLine debit/credit amounts must be >= 0.00, got {dec}")
        return dec


class JournalEntry(BaseModel):
    """Immutable double-entry journal entry with $\\sum \\Delta \\text{Credits} - \\sum \\Delta \\text{Debits} = 0$."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_id: str = Field(min_length=1)
    step: int = Field(default=0, ge=0)
    description: str = Field(default="")
    lines: tuple[JournalLine, ...] = Field(default_factory=tuple)

    @property
    def total_debits(self) -> Decimal:
        ensure_corporate_decimal_context()
        return sum((line.debit for line in self.lines), start=ZERO_DECIMAL)

    @property
    def total_credits(self) -> Decimal:
        ensure_corporate_decimal_context()
        return sum((line.credit for line in self.lines), start=ZERO_DECIMAL)

    @property
    def net_imbalance(self) -> Decimal:
        """Compute $\\sum \\Delta \\text{Credits} - \\sum \\Delta \\text{Debits}$ in exact Decimal."""
        ensure_corporate_decimal_context()
        return self.total_credits - self.total_debits

    @property
    def is_balanced(self) -> bool:
        return self.net_imbalance == ZERO_DECIMAL

    def to_action(self) -> Action:
        """Convert this `JournalEntry` into a domain-neutral `Action` for `TransitionEngine`."""
        return Action(
            id=self.entry_id,
            type="double_entry_journal",
            parameters={
                "description": self.description,
                "lines": [
                    {
                        "account_id": line.account_id,
                        "debit": str(line.debit),
                        "credit": str(line.credit),
                        "memo": line.memo,
                    }
                    for line in self.lines
                ],
            },
        )


def build_journal_action(
    entry_id: str,
    *,
    debits: Mapping[str, Decimal | str | int],
    credits: Mapping[str, Decimal | str | int],
    description: str = "",
) -> Action:
    """Create a double-entry `Action` from debit and credit account mappings, rejecting any `float`."""
    ensure_corporate_decimal_context()
    lines: list[dict[str, str]] = []
    for acc_id, amt in debits.items():
        deprecated_float_api_guard(amt, api_name="build_journal_action.debits")
        dec = to_exact_decimal(amt, field_name=f"debits[{acc_id!r}]")
        if dec < ZERO_DECIMAL:
            raise ValueError(f"Debit amount for {acc_id!r} must be >= 0.00, got {dec}")
        lines.append(
            {
                "account_id": str(acc_id),
                "debit": str(dec),
                "credit": "0.00",
                "memo": description,
            }
        )
    for acc_id, amt in credits.items():
        deprecated_float_api_guard(amt, api_name="build_journal_action.credits")
        dec = to_exact_decimal(amt, field_name=f"credits[{acc_id!r}]")
        if dec < ZERO_DECIMAL:
            raise ValueError(f"Credit amount for {acc_id!r} must be >= 0.00, got {dec}")
        lines.append(
            {
                "account_id": str(acc_id),
                "debit": "0.00",
                "credit": str(dec),
                "memo": description,
            }
        )
    return Action(
        id=entry_id,
        type="double_entry_journal",
        parameters={
            "description": description,
            "lines": lines,
        },
    )


class FinancialState(BaseModel):
    """Immutable corporate ledger state using exact `decimal.Decimal` monetary representation.

    Guarantees:
    - All account balances and liquidity thresholds are exact `Decimal` instances.
    - Passing any `float` raises `TypeError` (EWM-004 Section 6 & Section 13).
    - Fundamental accounting invariant:
      $$\\sum \\text{Assets} = \\sum \\text{Liabilities} + \\sum \\text{Equity}$$
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(default=0, ge=0)
    timestamp: Decimal = Field(default=ZERO_DECIMAL)
    accounts: dict[str, LedgerAccount] = Field(default_factory=dict)
    min_total_liquidity: Decimal = Field(
        default=ZERO_DECIMAL,
        description="Temporal liquidity floor required across all liquid asset accounts.",
    )
    journal_history: tuple[JournalEntry, ...] = Field(default_factory=tuple)

    @field_validator("timestamp", "min_total_liquidity", mode="before")
    @classmethod
    def _validate_decimal_fields(cls, v: Any) -> Decimal:
        deprecated_float_api_guard(v, api_name="FinancialState")
        return to_exact_decimal(v, field_name="FinancialState.field")

    @model_validator(mode="before")
    @classmethod
    def _reject_float_anywhere_in_init(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            for k, v in data.items():
                if isinstance(v, float):
                    deprecated_float_api_guard(v, api_name=f"FinancialState.{k}")
        return data

    @property
    def total_assets(self) -> Decimal:
        """Return exact $\\sum_{a \\in \\text{Assets}} a.\\text{balance}$."""
        ensure_corporate_decimal_context()
        return sum(
            (a.balance for a in self.accounts.values() if a.category == AccountCategory.ASSET),
            start=ZERO_DECIMAL,
        )

    @property
    def total_liabilities(self) -> Decimal:
        """Return exact $\\sum_{l \\in \\text{Liabilities}} l.\\text{balance}$."""
        ensure_corporate_decimal_context()
        return sum(
            (a.balance for a in self.accounts.values() if a.category == AccountCategory.LIABILITY),
            start=ZERO_DECIMAL,
        )

    @property
    def total_equity(self) -> Decimal:
        """Return exact $\\sum_{e \\in \\text{Equity}} e.\\text{balance}$."""
        ensure_corporate_decimal_context()
        return sum(
            (a.balance for a in self.accounts.values() if a.category == AccountCategory.EQUITY),
            start=ZERO_DECIMAL,
        )

    @property
    def total_liquidity(self) -> Decimal:
        """Return exact sum of balances across all liquid asset accounts (`is_liquid=True`)."""
        ensure_corporate_decimal_context()
        return sum(
            (
                a.balance
                for a in self.accounts.values()
                if a.category == AccountCategory.ASSET and a.is_liquid
            ),
            start=ZERO_DECIMAL,
        )

    @property
    def accounting_discrepancy(self) -> Decimal:
        """Compute $\\sum \\text{Assets} - (\\sum \\text{Liabilities} + \\sum \\text{Equity})$."""
        ensure_corporate_decimal_context()
        return self.total_assets - (self.total_liabilities + self.total_equity)

    @property
    def is_balanced(self) -> bool:
        """Return True iff $\\sum \\text{Assets} == \\sum \\text{Liabilities} + \\sum \\text{Equity}$ with 0.00 tolerance."""
        return self.accounting_discrepancy == ZERO_DECIMAL

    @property
    def meets_liquidity_constraints(self) -> bool:
        """Return True iff total liquidity and every individual account meet their minimum floors."""
        if self.total_liquidity < self.min_total_liquidity:
            return False
        for acc in self.accounts.values():
            if acc.min_balance is not None and acc.balance < acc.min_balance:
                return False
        return True

    def get_balance(self, account_id: str) -> Decimal:
        """Return the exact `Decimal` balance of `account_id`."""
        if account_id not in self.accounts:
            raise KeyError(f"Unknown ledger account {account_id!r}")
        return self.accounts[account_id].balance

    def apply_journal_entry(self, entry: JournalEntry) -> FinancialState:
        """Apply a `JournalEntry` in exact `Decimal` arithmetic and return a new `FinancialState`."""
        ensure_corporate_decimal_context()
        updated_accounts = dict(self.accounts)
        for line in entry.lines:
            if line.account_id not in updated_accounts:
                raise KeyError(
                    f"Unknown ledger account {line.account_id!r} in journal entry {entry.entry_id!r}"
                )
            acc = updated_accounts[line.account_id]
            updated_accounts[line.account_id] = acc.apply_debit_credit(
                debit=line.debit,
                credit=line.credit,
            )
        return FinancialState(
            step=self.step + 1,
            timestamp=self.timestamp + Decimal("1"),
            accounts=updated_accounts,
            min_total_liquidity=self.min_total_liquidity,
            journal_history=(*self.journal_history, entry),
        )

    def to_world_state(
        self,
        *,
        causal_drivers: Mapping[str, Decimal | str | int] | None = None,
    ) -> WorldState:
        """Embed this exact `FinancialState` into a domain-neutral `WorldState`.

        All exact `Decimal` balances are preserved losslessly as canonical strings inside
        `WorldState.memory['financial_ledger']` while exposing causal driver nodes in
        `WorldState.resources` for `TransitionEngine.branch().do()` graph surgery.
        """
        ensure_corporate_decimal_context()
        serialized_accounts = {
            aid: {
                "account_id": acc.account_id,
                "name": acc.name,
                "category": acc.category.value,
                "balance": str(acc.balance),
                "min_balance": None if acc.min_balance is None else str(acc.min_balance),
                "is_liquid": acc.is_liquid,
            }
            for aid, acc in self.accounts.items()
        }
        serialized_drivers: dict[str, str] = {}
        if causal_drivers:
            for dk, dv in causal_drivers.items():
                deprecated_float_api_guard(dv, api_name=f"causal_drivers[{dk!r}]")
                serialized_drivers[str(dk)] = str(to_exact_decimal(dv, field_name=dk))

        resources: dict[str, Resource] = {}
        for aid, acc in self.accounts.items():
            resources[aid] = Resource(
                id=aid,
                current=float(acc.balance),
                min_value=-1e18,
                max_value=1e18,
            )
        for dk, dstr in serialized_drivers.items():
            resources[dk] = Resource(
                id=dk,
                current=float(Decimal(dstr)),
                min_value=-1e18,
                max_value=1e18,
            )

        return WorldState(
            step=self.step,
            timestamp=float(self.timestamp),
            entities={
                "general_ledger": Entity(
                    id="general_ledger",
                    type="double_entry_ledger",
                    attributes={
                        "total_assets": str(self.total_assets),
                        "total_liabilities": str(self.total_liabilities),
                        "total_equity": str(self.total_equity),
                        "total_liquidity": str(self.total_liquidity),
                        "discrepancy": str(self.accounting_discrepancy),
                    },
                )
            },
            resources=resources,
            memory={
                "financial_ledger": {
                    "step": self.step,
                    "timestamp": str(self.timestamp),
                    "min_total_liquidity": str(self.min_total_liquidity),
                    "accounts": serialized_accounts,
                    "causal_drivers": serialized_drivers,
                    "journal_count": len(self.journal_history),
                }
            },
        )

    @classmethod
    def from_world_state(cls, state: WorldState) -> FinancialState:
        """Reconstruct the exact `Decimal`-backed `FinancialState` from a `WorldState`."""
        ensure_corporate_decimal_context()
        ledger_mem = state.memory.get("financial_ledger")
        if not isinstance(ledger_mem, Mapping):
            raise ValueError("WorldState does not contain a 'financial_ledger' memory payload.")

        raw_accounts = ledger_mem.get("accounts", {})
        accounts: dict[str, LedgerAccount] = {}
        for aid, acc_dict in raw_accounts.items():
            raw_min = acc_dict.get("min_balance")
            accounts[str(aid)] = LedgerAccount(
                account_id=str(acc_dict["account_id"]),
                name=str(acc_dict["name"]),
                category=AccountCategory(str(acc_dict["category"])),
                balance=Decimal(str(acc_dict["balance"])),
                min_balance=None if raw_min is None else Decimal(str(raw_min)),
                is_liquid=bool(acc_dict.get("is_liquid", False)),
            )

        return cls(
            step=int(state.step),
            timestamp=Decimal(str(ledger_mem.get("timestamp", state.step))),
            accounts=accounts,
            min_total_liquidity=Decimal(str(ledger_mem.get("min_total_liquidity", "0.00"))),
        )
