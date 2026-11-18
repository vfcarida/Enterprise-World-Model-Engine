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

import threading
import warnings
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation, getcontext
from enum import StrEnum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ewm_engine.core.actions import Action
from ewm_engine.core.deprecation import deprecated
from ewm_engine.core.ocel import (
    OcelEventRecord,
    OcelObjectRelation,
    OcelObjectTarget,
    StateEncoder,
)
from ewm_engine.core.state import Entity, Resource, WorldState
from ewm_engine.core.types import EngineError

MIN_CORPORATE_DECIMAL_PRECISION: int = 50
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


# ==============================================================================
# E6: Immutable Double-Entry Transactions, Tuple-Relational Equilibrium (h_a = h_o),
# Append-Only ImmutableLedger, and OCEL 2.0 E2O ("DebitAccount" / "CreditAccount")
# ==============================================================================


class BalanceError(EngineError):
    """Raised when a financial transaction violates double-entry balance (dr != cr) or h_a != h_o."""


class LedgerImmutabilityError(EngineError):
    """Raised when an UPDATE or DELETE operation is attempted on an append-only ImmutableLedger."""


class EntryDirection(StrEnum):
    """Accounting direction for an immutable ledger entry leg."""

    DEBIT = "DEBIT"
    CREDIT = "CREDIT"


class Entry(BaseModel):
    """A single immutable leg (Account, Magnitude, Direction) of an `ImmutableTransaction` (E6).

    Strictly requires `decimal.Decimal` (or `str` / `int` converted to `Decimal`) for
    `magnitude` and raises `TypeError` if a `float` is passed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    account: str = Field(min_length=1, description="Target ledger account identifier.")
    magnitude: Decimal = Field(
        description="Exact positive monetary magnitude in arbitrary-precision Decimal."
    )
    direction: EntryDirection = Field(
        description="Accounting leg direction: EntryDirection.DEBIT or EntryDirection.CREDIT."
    )
    category: Optional[AccountCategory] = Field(
        default=None,
        description="Optional balance-sheet classification (asset, liability, equity) for tuple-relational checks.",
    )
    memo: str = Field(default="", description="Optional line-item memo.")

    @field_validator("magnitude", mode="before")
    @classmethod
    def _validate_exact_decimal_magnitude(cls, v: Any) -> Decimal:
        deprecated_float_api_guard(v, api_name="Entry.magnitude")
        dec = to_exact_decimal(v, field_name="Entry.magnitude")
        if dec <= Decimal("0"):
            raise ValueError(f"Entry.magnitude must be strictly positive (> 0), got {dec}.")
        return dec

    @field_validator("direction", mode="before")
    @classmethod
    def _normalize_direction(cls, v: Any) -> EntryDirection:
        if isinstance(v, EntryDirection):
            return v
        if isinstance(v, str):
            norm = v.strip().upper()
            if norm in ("DR", "DEBIT"):
                return EntryDirection.DEBIT
            if norm in ("CR", "CREDIT"):
                return EntryDirection.CREDIT
        raise ValueError(f"Invalid EntryDirection {v!r}; expected 'DEBIT' or 'CREDIT'.")

    def inverted(self, *, memo_prefix: str = "REVERSAL: ") -> Entry:
        """Return the compensating inverse Entry with swapped DEBIT <-> CREDIT direction."""
        inv_dir = (
            EntryDirection.CREDIT
            if self.direction == EntryDirection.DEBIT
            else EntryDirection.DEBIT
        )
        new_memo = f"{memo_prefix}{self.memo}".strip() if self.memo else memo_prefix.strip()
        return Entry(
            account=self.account,
            magnitude=self.magnitude,
            direction=inv_dir,
            category=self.category,
            memo=new_memo,
        )


class ImmutableTransaction(BaseModel):
    """Immutable double-entry financial transaction guaranteeing $\\sum dr - \\sum cr = 0$ (E6).

    Enforces:
    1. All entry magnitudes are exact `Decimal` instances (`float` raises `TypeError`).
    2. At least two entries (one debit, one credit) with $\\sum \\text{debits} == \\sum \\text{credits}$
       ($dr - cr = \\text{Decimal('0')}$); raises `BalanceError` on any discrepancy, even $10^{-10}$ or smaller.
    3. When account categories are supplied on entries, validates the tuple-relational
       equilibrium condition $h_a = h_o$, where:
       - $h_a = \\sum_{e \\in \\text{Assets}} (dr_e - cr_e)$ (net change to asset classes),
       - $h_o = \\sum_{e \\in \\text{Liabilities} \\cup \\text{Equity}} (cr_e - dr_e)$ (net change to ownership claims).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    transaction_id: str = Field(min_length=1, description="Globally unique transaction identifier.")
    entries: Tuple[Entry, ...] = Field(
        description="Immutable tuple of Entry legs (Account, Magnitude, Direction)."
    )
    timestamp: Union[str, int, Decimal] = Field(
        default="2026-11-17T00:00:00Z",
        description="ISO-8601 timestamp or integer epoch step (float prohibited).",
    )
    description: str = Field(default="", description="Audit description of the transaction.")
    compensates_transaction_id: Optional[str] = Field(
        default=None,
        description="If this transaction is a compensating reversal entry, the ID of the reversed transaction.",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def _reject_float_timestamp(cls, v: Any) -> Union[str, int, Decimal]:
        deprecated_float_api_guard(v, api_name="ImmutableTransaction.timestamp")
        if isinstance(v, (str, int, Decimal)):
            return v
        raise TypeError(
            f"ImmutableTransaction.timestamp rejects {type(v).__name__}; use ISO-8601 str, int, or Decimal."
        )

    @model_validator(mode="after")
    def _validate_double_entry_and_equilibrium(self) -> ImmutableTransaction:
        ensure_corporate_decimal_context()
        if len(self.entries) < 2:
            raise BalanceError(
                f"ImmutableTransaction '{self.transaction_id}' requires at least 2 entries "
                f"(got {len(self.entries)})."
            )

        debits = sum(
            (e.magnitude for e in self.entries if e.direction == EntryDirection.DEBIT),
            start=Decimal("0"),
        )
        credits = sum(
            (e.magnitude for e in self.entries if e.direction == EntryDirection.CREDIT),
            start=Decimal("0"),
        )
        diff = debits - credits
        if diff != Decimal("0"):
            raise BalanceError(
                f"ImmutableTransaction '{self.transaction_id}' failed double-entry balance check: "
                f"sum(debits)={debits} != sum(credits)={credits} (dr - cr = {diff} != 0).",
                details={
                    "transaction_id": self.transaction_id,
                    "sum_debits": str(debits),
                    "sum_credits": str(credits),
                    "imbalance": str(diff),
                },
            )

        # If all entries carry explicit AccountCategory tags, verify h_a == h_o directly
        if all(e.category is not None for e in self.entries):
            h_a, h_o = self.compute_tuple_equilibrium({})
            if h_a != h_o:
                raise BalanceError(
                    f"ImmutableTransaction '{self.transaction_id}' violated tuple-relational "
                    f"equilibrium: h_a ({h_a}) != h_o ({h_o})."
                )

        return self

    @property
    def total_debits(self) -> Decimal:
        """Return exact $\\sum dr$ across all debit entries."""
        ensure_corporate_decimal_context()
        return sum(
            (e.magnitude for e in self.entries if e.direction == EntryDirection.DEBIT),
            start=Decimal("0"),
        )

    @property
    def total_credits(self) -> Decimal:
        """Return exact $\\sum cr$ across all credit entries."""
        ensure_corporate_decimal_context()
        return sum(
            (e.magnitude for e in self.entries if e.direction == EntryDirection.CREDIT),
            start=Decimal("0"),
        )

    def compute_tuple_equilibrium(
        self,
        account_categories: Mapping[str, AccountCategory],
    ) -> Tuple[Decimal, Decimal]:
        """Compute $(h_a, h_o)$ for the tuple-relational equilibrium condition $h_a = h_o$ (E6 Req 8).

        Let:
        - $h_a = \\sum_{e \\in \\text{Asset Classes}} (dr_e - cr_e)$ be the net change to asset classes, and
        - $h_o = \\sum_{e \\in \\text{Ownership Claims}} (cr_e - dr_e)$ be the net change to ownership
          claims (liabilities and equity).

        Returns:
            Tuple `(h_a, h_o)` in exact `Decimal`.
        """
        ensure_corporate_decimal_context()
        h_a = Decimal("0")
        h_o = Decimal("0")
        for entry in self.entries:
            cat = entry.category or account_categories.get(entry.account)
            if cat is None:
                raise BalanceError(
                    f"Account '{entry.account}' in transaction '{self.transaction_id}' has no registered AccountCategory."
                )
            if cat == AccountCategory.ASSET:
                if entry.direction == EntryDirection.DEBIT:
                    h_a += entry.magnitude
                else:
                    h_a -= entry.magnitude
            else:
                # Ownership claims: LIABILITY or EQUITY (credit-normal)
                if entry.direction == EntryDirection.CREDIT:
                    h_o += entry.magnitude
                else:
                    h_o -= entry.magnitude
        return h_a, h_o

    def create_reversal(
        self,
        reversal_transaction_id: str,
        *,
        timestamp: Union[str, int, Decimal] = "2026-11-17T00:00:01Z",
        reason: str = "Compensating reversal entry",
    ) -> ImmutableTransaction:
        """Construct a compensating reversal transaction that mathematically zeroes out this transaction."""
        reversed_entries = tuple(e.inverted() for e in self.entries)
        return ImmutableTransaction(
            transaction_id=reversal_transaction_id,
            entries=reversed_entries,
            timestamp=timestamp,
            description=f"{reason} (reverses {self.transaction_id})",
            compensates_transaction_id=self.transaction_id,
        )

    def to_ocel_event(self) -> OcelEventRecord:
        """Map this `ImmutableTransaction` into an E2 OCEL 2.0 `OcelEventRecord` (E6 Req 7).

        Uses explicit OCEL 2.0 E2O object qualifiers:
        - `"DebitAccount"` for every `Entry` with `direction == EntryDirection.DEBIT`
        - `"CreditAccount"` for every `Entry` with `direction == EntryDirection.CREDIT`
        and creates directed O2O value-flow links from each `CreditAccount` to each `DebitAccount`.
        """
        targets: List[OcelObjectTarget] = []
        debit_accounts: List[str] = []
        credit_accounts: List[str] = []

        for entry in self.entries:
            qualifier = (
                "DebitAccount" if entry.direction == EntryDirection.DEBIT else "CreditAccount"
            )
            if entry.direction == EntryDirection.DEBIT:
                debit_accounts.append(entry.account)
            else:
                credit_accounts.append(entry.account)

            obj_attrs: Dict[str, Any] = {}
            if entry.category is not None:
                obj_attrs["category"] = entry.category.value

            targets.append(
                OcelObjectTarget(
                    object_id=entry.account,
                    object_type="LedgerAccount",
                    qualifier=qualifier,
                    attributes=obj_attrs,
                )
            )

        o2o_relations: List[OcelObjectRelation] = []
        for cr_acc in credit_accounts:
            for dr_acc in debit_accounts:
                if cr_acc != dr_acc:
                    o2o_relations.append(
                        OcelObjectRelation(
                            source_id=cr_acc,
                            target_id=dr_acc,
                            qualifier="double_entry_transfer",
                        )
                    )

        ts_val: Union[str, int] = (
            str(self.timestamp) if isinstance(self.timestamp, Decimal) else self.timestamp
        )
        # If timestamp is a plain integer step or decimal string without ISO format, map to epoch int or ISO string
        if isinstance(ts_val, str) and ts_val.isdigit():
            ts_val = int(ts_val)

        return OcelEventRecord(
            event_id=self.transaction_id,
            event_type=(
                "CompensatingReversalTransaction"
                if self.compensates_transaction_id
                else "DoubleEntryTransaction"
            ),
            timestamp=ts_val,
            objects=tuple(targets),
            attributes={
                "total_debits": str(self.total_debits),
                "total_credits": str(self.total_credits),
                "entry_count": len(self.entries),
                "description": self.description,
                "is_reversal": self.compensates_transaction_id is not None,
                "compensates_transaction_id": self.compensates_transaction_id or "",
            },
            object_relations=tuple(o2o_relations),
        )


class ImmutableLedger:
    """Append-only, double-entry financial ledger enforcing $dr - cr = 0$ and $h_a = h_o$ (E6).

    Operational Safeguards (E6 Requirement 11):
    - Strictly prohibits `UPDATE` or `DELETE` operations on committed ledger state; calling
      `update_transaction`, `delete_transaction`, item assignment, or item deletion raises
      `LedgerImmutabilityError`.
    - Error correction is permitted exclusively via `reverse_transaction(...)`, which appends
      a compensating reversal `ImmutableTransaction`.
    - Automatically maps every committed financial transaction into an E2 OCEL 2.0 `StateEncoder`
      using explicit `"DebitAccount"` and `"CreditAccount"` E2O qualifiers.
    """

    def __init__(
        self,
        accounts: Optional[Mapping[str, LedgerAccount]] = None,
        *,
        ocel_encoder: Optional[StateEncoder] = None,
    ) -> None:
        ensure_corporate_decimal_context()
        self._lock = threading.RLock()
        self._accounts: Dict[str, LedgerAccount] = dict(accounts or {})
        self._transactions: List[ImmutableTransaction] = []
        self._tx_index: Dict[str, ImmutableTransaction] = {}
        self._ocel_encoder: Optional[StateEncoder] = ocel_encoder

        # Verify initial balance sheet equilibrium (h_a == h_o) if accounts are non-empty
        if self._accounts:
            discrepancy = self.total_assets - (self.total_liabilities + self.total_equity)
            if discrepancy != Decimal("0"):
                raise BalanceError(
                    f"Initial ImmutableLedger accounts violate Assets == Liabilities + Equity "
                    f"(discrepancy = {discrepancy} != 0)."
                )

    @property
    def accounts(self) -> Dict[str, LedgerAccount]:
        """Return a defensive copy of all ledger accounts."""
        with self._lock:
            return dict(self._accounts)

    @property
    def transactions(self) -> Tuple[ImmutableTransaction, ...]:
        """Return the immutable sequence of committed transactions."""
        with self._lock:
            return tuple(self._transactions)

    @property
    def total_assets(self) -> Decimal:
        """Return exact $\\sum_{a \\in \\text{Assets}} a.\\text{balance}$."""
        ensure_corporate_decimal_context()
        with self._lock:
            return sum(
                (a.balance for a in self._accounts.values() if a.category == AccountCategory.ASSET),
                start=Decimal("0"),
            )

    @property
    def total_liabilities(self) -> Decimal:
        """Return exact $\\sum_{l \\in \\text{Liabilities}} l.\\text{balance}$."""
        ensure_corporate_decimal_context()
        with self._lock:
            return sum(
                (
                    a.balance
                    for a in self._accounts.values()
                    if a.category == AccountCategory.LIABILITY
                ),
                start=Decimal("0"),
            )

    @property
    def total_equity(self) -> Decimal:
        """Return exact $\\sum_{e \\in \\text{Equity}} e.\\text{balance}$."""
        ensure_corporate_decimal_context()
        with self._lock:
            return sum(
                (
                    a.balance
                    for a in self._accounts.values()
                    if a.category == AccountCategory.EQUITY
                ),
                start=Decimal("0"),
            )

    @property
    def ownership_claims(self) -> Decimal:
        """Return total ownership claims ($\\sum \\text{Liabilities} + \\sum \\text{Equity}$)."""
        ensure_corporate_decimal_context()
        with self._lock:
            return self.total_liabilities + self.total_equity

    def register_account(self, account: LedgerAccount) -> None:
        """Register a new zero-balance `LedgerAccount` in the chart of accounts."""
        with self._lock:
            if account.account_id in self._accounts:
                raise LedgerImmutabilityError(
                    f"Account '{account.account_id}' already exists; cannot overwrite existing account."
                )
            if account.balance != Decimal("0"):
                raise BalanceError(
                    f"Newly registered account '{account.account_id}' must start with 0 balance; "
                    f"fund it via a balanced ImmutableTransaction."
                )
            self._accounts[account.account_id] = account

    def get_balance(self, account_id: str) -> Decimal:
        """Return the exact `Decimal` balance of `account_id`."""
        with self._lock:
            if account_id not in self._accounts:
                raise KeyError(f"Unknown ledger account '{account_id}'.")
            return self._accounts[account_id].balance

    def commit(self, transaction: ImmutableTransaction) -> OcelEventRecord:
        """Validate $dr - cr = 0$ and $h_a = h_o$, commit `transaction`, and encode to OCEL 2.0.

        Raises:
            TypeError: If `transaction` is not an `ImmutableTransaction`.
            LedgerImmutabilityError: If `transaction_id` was already committed.
            BalanceError: If $dr - cr \\neq 0$ or tuple-relational equilibrium $h_a \\neq h_o$ fails.
        """
        ensure_corporate_decimal_context()
        if not isinstance(transaction, ImmutableTransaction):
            raise TypeError(
                f"ImmutableLedger.commit requires an ImmutableTransaction, got {type(transaction).__name__}."
            )
        with self._lock:
            if transaction.transaction_id in self._tx_index:
                raise LedgerImmutabilityError(
                    f"Transaction '{transaction.transaction_id}' is already committed; "
                    "ledger entries are immutable and cannot be overwritten."
                )

            # Resolve account categories and auto-register any new accounts with explicit Entry.category
            cat_map: Dict[str, AccountCategory] = {
                aid: acc.category for aid, acc in self._accounts.items()
            }
            for entry in transaction.entries:
                if entry.account not in cat_map:
                    if entry.category is None:
                        raise BalanceError(
                            f"Account '{entry.account}' is not registered in ImmutableLedger and "
                            "Entry.category was not specified."
                        )
                    cat_map[entry.account] = entry.category

            # Scientific/mathematical requirement (E6 Req 8): Enforce h_a == h_o prior to mutating state
            h_a, h_o = transaction.compute_tuple_equilibrium(cat_map)
            if h_a != h_o:
                raise BalanceError(
                    f"Tuple-relational equilibrium violated for '{transaction.transaction_id}': "
                    f"h_a ({h_a}) != h_o ({h_o}).",
                    details={"h_a": str(h_a), "h_o": str(h_o)},
                )

            # Stage account balance updates
            staged_accounts = dict(self._accounts)
            for entry in transaction.entries:
                if entry.account not in staged_accounts:
                    staged_accounts[entry.account] = LedgerAccount(
                        account_id=entry.account,
                        name=entry.account,
                        category=cat_map[entry.account],
                        balance=Decimal("0"),
                    )
                acc = staged_accounts[entry.account]
                if entry.direction == EntryDirection.DEBIT:
                    staged_accounts[entry.account] = acc.apply_debit_credit(
                        debit=entry.magnitude, credit=Decimal("0")
                    )
                else:
                    staged_accounts[entry.account] = acc.apply_debit_credit(
                        debit=Decimal("0"), credit=entry.magnitude
                    )

            # Verify post-transaction global balance-sheet conservation
            new_assets = sum(
                (
                    a.balance
                    for a in staged_accounts.values()
                    if a.category == AccountCategory.ASSET
                ),
                start=Decimal("0"),
            )
            new_claims = sum(
                (
                    a.balance
                    for a in staged_accounts.values()
                    if a.category != AccountCategory.ASSET
                ),
                start=Decimal("0"),
            )
            if (new_assets - self.total_assets) != h_a or (
                new_claims - self.ownership_claims
            ) != h_o:
                raise BalanceError("Internal conservation mismatch during transaction application.")
            if new_assets != new_claims:
                raise BalanceError(
                    f"Post-transaction balance sheet out of equilibrium: Assets ({new_assets}) != Claims ({new_claims})."
                )

            ocel_record = transaction.to_ocel_event()
            if self._ocel_encoder is not None:
                self._ocel_encoder.encode_event(ocel_record)

            self._accounts = staged_accounts
            self._transactions.append(transaction)
            self._tx_index[transaction.transaction_id] = transaction
            return ocel_record

    def reverse_transaction(
        self,
        original_transaction_id: str,
        reversal_transaction_id: str,
        *,
        timestamp: Union[str, int, Decimal] = "2026-11-17T00:00:01Z",
        reason: str = "Compensating error-correction reversal",
    ) -> ImmutableTransaction:
        """Correct an erroneous committed transaction exclusively via a compensating reversal entry (E6 Req 11)."""
        with self._lock:
            if original_transaction_id not in self._tx_index:
                raise KeyError(f"Cannot reverse unknown transaction '{original_transaction_id}'.")
            original_tx = self._tx_index[original_transaction_id]
            reversal_tx = original_tx.create_reversal(
                reversal_transaction_id,
                timestamp=timestamp,
                reason=reason,
            )
            self.commit(reversal_tx)
            return reversal_tx

    def update_transaction(self, *args: Any, **kwargs: Any) -> None:
        """Operational Safeguard (E6 Req 11): UPDATE operations on ledger state are strictly prohibited."""
        _ = (args, kwargs)
        raise LedgerImmutabilityError(
            "UPDATE operations are strictly prohibited on ImmutableLedger. "
            "Mandated policy: submit a compensating reversal entry via reverse_transaction()."
        )

    def delete_transaction(self, *args: Any, **kwargs: Any) -> None:
        """Operational Safeguard (E6 Req 11): DELETE operations on ledger state are strictly prohibited."""
        _ = (args, kwargs)
        raise LedgerImmutabilityError(
            "DELETE operations are strictly prohibited on ImmutableLedger. "
            "Mandated policy: submit a compensating reversal entry via reverse_transaction()."
        )

    def __setitem__(self, key: Any, value: Any) -> None:
        _ = (key, value)
        raise LedgerImmutabilityError(
            "In-place mutation (__setitem__) is prohibited on ImmutableLedger."
        )

    def __delitem__(self, key: Any) -> None:
        _ = key
        raise LedgerImmutabilityError("Deletion (__delitem__) is prohibited on ImmutableLedger.")


@deprecated(
    since="1.5.0",
    removed_in="2.0.0",
    alternative="Probabilistic financial rewards are deprecated; use ImmutableLedger and deterministic double-entry conservation.",
)
def compute_probabilistic_financial_reward(*args: Any, **kwargs: Any) -> Decimal:
    """Deprecated probabilistic financial reward function (E6 Requirement 13).

    Financial simulations require exact double-entry conservation devoid of probabilistic
    reward approximations.
    """
    _ = (args, kwargs)
    return ZERO_DECIMAL
