"""Corporate Treasury Double-Entry Engine, Liquidity Drain Stress Test, and 1,000-Step Benchmark (`EWM-004`).

Wires `FinancialState` and `DoubleEntryConstraint` into the `EWM-002` `HybridTransitionRouter`
and `TransitionEngine`, enabling causal `engine.branch().do(...)` stress testing of sudden
liquidity drains with strictly `Decimal('0.00')` balance-sheet discrepancy.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.core.actions import Action
from ewm_engine.core.context import ContextSnapshot
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.scm import CausalGraph
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import RandomGenerator
from ewm_engine.domains.finance.constraints import (
    DoubleEntryConstraint,
    parse_journal_entry_from_action,
)
from ewm_engine.domains.finance.state import (
    ZERO_DECIMAL,
    AccountCategory,
    FinancialState,
    JournalEntry,
    JournalLine,
    LedgerAccount,
    build_journal_action,
    ensure_corporate_decimal_context,
    to_exact_decimal,
)
from ewm_engine.dynamics.base import (
    InfeasibleResidualPolicy,
    MechanismKind,
    MechanismProvenance,
    TransitionResult,
)
from ewm_engine.dynamics.router import HybridTransitionRouter, TransitionEngine
from ewm_engine.provenance.evidence import EvidenceLevel


def create_default_treasury_state(
    *,
    cash: Decimal | str = Decimal("10000000.00"),
    treasury_bills: Decimal | str = Decimal("5000000.00"),
    loans_receivable: Decimal | str = Decimal("25000000.00"),
    demand_deposits: Decimal | str = Decimal("22000000.00"),
    long_term_debt: Decimal | str = Decimal("10000000.00"),
    common_equity: Decimal | str = Decimal("5000000.00"),
    retained_earnings: Decimal | str = Decimal("3000000.00"),
    min_total_liquidity: Decimal | str = Decimal("2000000.00"),
    min_cash_balance: Decimal | str = Decimal("500000.00"),
) -> FinancialState:
    """Create a balanced corporate treasury `FinancialState` in exact `Decimal`.

    Default Balance Sheet:
    - Assets = Cash ($10M) + Treasury Bills ($5M) + Loans Receivable ($25M) = $40,000,000.00
    - Liabilities = Demand Deposits ($22M) + Long-Term Debt ($10M) = $32,000,000.00
    - Equity = Common Equity ($5M) + Retained Earnings ($3M) = $8,000,000.00
    - Identity: $40,000,000.00 == $32,000,000.00 + $8,000,000.00 (Discrepancy = 0.00)
    """
    ensure_corporate_decimal_context()
    fin_state = FinancialState(
        step=0,
        timestamp=ZERO_DECIMAL,
        min_total_liquidity=to_exact_decimal(min_total_liquidity, field_name="min_total_liquidity"),
        accounts={
            "cash": LedgerAccount(
                account_id="cash",
                name="Cash and Central Bank Reserves",
                category=AccountCategory.ASSET,
                balance=to_exact_decimal(cash, field_name="cash"),
                min_balance=to_exact_decimal(min_cash_balance, field_name="min_cash_balance"),
                is_liquid=True,
            ),
            "treasury_bills": LedgerAccount(
                account_id="treasury_bills",
                name="High-Quality Liquid Sovereign Treasury Bills",
                category=AccountCategory.ASSET,
                balance=to_exact_decimal(treasury_bills, field_name="treasury_bills"),
                min_balance=ZERO_DECIMAL,
                is_liquid=True,
            ),
            "loans_receivable": LedgerAccount(
                account_id="loans_receivable",
                name="Corporate Loans Receivable",
                category=AccountCategory.ASSET,
                balance=to_exact_decimal(loans_receivable, field_name="loans_receivable"),
                min_balance=ZERO_DECIMAL,
                is_liquid=False,
            ),
            "demand_deposits": LedgerAccount(
                account_id="demand_deposits",
                name="Institutional & Retail Demand Deposits",
                category=AccountCategory.LIABILITY,
                balance=to_exact_decimal(demand_deposits, field_name="demand_deposits"),
                min_balance=ZERO_DECIMAL,
                is_liquid=False,
            ),
            "long_term_debt": LedgerAccount(
                account_id="long_term_debt",
                name="Senior Notes & Repo Borrowings",
                category=AccountCategory.LIABILITY,
                balance=to_exact_decimal(long_term_debt, field_name="long_term_debt"),
                min_balance=ZERO_DECIMAL,
                is_liquid=False,
            ),
            "common_equity": LedgerAccount(
                account_id="common_equity",
                name="Common Stock & Paid-In Capital",
                category=AccountCategory.EQUITY,
                balance=to_exact_decimal(common_equity, field_name="common_equity"),
                min_balance=ZERO_DECIMAL,
                is_liquid=False,
            ),
            "retained_earnings": LedgerAccount(
                account_id="retained_earnings",
                name="Retained Earnings",
                category=AccountCategory.EQUITY,
                balance=to_exact_decimal(retained_earnings, field_name="retained_earnings"),
                min_balance=None,
                is_liquid=False,
            ),
        },
    )
    if not fin_state.is_balanced:
        raise ValueError(
            f"Initial FinancialState violates accounting identity: discrepancy={fin_state.accounting_discrepancy}"
        )
    return fin_state


class DoubleEntryLedgerDynamics:
    """Deterministic `DynamicsModel` executing exact `Decimal` double-entry transitions.

    Processes both explicit journal `Action`s and causal `do()` driver interventions
    (`deposit_outflow_amount`, `emergency_repo_draw`, `market_stress`) while preserving
    $\\sum \\text{Assets} = \\sum \\text{Liabilities} + \\sum \\text{Equity}$ to `Decimal('0.00')`.
    """

    def __init__(
        self,
        *,
        name: str = "DoubleEntryLedgerDynamics",
        version: str = "1.0.0",
        auto_liquidate_tbills: bool = True,
    ) -> None:
        self.name = name
        self.version = version
        self.auto_liquidate_tbills = bool(auto_liquidate_tbills)
        self._provenance = MechanismProvenance(
            mechanism_id=name,
            version=version,
            kind=MechanismKind.KNOWN_DETERMINISTIC,
            evidence_level=EvidenceLevel.STRUCTURAL,
            source="exact_decimal_double_entry_ledger",
            uncertainty_contract="point_deterministic",
            invariants_affected=("double_entry_accounting_invariant", "temporal_liquidity_floor"),
            failure_semantics=InfeasibleResidualPolicy.REJECT,
        )

    @property
    def mechanism_provenance(self) -> MechanismProvenance:
        return self._provenance

    @staticmethod
    def _extract_driver_decimal(state: WorldState, driver_key: str) -> Decimal:
        """Read a causal driver value as an exact `Decimal` from `state`."""
        ensure_corporate_decimal_context()
        ledger_mem = state.memory.get("financial_ledger", {})
        drivers = ledger_mem.get("causal_drivers", {}) if isinstance(ledger_mem, Mapping) else {}
        default_dec = Decimal(str(drivers.get(driver_key, "0.00")))

        # Check if `do()` or `Intervention` updated the resource or memory node directly
        if driver_key in state.memory:
            return Decimal(str(state.memory[driver_key]))
        res = state.resources.get(driver_key)
        if res is not None:
            # Convert via round(val, 2) string representation to avoid float binary noise from do() resource storage
            res_dec = Decimal(f"{res.current:.2f}")
            if res_dec != default_dec:
                return res_dec
        return default_dec

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
        runtime_context: ContextSnapshot | None = None,
    ) -> TransitionResult:
        """Apply double-entry journal actions and active causal liquidity drivers in exact Decimal."""
        ensure_corporate_decimal_context()
        fin_state = FinancialState.from_world_state(state)

        market_stress = self._extract_driver_decimal(state, "market_stress")
        outflow_amt = self._extract_driver_decimal(state, "deposit_outflow_amount")
        repo_draw = self._extract_driver_decimal(state, "emergency_repo_draw")

        # If market_stress > 0 and deposit_outflow_amount was not explicitly forced, derive outflow
        if market_stress > ZERO_DECIMAL and outflow_amt == ZERO_DECIMAL:
            outflow_amt = (market_stress * Decimal("500000.00")).quantize(Decimal("0.01"))

        # 1. Apply any causal deposit outflow as a balanced double-entry journal entry:
        #    Debit demand_deposits (Liability decreases), Credit cash (Asset decreases)
        if outflow_amt > ZERO_DECIMAL:
            cash_bal = fin_state.get_balance("cash")
            min_cash = fin_state.accounts["cash"].min_balance or ZERO_DECIMAL
            tbill_bal = fin_state.get_balance("treasury_bills")

            # If cash would drop below min_cash + buffer and T-bills are available, liquidate T-bills first:
            # Debit cash (+Asset), Credit treasury_bills (-Asset)
            projected_cash = cash_bal - outflow_amt
            if (
                self.auto_liquidate_tbills
                and projected_cash < min_cash
                and tbill_bal > ZERO_DECIMAL
            ):
                needed = min(tbill_bal, (min_cash - projected_cash) + Decimal("250000.00"))
                if needed > ZERO_DECIMAL:
                    tbill_entry = JournalEntry(
                        entry_id=f"auto_tbill_liquidation_step_{state.step}",
                        step=state.step,
                        description="Automatic T-bill collateral liquidation into cash",
                        lines=(
                            JournalLine(account_id="cash", debit=needed, credit=ZERO_DECIMAL),
                            JournalLine(
                                account_id="treasury_bills", debit=ZERO_DECIMAL, credit=needed
                            ),
                        ),
                    )
                    fin_state = fin_state.apply_journal_entry(tbill_entry)

            outflow_entry = JournalEntry(
                entry_id=f"deposit_outflow_step_{state.step}",
                step=state.step,
                description="Causal deposit outflow settlement",
                lines=(
                    JournalLine(
                        account_id="demand_deposits",
                        debit=outflow_amt,
                        credit=ZERO_DECIMAL,
                    ),
                    JournalLine(
                        account_id="cash",
                        debit=ZERO_DECIMAL,
                        credit=outflow_amt,
                    ),
                ),
            )
            fin_state = fin_state.apply_journal_entry(outflow_entry)

        # 2. Apply any causal emergency repo facility draw as a balanced double-entry journal entry:
        #    Debit cash (+Asset), Credit long_term_debt (+Liability)
        if repo_draw > ZERO_DECIMAL:
            repo_entry = JournalEntry(
                entry_id=f"emergency_repo_draw_step_{state.step}",
                step=state.step,
                description="Causal central bank / repo liquidity injection",
                lines=(
                    JournalLine(account_id="cash", debit=repo_draw, credit=ZERO_DECIMAL),
                    JournalLine(account_id="long_term_debt", debit=ZERO_DECIMAL, credit=repo_draw),
                ),
            )
            fin_state = fin_state.apply_journal_entry(repo_entry)

        # 3. Apply all explicit enterprise double-entry actions accepted by Pre-rules
        for act in actions:
            entry = parse_journal_entry_from_action(act, step=state.step)
            if entry.lines:
                fin_state = fin_state.apply_journal_entry(entry)

        next_world = fin_state.to_world_state(
            causal_drivers={
                "market_stress": market_stress,
                "deposit_outflow_amount": outflow_amt,
                "emergency_repo_draw": repo_draw,
            }
        )
        # Preserve step on `next_world` because `HybridTransitionRouter.step` advances time by +1
        next_world = next_world.model_copy(
            update={"step": state.step, "timestamp": state.timestamp}
        )

        return TransitionResult(
            next_state=next_world,
            applied_changes={
                "total_assets": str(fin_state.total_assets),
                "total_liabilities": str(fin_state.total_liabilities),
                "total_equity": str(fin_state.total_equity),
                "total_liquidity": str(fin_state.total_liquidity),
                "accounting_discrepancy": str(fin_state.accounting_discrepancy),
            },
            evidence_level=EvidenceLevel.STRUCTURAL,
            diagnostics={
                "is_balanced": fin_state.is_balanced,
                "accounting_discrepancy": str(fin_state.accounting_discrepancy),
            },
            model_name=self.name,
        )


def build_corporate_treasury_engine(
    initial_financial_state: FinancialState | None = None,
    *,
    enforce_liquidity_floor: bool = True,
    raise_on_pre_rule_violation: bool = False,
) -> tuple[TransitionEngine, FinancialState]:
    """Build a `TransitionEngine` wired with `DoubleEntryConstraint` and `DoubleEntryLedgerDynamics`."""
    ensure_corporate_decimal_context()
    fin_state = (
        initial_financial_state
        if initial_financial_state is not None
        else create_default_treasury_state()
    )
    init_world = fin_state.to_world_state(
        causal_drivers={
            "market_stress": Decimal("0.00"),
            "deposit_outflow_amount": Decimal("0.00"),
            "emergency_repo_draw": Decimal("0.00"),
        }
    )

    double_entry_constraint = DoubleEntryConstraint(
        enforce_liquidity_floor=enforce_liquidity_floor,
    )
    reg = ConstraintRegistry([double_entry_constraint])

    treasury_graph = CausalGraph(
        nodes={
            "market_stress",
            "deposit_outflow_amount",
            "emergency_repo_draw",
            "cash",
            "treasury_bills",
            "demand_deposits",
            "long_term_debt",
        },
        edges={
            ("market_stress", "deposit_outflow_amount"),
            ("deposit_outflow_amount", "cash"),
            ("deposit_outflow_amount", "demand_deposits"),
            ("treasury_bills", "cash"),
            ("emergency_repo_draw", "cash"),
            ("emergency_repo_draw", "long_term_debt"),
        },
    )

    router = HybridTransitionRouter(
        structural_model=DoubleEntryLedgerDynamics(),
        constraints=reg,
        pre_rules=(double_entry_constraint.as_pre_rule,),
        post_rules=(double_entry_constraint.as_post_rule,),
        causal_graph=treasury_graph,
        raise_on_pre_rule_violation=raise_on_pre_rule_violation,
        name="FinanceDoubleEntryRouter",
    )
    engine = TransitionEngine(
        router=router,
        initial_state=init_world,
        branch_id="treasury_main",
    )
    return engine, fin_state


class LiquidityStressTestReport(BaseModel):
    """Summary of the counterfactual liquidity drain stress-test scenario (`EWM-004`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    steps: int
    baseline_final_cash: Decimal
    baseline_final_liquidity: Decimal
    stress_final_cash: Decimal
    stress_final_tbills: Decimal
    stress_final_liquidity: Decimal
    recovery_final_cash: Decimal
    recovery_final_liquidity: Decimal
    max_abs_discrepancy_across_all_branches: Decimal
    all_steps_balanced: bool
    severed_edges_on_stress_branch: tuple[tuple[str, str], ...]


def run_liquidity_drain_stress_test(steps: int = 5) -> LiquidityStressTestReport:
    """Execute the end-to-end corporate liquidity drain stress test using `engine.branch().do()`.

    Compares three isolated causal branches over `steps`:
    1. `baseline`: Normal fee/interest income (`Debit cash $50,000.00`, `Credit retained_earnings $50,000.00`).
    2. `stress_branch`: Uses `engine.branch().do({"deposit_outflow_amount": 2500000})` to simulate a
       sudden $2.5M/step institutional deposit run, triggering automatic T-bill collateral liquidation
       while severing inbound edges to `deposit_outflow_amount`.
    3. `recovery_branch`: Branches from `stress_branch` and applies
       `do({"deposit_outflow_amount": 2500000, "emergency_repo_draw": 1500000})` to inject repo liquidity.
    """
    ensure_corporate_decimal_context()
    base_engine, _ = build_corporate_treasury_state_and_engine()

    # Spawn isolated counterfactual branches using the EWM-002 `branch().do()` causal API
    stress_branch = base_engine.branch("liquidity_drain_shock").do(
        {"deposit_outflow_amount": 2500000}
    )
    recovery_branch = base_engine.branch("repo_facility_recovery").do(
        {"deposit_outflow_amount": 2500000, "emergency_repo_draw": 1500000}
    )

    severed = stress_branch.causal_execution_log[-1].severed_inbound_edges
    max_disc = ZERO_DECIMAL
    all_balanced = True

    for step_idx in range(steps):
        fee_action = build_journal_action(
            f"fee_income_{step_idx}",
            debits={"cash": Decimal("50000.00")},
            credits={"retained_earnings": Decimal("50000.00")},
            description="Recurring corporate treasury fee settlement",
        )
        out_base = base_engine.step(action=fee_action)
        out_stress = stress_branch.step(action=fee_action)
        out_rec = recovery_branch.step(action=fee_action)

        for outcome in (out_base, out_stress, out_rec):
            fin = FinancialState.from_world_state(outcome.next_state)
            disc = abs(fin.accounting_discrepancy)
            if disc > max_disc:
                max_disc = disc
            if not fin.is_balanced or outcome.pre_rule_rejected or outcome.post_rule_rejected:
                all_balanced = False

    assert base_engine.current_state is not None
    assert stress_branch.current_state is not None
    assert recovery_branch.current_state is not None

    fin_base = FinancialState.from_world_state(base_engine.current_state)
    fin_stress = FinancialState.from_world_state(stress_branch.current_state)
    fin_rec = FinancialState.from_world_state(recovery_branch.current_state)

    return LiquidityStressTestReport(
        steps=steps,
        baseline_final_cash=fin_base.get_balance("cash"),
        baseline_final_liquidity=fin_base.total_liquidity,
        stress_final_cash=fin_stress.get_balance("cash"),
        stress_final_tbills=fin_stress.get_balance("treasury_bills"),
        stress_final_liquidity=fin_stress.total_liquidity,
        recovery_final_cash=fin_rec.get_balance("cash"),
        recovery_final_liquidity=fin_rec.total_liquidity,
        max_abs_discrepancy_across_all_branches=max_disc,
        all_steps_balanced=all_balanced,
        severed_edges_on_stress_branch=severed,
    )


def build_corporate_treasury_state_and_engine() -> tuple[TransitionEngine, FinancialState]:
    """Helper returning a fresh corporate treasury `(TransitionEngine, FinancialState)` pair."""
    return build_corporate_treasury_engine()


class DecimalVsFloatBenchmarkReport(BaseModel):
    """1,000-step performance and numerical accuracy comparison of `Decimal` vs `float`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    steps: int
    decimal_engine_total_ms: float
    decimal_pure_ledger_ms: float
    float_pure_ledger_ms: float
    decimal_final_discrepancy: Decimal
    float_final_discrepancy_str: str
    decimal_zero_drift: bool


def benchmark_decimal_vs_float_rollout(steps: int = 1000) -> DecimalVsFloatBenchmarkReport:
    """Run a 1,000-step rollout comparing exact `Decimal` ledger execution vs `float` arithmetic.

    Evaluates:
    1. Full `TransitionEngine` + `DoubleEntryConstraint` 1,000-step rollout in exact `Decimal`.
    2. Pure `FinancialState.apply_journal_entry` 1,000-step rollout in exact `Decimal`.
    3. Equivalent 1,000-step IEEE-754 `float` ledger accumulation on large corporate balances
       (`$10,000,000,000,000.00` + repeating cent postings `$0.10`, `$0.20`, `$0.07`).
    """
    ensure_corporate_decimal_context()
    # Initialize with large multi-trillion corporate balance to test precision & float drift
    init_state = create_default_treasury_state(
        cash=Decimal("10000000000000.00"),
        treasury_bills=Decimal("5000000000000.00"),
        loans_receivable=Decimal("25000000000000.00"),
        demand_deposits=Decimal("22000000000000.00"),
        long_term_debt=Decimal("10000000000000.00"),
        common_equity=Decimal("5000000000000.00"),
        retained_earnings=Decimal("3000000000000.00"),
    )
    engine, _ = build_corporate_treasury_engine(init_state)

    cent_cycle = (Decimal("0.10"), Decimal("0.20"), Decimal("0.07"), Decimal("19.99"))

    # 1. Full TransitionEngine + DoubleEntryConstraint 1,000-step rollout
    t0_eng = time.perf_counter()
    for i in range(steps):
        amt = cent_cycle[i % len(cent_cycle)]
        act = build_journal_action(
            f"tx_{i}",
            debits={"cash": amt},
            credits={"retained_earnings": amt},
            description="Sub-cent precision corporate fee posting",
        )
        out = engine.step(action=act)
        if out.pre_rule_rejected or out.post_rule_rejected:
            raise RuntimeError(f"Unexpected rejection at step {i}")
    decimal_engine_ms = (time.perf_counter() - t0_eng) * 1000.0

    assert engine.current_state is not None
    final_fin = FinancialState.from_world_state(engine.current_state)
    dec_discrepancy = final_fin.accounting_discrepancy

    # 2. Pure Decimal FinancialState rollout
    pure_state = init_state
    t0_dec = time.perf_counter()
    for i in range(steps):
        amt = cent_cycle[i % len(cent_cycle)]
        entry = JournalEntry(
            entry_id=f"pure_{i}",
            step=i,
            lines=(
                JournalLine(account_id="cash", debit=amt, credit=ZERO_DECIMAL),
                JournalLine(account_id="retained_earnings", debit=ZERO_DECIMAL, credit=amt),
            ),
        )
        pure_state = pure_state.apply_journal_entry(entry)
    decimal_pure_ms = (time.perf_counter() - t0_dec) * 1000.0

    # 3. Equivalent float accumulation over 1,000 steps
    float_cycle = (0.10, 0.20, 0.07, 19.99)
    f_cash = 10000000000000.0
    f_tbills = 5000000000000.0
    f_loans = 25000000000000.0
    f_dep = 22000000000000.0
    f_debt = 10000000000000.0
    f_eq = 5000000000000.0
    f_ret = 3000000000000.0

    t0_flt = time.perf_counter()
    for i in range(steps):
        f_amt = float_cycle[i % len(float_cycle)]
        f_cash += f_amt
        # Split credit across two equity sub-accounts to expose IEEE-754 rounding asymmetry
        f_ret += f_amt * 0.30
        f_eq += f_amt * 0.70
    float_pure_ms = (time.perf_counter() - t0_flt) * 1000.0

    # Compare float result against exact mathematical sum
    exact_added = sum((cent_cycle[i % len(cent_cycle)] for i in range(steps)), start=ZERO_DECIMAL)
    exact_total_assets = Decimal("40000000000000.00") + exact_added
    float_total_assets = Decimal(f"{f_cash + f_tbills + f_loans:.6f}")
    float_liab_plus_eq = Decimal(f"{(f_dep + f_debt) + (f_eq + f_ret):.6f}")
    float_drift = max(
        abs(float_total_assets - exact_total_assets),
        abs(float_total_assets - float_liab_plus_eq),
    )

    return DecimalVsFloatBenchmarkReport(
        steps=steps,
        decimal_engine_total_ms=decimal_engine_ms,
        decimal_pure_ledger_ms=decimal_pure_ms,
        float_pure_ledger_ms=float_pure_ms,
        decimal_final_discrepancy=dec_discrepancy,
        float_final_discrepancy_str=f"{float_drift:.6f}",
        decimal_zero_drift=(dec_discrepancy == ZERO_DECIMAL and pure_state.is_balanced),
    )
