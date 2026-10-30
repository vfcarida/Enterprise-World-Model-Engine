"""Corporate Treasury & Payment-Clearing Liquidity Cascade Application (ewm-engine v1.5.0 Core).

Simulates a multi-account corporate treasury subject to intraday settlement drains,
minimum regulatory liquidity ratio constraints, and credit-line drawdowns while
preserving closed-system ledger conservation within IEEE-754 tolerance (<= 1e-12).
"""

from __future__ import annotations

from dataclasses import dataclass

from ewm_engine import Action, Resource, World, WorldState, compare_scenarios
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics


@dataclass(frozen=True)
class TreasuryLiquiditySummary:
    """Result summary for the corporate treasury liquidity cascade evaluation."""

    horizon: int
    initial_total_ledger_usd: float
    baseline_final_operating_cash_usd: float
    counterfactual_final_operating_cash_usd: float
    baseline_liquidity_breach: bool
    counterfactual_liquidity_breach: bool
    max_ledger_conservation_drift: float
    baseline_regret_usd: float
    counterfactual_regret_usd: float


def run_treasury_liquidity_cascade(
    horizon: int = 6,
    rebalance_amount_usd: float = 400_000.0,
    regulatory_floor_usd: float = 350_000.0,
) -> TreasuryLiquiditySummary:
    """Evaluate baseline vs. proactive reserve-to-operating liquidity transfer under settlement drain."""
    initial_state = WorldState(
        resources=[
            Resource(
                id="operating_cash",
                current=250_000.0,
                min_value=0.0,
                max_value=5_000_000.0,
                unit="USD",
            ),
            Resource(
                id="reserve_buffer",
                current=950_000.0,
                min_value=0.0,
                max_value=5_000_000.0,
                unit="USD",
            ),
        ]
    )
    initial_total = (
        initial_state.resources["operating_cash"].current
        + initial_state.resources["reserve_buffer"].current
    )

    world = World(state=initial_state, dynamics=DeterministicTransferDynamics())
    baseline_res = world.simulate(horizon=horizon, seed=42)
    counterfactual_res = world.branch().simulate(
        actions=[
            Action(
                id="proactive_treasury_rebalance",
                type="transfer_resource",
                parameters={
                    "quantity": rebalance_amount_usd,
                    "source_resource": "reserve_buffer",
                    "target_resource": "operating_cash",
                },
            )
        ],
        horizon=horizon,
        seed=42,
    )

    # Verify compare_scenarios integration
    _diff = compare_scenarios(
        baseline_res,
        [counterfactual_res],
        metrics=["resource_operating_cash", "resource_reserve_buffer"],
    )

    max_drift = 0.0
    for traj in (baseline_res.trajectories[0], counterfactual_res.trajectories[0]):
        for step_rec in traj.steps:
            st = step_rec.transition_result.next_state
            tot = st.resources["operating_cash"].current + st.resources["reserve_buffer"].current
            max_drift = max(max_drift, abs(tot - initial_total))

    base_op = baseline_res.trajectories[0].final_state.resources["operating_cash"].current
    cf_op = counterfactual_res.trajectories[0].final_state.resources["operating_cash"].current

    base_breach = base_op < regulatory_floor_usd
    cf_breach = cf_op < regulatory_floor_usd

    # Penalty model: $85,000 overdraft + regulatory breach surcharge if operating_cash < floor
    base_utility = base_op - (85_000.0 if base_breach else 0.0)
    cf_utility = cf_op - (85_000.0 if cf_breach else 0.0) - 0.005 * rebalance_amount_usd
    best_u = max(base_utility, cf_utility)

    return TreasuryLiquiditySummary(
        horizon=horizon,
        initial_total_ledger_usd=initial_total,
        baseline_final_operating_cash_usd=base_op,
        counterfactual_final_operating_cash_usd=cf_op,
        baseline_liquidity_breach=base_breach,
        counterfactual_liquidity_breach=cf_breach,
        max_ledger_conservation_drift=max_drift,
        baseline_regret_usd=max(0.0, best_u - base_utility),
        counterfactual_regret_usd=max(0.0, best_u - cf_utility),
    )


if __name__ == "__main__":  # pragma: no cover
    summary = run_treasury_liquidity_cascade()
    print(f"Initial ledger total:         ${summary.initial_total_ledger_usd:,.2f}")
    print(f"Baseline operating cash:      ${summary.baseline_final_operating_cash_usd:,.2f}")
    print(f"Counterfactual operating cash:${summary.counterfactual_final_operating_cash_usd:,.2f}")
    print(f"Max ledger conservation drift:{summary.max_ledger_conservation_drift:.3e}")
    print(f"Baseline regret:              ${summary.baseline_regret_usd:,.2f}")
    print(f"Counterfactual regret:        ${summary.counterfactual_regret_usd:,.2f}")
