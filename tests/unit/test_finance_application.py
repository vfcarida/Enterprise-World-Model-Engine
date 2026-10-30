"""Unit tests for the Corporate Treasury & Liquidity Cascade application."""

from __future__ import annotations

from examples.finance import run_treasury_liquidity_cascade


def test_treasury_liquidity_cascade_conservation_and_regret() -> None:
    res = run_treasury_liquidity_cascade(
        horizon=6,
        rebalance_amount_usd=400_000.0,
        regulatory_floor_usd=350_000.0,
    )
    assert res.initial_total_ledger_usd == 1_200_000.0
    assert res.baseline_final_operating_cash_usd == 250_000.0
    assert res.counterfactual_final_operating_cash_usd == 650_000.0
    assert res.baseline_liquidity_breach is True
    assert res.counterfactual_liquidity_breach is False
    assert res.max_ledger_conservation_drift <= 1e-12
    assert res.baseline_regret_usd > 0.0
    assert res.counterfactual_regret_usd == 0.0
