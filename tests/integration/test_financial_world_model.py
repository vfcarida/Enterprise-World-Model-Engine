"""End-to-end integration and comparative backtesting tests for the Financial World Model."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ewm_engine.financial.data import BitemporalSynchronizer, EDGARLoader, Market1TLoader
from ewm_engine.financial.evaluation import (
    FinancialBacktester,
    build_historical_regime_episode,
    compute_coverage_calibration,
    compute_mae,
    compute_rank_ic,
)


@pytest.mark.integration
def test_end_to_end_fwm_comparative_backtest_across_regimes() -> None:
    """Run full comparative backtest on 2020 crash, 2022 Fed tightening, and bull expansion."""
    backtester = FinancialBacktester(nominal_coverage=0.90)
    results = backtester.run_comparative_backtest(
        regimes=("2020_covid_crash", "2022_fed_tightening", "bull_expansion"),
        seed=42,
    )

    expected_models = {
        "ARIMA",
        "GARCH",
        "XGBoost",
        "XGBoost+Features",
        "Predict-Only JEPA",
        "Full FWM",
    }
    assert set(results.keys()) == expected_models

    for model_name, metrics_list in results.items():
        assert len(metrics_list) == 3, f"{model_name} missing regime evaluations"
        for m in metrics_list:
            assert -1.0 <= m.rank_ic_volatility <= 1.0
            assert -1.0 <= m.rank_ic_spread <= 1.0
            assert m.volatility_mae >= 0.0
            assert m.spread_mae_bps >= 0.0
            assert 0.0 <= m.empirical_coverage <= 1.0

    # Verify Full FWM achieves strong positive Rank IC and lower MAE than statistical baselines on shock regimes
    fwm_crash = results["Full FWM"][0]
    arima_crash = results["ARIMA"][0]
    garch_crash = results["GARCH"][0]

    assert fwm_crash.regime_name == "2020_covid_crash"
    assert fwm_crash.rank_ic_volatility > 0.65
    assert fwm_crash.volatility_mae < arima_crash.volatility_mae
    assert fwm_crash.volatility_mae < garch_crash.volatility_mae
    assert fwm_crash.spread_mae_bps < arima_crash.spread_mae_bps

    fwm_tightening = results["Full FWM"][1]
    arima_tightening = results["ARIMA"][1]
    assert fwm_tightening.regime_name == "2022_fed_tightening"
    assert fwm_tightening.rank_ic_volatility > 0.65
    assert fwm_tightening.volatility_mae < arima_tightening.volatility_mae


@pytest.mark.integration
def test_end_to_end_parquet_and_edgar_to_fwm_inference(tmp_path: Path) -> None:
    """Verify full data ingestion -> bitemporal sync -> FWM counterfactual inference."""
    date_dt = datetime(2026, 6, 1)
    tickers = ["NVDA", "AMD", "TSM", "AAPL", "MSFT", "JPM", "XOM", "WMT"]

    rows = []
    for sec in range(5):
        for idx, t in enumerate(tickers):
            px = 100.0 + idx * 10.0 + sec * 0.1
            rows.append(
                {
                    "timestamp": date_dt.replace(hour=9, minute=30, second=sec),
                    "ticker": t,
                    "best_bid": px - 0.05,
                    "best_ask": px + 0.05,
                    "bid_size": 500.0,
                    "ask_size": 500.0,
                    "last_price": px,
                    "volume": 1000.0,
                    "vwap": px,
                }
            )
    pd.DataFrame(rows).to_parquet(tmp_path / "market1t_20260601.parquet", index=False)

    fundamentals_dict: dict[str, dict[str, float]] = {}
    edgar = EDGARLoader(tmp_path)
    for idx, t in enumerate(tickers):
        pd.DataFrame(
            [
                {
                    "filing_date": "2026-05-01",
                    "revenue": 10000.0 + idx * 1000.0,
                    "net_income": 2500.0,
                    "total_assets": 50000.0,
                    "total_liabilities": 20000.0,
                    "free_cash_flow": 3000.0,
                    "debt_to_equity": 0.4,
                }
            ]
        ).to_csv(tmp_path / f"{t}_fundamentals.csv", index=False)
        fundamentals_dict[t] = edgar.parse_filing_to_fundamentals(t, date_dt)

    m1t = Market1TLoader(tmp_path)
    raw_df = m1t.load_date_range(date_dt, date_dt, tickers)
    sync_1hz = m1t.synchronize_to_1hz_grid(raw_df)
    merged_df = BitemporalSynchronizer.merge_market_and_fundamentals(sync_1hz, fundamentals_dict)
    assert len(merged_df) == 5 * len(tickers)

    # Evaluate metric edge cases
    assert compute_rank_ic(np.ones(4), np.arange(4, dtype=float)) == 0.0
    assert compute_mae(np.array([1.0, 2.0]), np.array([1.5, 2.5])) == pytest.approx(0.5)
    cov, err = compute_coverage_calibration(
        np.array([1.0, 2.0]),
        np.array([0.5, 1.5]),
        np.array([1.5, 2.5]),
        nominal_coverage=0.90,
    )
    assert cov == 1.0
    assert err == pytest.approx(0.10)
    ep = build_historical_regime_episode("bull_expansion")
    assert ep.regime_name == "bull_expansion"
