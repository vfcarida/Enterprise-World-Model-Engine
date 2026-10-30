"""Unit and property tests for Market-1T, SEC EDGAR, and BitemporalSynchronizer pipelines."""
# ruff: noqa: E402

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

pd = pytest.importorskip("pandas")

from ewm_engine.financial.data import (
    FUNDAMENTAL_METRICS,
    BitemporalSynchronizer,
    EDGARLoader,
    Market1TLoader,
)


def _write_sample_market1t_parquet(path: Path, date_dt: datetime) -> None:
    """Create a realistic Market-1T Parquet file with irregular sub-minute ticks."""
    rows = [
        {
            "timestamp": date_dt.replace(hour=9, minute=30, second=0),
            "ticker": "NVDA",
            "best_bid": 120.0,
            "best_ask": 120.12,
            "bid_size": 400.0,
            "ask_size": 350.0,
            "last_price": 120.06,
            "volume": 1200.0,
            "vwap": 120.05,
        },
        # Gap at second=1 (to test 1Hz forward-fill)
        {
            "timestamp": date_dt.replace(hour=9, minute=30, second=2),
            "ticker": "NVDA",
            "best_bid": 120.20,
            "best_ask": 120.32,
            "bid_size": 500.0,
            "ask_size": 450.0,
            "last_price": 120.26,
            "volume": 1800.0,
            "vwap": 120.18,
        },
        {
            "timestamp": date_dt.replace(hour=9, minute=30, second=0),
            "ticker": "AAPL",
            "best_bid": 180.0,
            "best_ask": 180.09,
            "bid_size": 600.0,
            "ask_size": 600.0,
            "last_price": 180.05,
            "volume": 2000.0,
            "vwap": 180.04,
        },
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


@pytest.mark.unit
def test_market1t_loader_and_1hz_synchronization(tmp_path: Path) -> None:
    """Verify Market-1T Parquet ingestion, spread_bps calculation, caching, and 1Hz resampling."""
    d1 = datetime(2026, 6, 1)
    d2 = datetime(2026, 6, 2)  # Missing file day
    parquet_path = tmp_path / "market1t_20260601.parquet"
    _write_sample_market1t_parquet(parquet_path, d1)

    loader = Market1TLoader(data_dir=tmp_path)
    df = loader.load_date_range(d1, d2, tickers=["NVDA", "AAPL"])
    assert not df.empty
    assert set(df["ticker"].unique()) == {"NVDA", "AAPL"}
    assert "spread_bps" in df.columns
    assert "event_time" in df.columns
    assert "knowledge_time" in df.columns

    # Verify spread_bps on NVDA first row: (120.12 - 120.0) / 120.0 * 10000 = 10.0 bps
    nvda_first = df[df["ticker"] == "NVDA"].iloc[0]
    assert nvda_first["spread_bps"] == pytest.approx(10.0, rel=1e-4)

    # Cached load returns identical frame
    df_cached = loader.load_date_range(d1, d1, tickers=["NVDA", "AAPL"])
    assert len(df_cached) == len(df)

    # Synchronize to 1Hz grid fills the missing second=1 for NVDA via causal forward-fill
    grid_df = loader.synchronize_to_1hz_grid(df)
    nvda_grid = grid_df[grid_df["ticker"] == "NVDA"].sort_values("timestamp")
    assert len(nvda_grid) == 3  # seconds 0, 1, 2
    sec1_row = nvda_grid.iloc[1]
    assert sec1_row["last_price"] == pytest.approx(120.06)
    assert sec1_row["volume"] == pytest.approx(0.0)

    # Empty date range returns empty DataFrame
    empty_df = loader.load_date_range(datetime(2025, 1, 1), datetime(2025, 1, 1), ["NVDA"])
    assert empty_df.empty
    assert loader.synchronize_to_1hz_grid(empty_df).empty


@pytest.mark.unit
def test_edgar_loader_and_bitemporal_fundamentals(tmp_path: Path) -> None:
    """Verify EDGARLoader downloads and parses 6 fundamental metrics without future leakage."""
    loader = EDGARLoader(download_dir=tmp_path)
    loader.download_filings("NVDA", form_type="10-Q", num_filings=2)

    # Missing CSV raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Fundamentals CSV not found"):
        loader.parse_filing_to_fundamentals("UNKNOWN", datetime(2026, 6, 1))

    csv_path = tmp_path / "NVDA_fundamentals.csv"
    pd.DataFrame(
        [
            {
                "event_time": "2026-01-31",
                "filing_date": "2026-02-25",
                "knowledge_time": "2026-02-25",
                "revenue": 22100.0,
                "net_income": 12285.0,
                "total_assets": 65728.0,
                "total_liabilities": 22750.0,
                "free_cash_flow": 11200.0,
                "debt_to_equity": 0.22,
            },
            {
                "event_time": "2026-04-30",
                "filing_date": "2026-05-28",
                "knowledge_time": "2026-05-28",
                "revenue": 26044.0,
                "net_income": 14881.0,
                "total_assets": 77072.0,
                "total_liabilities": 27930.0,
                "free_cash_flow": 14900.0,
                "debt_to_equity": 0.19,
            },
        ]
    ).to_csv(csv_path, index=False)

    # Query on May 10, 2026 (after Q2 event_time Apr 30, but BEFORE Q2 filing_date May 28!)
    # Must return Q1 fundamentals (revenue=22100.0), NOT Q2!
    f_may10 = loader.parse_filing_to_fundamentals("NVDA", datetime(2026, 5, 10))
    assert set(f_may10.keys()) == set(FUNDAMENTAL_METRICS)
    assert f_may10["revenue"] == pytest.approx(22100.0)

    # Query on June 1, 2026 (after Q2 filing_date May 28) -> returns Q2 fundamentals
    f_jun01 = loader.parse_filing_to_fundamentals("NVDA", datetime(2026, 6, 1))
    assert f_jun01["revenue"] == pytest.approx(26044.0)


@pytest.mark.unit
def test_bitemporal_synchronizer_point_in_time_and_dict_merge(tmp_path: Path) -> None:
    """Verify BitemporalSynchronizer merges 1Hz market data and fundamentals without lookahead."""
    market_df = pd.DataFrame(
        [
            {
                "ticker": "NVDA",
                "timestamp": pd.Timestamp("2026-05-15 10:00:00"),
                "knowledge_time": pd.Timestamp("2026-05-15 10:00:00"),
                "best_bid": 115.0,
                "best_ask": 115.1,
                "last_price": 115.05,
            },
            {
                "ticker": "NVDA",
                "timestamp": pd.Timestamp("2026-06-01 10:00:00"),
                "knowledge_time": pd.Timestamp("2026-06-01 10:00:00"),
                "best_bid": 125.0,
                "best_ask": 125.1,
                "last_price": 125.05,
            },
        ]
    )

    fundamentals_df = pd.DataFrame(
        [
            {
                "ticker": "NVDA",
                "event_time": pd.Timestamp("2026-01-31"),
                "filing_date": pd.Timestamp("2026-02-25"),
                "knowledge_time": pd.Timestamp("2026-02-25"),
                "revenue": 22100.0,
                "net_income": 12285.0,
                "total_assets": 65728.0,
                "total_liabilities": 22750.0,
                "free_cash_flow": 11200.0,
                "debt_to_equity": 0.22,
            },
            {
                "ticker": "NVDA",
                "event_time": pd.Timestamp("2026-04-30"),
                "filing_date": pd.Timestamp("2026-05-28"),
                "knowledge_time": pd.Timestamp("2026-05-28"),
                "revenue": 26044.0,
                "net_income": 14881.0,
                "total_assets": 77072.0,
                "total_liabilities": 27930.0,
                "free_cash_flow": 14900.0,
                "debt_to_equity": 0.19,
            },
        ]
    )

    synced = BitemporalSynchronizer.synchronize_point_in_time(market_df, fundamentals_df)
    assert BitemporalSynchronizer.verify_no_future_leakage(synced) is True
    assert synced.iloc[0]["revenue"] == pytest.approx(22100.0)
    assert synced.iloc[1]["revenue"] == pytest.approx(26044.0)

    # Also test dictionary merge API
    dict_merged = BitemporalSynchronizer.merge_market_and_fundamentals(
        market_df,
        {
            "NVDA": {
                "revenue": 26044.0,
                "net_income": 14881.0,
                "total_assets": 77072.0,
                "total_liabilities": 27930.0,
                "free_cash_flow": 14900.0,
                "debt_to_equity": 0.19,
            }
        },
    )
    assert all(col in dict_merged.columns for col in FUNDAMENTAL_METRICS)
    assert dict_merged.iloc[0]["revenue"] == pytest.approx(26044.0)

    # Edge cases: empty market_df, empty fundamentals_df, missing knowledge_time columns
    empty_market = BitemporalSynchronizer.synchronize_point_in_time(pd.DataFrame(), fundamentals_df)
    assert empty_market.empty
    empty_fund = BitemporalSynchronizer.synchronize_point_in_time(
        market_df.drop(columns=["knowledge_time"]), pd.DataFrame()
    )
    assert len(empty_fund) == 2
    assert empty_fund.iloc[0]["revenue"] == pytest.approx(0.0)
    assert BitemporalSynchronizer.verify_no_future_leakage(empty_fund) is True

    no_kt_synced = BitemporalSynchronizer.synchronize_point_in_time(
        market_df.drop(columns=["knowledge_time"]),
        fundamentals_df.drop(columns=["knowledge_time", "free_cash_flow"]),
    )
    assert no_kt_synced.iloc[0]["free_cash_flow"] == pytest.approx(0.0)


@pytest.mark.property
@given(
    lag_days=st.integers(min_value=1, max_value=60),
    query_offset_days=st.integers(min_value=-30, max_value=90),
)
def test_property_bitemporal_synchronization_never_leaks_future_filings(
    lag_days: int, query_offset_days: int
) -> None:
    """Property: Point-in-time synchronization never attaches a filing before its knowledge_time."""
    event_dt = pd.Timestamp("2026-03-31")
    knowledge_dt = event_dt + timedelta(days=lag_days)
    market_dt = event_dt + timedelta(days=query_offset_days)

    market_df = pd.DataFrame(
        [{"ticker": "AAPL", "timestamp": market_dt, "knowledge_time": market_dt}]
    )
    fund_df = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "event_time": event_dt,
                "filing_date": knowledge_dt,
                "knowledge_time": knowledge_dt,
                "revenue": 95000.0,
                "net_income": 24000.0,
                "total_assets": 350000.0,
                "total_liabilities": 280000.0,
                "free_cash_flow": 27000.0,
                "debt_to_equity": 1.45,
            }
        ]
    )

    synced = BitemporalSynchronizer.synchronize_point_in_time(market_df, fund_df)
    assert BitemporalSynchronizer.verify_no_future_leakage(synced) is True
    if market_dt < knowledge_dt:
        assert synced.iloc[0]["revenue"] == pytest.approx(0.0)
    else:
        assert synced.iloc[0]["revenue"] == pytest.approx(95000.0)
