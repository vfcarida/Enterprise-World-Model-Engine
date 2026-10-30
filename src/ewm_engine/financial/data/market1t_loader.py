"""Market-1T high-frequency (1Hz) limit order book and microstructure data loader.

Ingests partitioned Parquet files, enforces bitemporal timestamps (event_time,
knowledge_time, ingestion_time), computes bid-ask spread in basis points, and
synchronizes asynchronous ticks to a uniform 1Hz grid via causal forward-fill.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger("ewm_engine.financial.data.market1t_loader")

REQUIRED_MARKET_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "ticker",
    "best_bid",
    "best_ask",
    "bid_size",
    "ask_size",
    "last_price",
    "volume",
    "vwap",
)


class Market1TLoader:
    """Ingests Market-1T dataset (Parquet format, partitioned by date)."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.cache: dict[tuple[str, tuple[str, ...]], pd.DataFrame] = {}

    def load_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        tickers: list[str],
    ) -> pd.DataFrame:
        """Load 1Hz market data for specified date range and tickers.

        Args:
            start_date: Inclusive start date.
            end_date: Inclusive end date.
            tickers: List of equity ticker symbols to filter.

        Returns:
            DataFrame with columns: timestamp, ticker, best_bid, best_ask, bid_size,
            ask_size, last_price, volume, vwap, spread_bps, event_time, knowledge_time.
        """
        dfs: list[pd.DataFrame] = []
        current_date = start_date
        ticker_key = tuple(sorted(tickers))

        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            cache_key = (date_str, ticker_key)

            if cache_key in self.cache:
                dfs.append(self.cache[cache_key].copy())
                current_date += timedelta(days=1)
                continue

            file_path = self.data_dir / f"market1t_{date_str}.parquet"
            if not file_path.exists():
                logger.warning("Missing Market-1T data for %s at %s", date_str, file_path)
                current_date += timedelta(days=1)
                continue

            df = pd.read_parquet(file_path, filters=[("ticker", "in", list(tickers))])
            if df.empty:
                current_date += timedelta(days=1)
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Ensure bitemporal fields exist and respect event_time <= knowledge_time
            if "event_time" not in df.columns:
                df["event_time"] = df["timestamp"]
            else:
                df["event_time"] = pd.to_datetime(df["event_time"])

            if "knowledge_time" not in df.columns:
                df["knowledge_time"] = df["timestamp"]
            else:
                df["knowledge_time"] = pd.to_datetime(df["knowledge_time"])

            if "ingestion_time" not in df.columns:
                now_utc = datetime.now(UTC).replace(tzinfo=None)
                df["ingestion_time"] = df["knowledge_time"].clip(lower=pd.Timestamp(now_utc))

            # Validate bitemporal causality (knowledge_time >= event_time)
            if (df["knowledge_time"] < df["event_time"]).any():
                raise ValueError(
                    f"Bitemporal causality violation in {file_path}: knowledge_time < event_time"
                )

            # Calculate bid-ask spread in basis points
            df["spread_bps"] = ((df["best_ask"] - df["best_bid"]) / df["best_bid"]) * 10000.0

            self.cache[cache_key] = df.copy()
            dfs.append(df)
            current_date += timedelta(days=1)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)
        return combined.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    def synchronize_to_1hz_grid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate asynchronous trades/quotes to uniform 1Hz grid (one row per ticker per second).

        Uses causal forward-fill for missing seconds (last-known values only; never backfill).
        """
        if df.empty:
            return df.copy()

        work = df.copy()
        work["timestamp"] = pd.to_datetime(work["timestamp"])
        if "spread_bps" not in work.columns and {"best_ask", "best_bid"}.issubset(work.columns):
            work["spread_bps"] = (
                (work["best_ask"] - work["best_bid"]) / work["best_bid"]
            ) * 10000.0

        frames: list[pd.DataFrame] = []
        price_cols = ["best_bid", "best_ask", "last_price", "vwap", "spread_bps"]
        vol_cols = ["bid_size", "ask_size", "volume"]

        for ticker, group in work.groupby("ticker", sort=True):
            resampled = (
                group.sort_values("timestamp")
                .set_index("timestamp")
                .resample("1s")
                .agg(
                    {
                        "best_bid": "last",
                        "best_ask": "last",
                        "bid_size": "sum",
                        "ask_size": "sum",
                        "last_price": "last",
                        "volume": "sum",
                        "vwap": "mean",
                        "spread_bps": "mean",
                    }
                )
            )
            resampled[price_cols] = resampled[price_cols].ffill()
            resampled[vol_cols] = resampled[vol_cols].fillna(0.0)
            resampled["ticker"] = str(ticker)
            resampled = resampled.reset_index()
            resampled["event_time"] = resampled["timestamp"]
            resampled["knowledge_time"] = resampled["timestamp"]
            frames.append(resampled)

        result = pd.concat(frames, ignore_index=True)
        ordered_cols = [
            "ticker",
            "timestamp",
            "best_bid",
            "best_ask",
            "bid_size",
            "ask_size",
            "last_price",
            "volume",
            "vwap",
            "spread_bps",
            "event_time",
            "knowledge_time",
        ]
        existing_cols = [c for c in ordered_cols if c in result.columns]
        return result[existing_cols].sort_values(["ticker", "timestamp"]).reset_index(drop=True)
