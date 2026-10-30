"""Bitemporal synchronization of 1Hz Market-1T microstructure and SEC EDGAR fundamentals.

Enforces strict point-in-time causality (`fundamental.knowledge_time <= market.knowledge_time`)
so that quarterly accounting statements are never leaked prior to their public SEC
filing timestamp.
"""

from __future__ import annotations

import pandas as pd

from ewm_engine.financial.data.edgar_loader import FUNDAMENTAL_METRICS


class BitemporalSynchronizer:
    """Synchronizes market microstructure + fundamental data with bitemporal discipline."""

    @staticmethod
    def merge_market_and_fundamentals(
        market_df: pd.DataFrame,
        fundamentals_dict: dict[str, dict[str, float]],
    ) -> pd.DataFrame:
        """Merge 1Hz market data with fundamental metrics dictionary.

        If a ticker entry in `fundamentals_dict` includes a `knowledge_time` timestamp
        (epoch or ISO/Timestamp), fundamentals are only populated for market rows where
        `market_df['knowledge_time'] >= fundamental_knowledge_time` (preventing leakage).

        Args:
            market_df: 1Hz market DataFrame containing at least `ticker` and `timestamp`.
            fundamentals_dict: Mapping `{ticker: {metric: value}}`.

        Returns:
            Merged DataFrame with all market and 6 fundamental columns.
        """
        if market_df.empty:
            out = market_df.copy()
            for col in FUNDAMENTAL_METRICS:
                out[col] = pd.Series(dtype="float64")
            return out

        merged = market_df.copy()
        if "knowledge_time" not in merged.columns and "timestamp" in merged.columns:
            merged["knowledge_time"] = pd.to_datetime(merged["timestamp"])
        elif "knowledge_time" in merged.columns:
            merged["knowledge_time"] = pd.to_datetime(merged["knowledge_time"])

        for col in FUNDAMENTAL_METRICS:
            merged[col] = merged["ticker"].map(
                lambda t, c=col: float(fundamentals_dict.get(str(t), {}).get(c, 0.0))
            )

        # Enforce knowledge_time guard if provided in fundamentals_dict
        for ticker, f_data in fundamentals_dict.items():
            if "knowledge_time" in f_data and "knowledge_time" in merged.columns:
                f_kt = pd.Timestamp(f_data["knowledge_time"])
                pre_knowledge_mask = (merged["ticker"] == ticker) & (
                    merged["knowledge_time"] < f_kt
                )
                if pre_knowledge_mask.any():
                    for col in FUNDAMENTAL_METRICS:
                        merged.loc[pre_knowledge_mask, col] = 0.0

        return merged

    @staticmethod
    def synchronize_point_in_time(
        market_df: pd.DataFrame,
        fundamentals_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Point-in-time bitemporal join using backward as-of matching on `knowledge_time`.

        Guarantees that a market observation at `knowledge_time = t` only sees SEC filings
        whose `knowledge_time <= t`, even when the filing's fiscal `event_time <= t`.
        """
        if market_df.empty:
            return BitemporalSynchronizer.merge_market_and_fundamentals(market_df, {})

        m_df = market_df.copy()
        m_df["timestamp"] = pd.to_datetime(m_df["timestamp"])
        if "knowledge_time" not in m_df.columns:
            m_df["knowledge_time"] = m_df["timestamp"]
        else:
            m_df["knowledge_time"] = pd.to_datetime(m_df["knowledge_time"])

        if fundamentals_df.empty:
            return BitemporalSynchronizer.merge_market_and_fundamentals(m_df, {})

        f_df = fundamentals_df.copy()
        if "knowledge_time" not in f_df.columns:
            f_df["knowledge_time"] = pd.to_datetime(f_df["filing_date"])
        else:
            f_df["knowledge_time"] = pd.to_datetime(f_df["knowledge_time"])

        f_df = f_df.rename(columns={"knowledge_time": "fundamental_knowledge_time"})
        keep_cols = ["ticker", "fundamental_knowledge_time", *FUNDAMENTAL_METRICS]
        available_cols = [c for c in keep_cols if c in f_df.columns]
        f_sub = f_df[available_cols].sort_values("fundamental_knowledge_time")

        m_sorted = m_df.sort_values("knowledge_time")
        joined = pd.merge_asof(
            m_sorted,
            f_sub,
            by="ticker",
            left_on="knowledge_time",
            right_on="fundamental_knowledge_time",
            direction="backward",
        )
        for col in FUNDAMENTAL_METRICS:
            if col in joined.columns:
                joined[col] = joined[col].fillna(0.0)
            else:
                joined[col] = 0.0

        return joined.sort_values(["ticker", "timestamp"]).reset_index(drop=True)

    @staticmethod
    def verify_no_future_leakage(synchronized_df: pd.DataFrame) -> bool:
        """Audit that no row in a synchronized DataFrame used future knowledge."""
        if (
            "fundamental_knowledge_time" not in synchronized_df.columns
            or "knowledge_time" not in synchronized_df.columns
        ):
            return True
        valid_mask = synchronized_df["fundamental_knowledge_time"].notna()
        if not valid_mask.any():
            return True
        leaked = (
            synchronized_df.loc[valid_mask, "fundamental_knowledge_time"]
            > synchronized_df.loc[valid_mask, "knowledge_time"]
        )
        return not bool(leaked.any())
