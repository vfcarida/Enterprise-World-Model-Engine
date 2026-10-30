"""Domain-specific node feature schema for Financial World Model entities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass
class FinancialNodeFeatures:
    """Standardized feature schema for financial entities."""

    market_cap: float  # Market capitalization in billions USD
    vwap: float  # Volume-weighted average price (last 1 hour)
    bid_ask_spread_bps: float  # Bid-ask spread in basis points
    sector_id: int  # GICS sector code (0-11)
    volatility_30d: float  # 30-day realized volatility

    def to_dict(self) -> dict[str, float]:
        """Serialize features to a dictionary with float values."""
        return {
            "market_cap": float(self.market_cap),
            "vwap": float(self.vwap),
            "bid_ask_spread_bps": float(self.bid_ask_spread_bps),
            "sector_id": float(self.sector_id),
            "volatility_30d": float(self.volatility_30d),
        }

    def to_vector(self) -> np.ndarray:
        """Convert features to a 1D float32 numpy array in canonical feature order."""
        return np.array(
            [
                self.market_cap,
                self.vwap,
                self.bid_ask_spread_bps,
                float(self.sector_id),
                self.volatility_30d,
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, float]) -> FinancialNodeFeatures:
        """Construct FinancialNodeFeatures from a feature dictionary."""
        return cls(
            market_cap=float(data["market_cap"]),
            vwap=float(data["vwap"]),
            bid_ask_spread_bps=float(data["bid_ask_spread_bps"]),
            sector_id=int(data["sector_id"]),
            volatility_30d=float(data["volatility_30d"]),
        )

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return canonical feature names."""
        return [
            "market_cap",
            "vwap",
            "bid_ask_spread_bps",
            "sector_id",
            "volatility_30d",
        ]
