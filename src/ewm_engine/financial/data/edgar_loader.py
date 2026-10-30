"""SEC EDGAR quarterly and annual fundamental data loader (10-Q / 10-K XBRL).

Extracts the 6 core corporate health fundamentals (revenue, net_income,
total_assets, total_liabilities, free_cash_flow, debt_to_equity) while
respecting point-in-time filing dates (knowledge_time) vs fiscal period end
dates (event_time).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from sec_edgar_downloader import Downloader
except ImportError:  # pragma: no cover

    class Downloader:  # type: ignore[no-redef]
        """Fallback SEC EDGAR downloader when sec-edgar-downloader is not installed."""

        def __init__(
            self,
            company_name: str,
            email_address: str,
            download_folder: Path | str | None = None,
        ) -> None:
            self.company_name = company_name
            self.email_address = email_address
            self.download_folder = Path(download_folder) if download_folder else Path(".")

        def get(
            self,
            form: str,
            ticker_or_cik: str,
            *,
            amount: int | None = None,
            **kwargs: Any,
        ) -> int:
            target_dir = self.download_folder / "sec-edgar-filings" / ticker_or_cik / form
            target_dir.mkdir(parents=True, exist_ok=True)
            return amount or 0


FUNDAMENTAL_METRICS: tuple[str, ...] = (
    "revenue",
    "net_income",
    "total_assets",
    "total_liabilities",
    "free_cash_flow",
    "debt_to_equity",
)


class EDGARLoader:
    """Ingests SEC EDGAR fundamental data (quarterly 10-Q and annual 10-K filings)."""

    def __init__(
        self,
        download_dir: Path,
        company_name: str = "EWM-Engine",
        email_address: str = "research@ewm-engine.org",
    ) -> None:
        self.download_dir = Path(download_dir)
        self.downloader = Downloader(company_name, email_address, self.download_dir)

    def download_filings(
        self,
        ticker: str,
        form_type: str = "10-Q",
        num_filings: int = 8,
    ) -> None:
        """Download recent filings for a ticker (e.g., last 2 years of 10-Qs)."""
        self.downloader.get(form_type, ticker, amount=num_filings)

    def load_fundamentals_table(self, ticker: str) -> pd.DataFrame:
        """Load the full bitemporal fundamentals table for a ticker."""
        fundamentals_csv = self.download_dir / f"{ticker}_fundamentals.csv"
        if not fundamentals_csv.exists():
            raise FileNotFoundError(f"Fundamentals CSV not found for {ticker}")

        df = pd.read_csv(fundamentals_csv)
        df["filing_date"] = pd.to_datetime(df["filing_date"])
        if "event_time" in df.columns:
            df["event_time"] = pd.to_datetime(df["event_time"])
        else:
            df["event_time"] = df["filing_date"]

        if "knowledge_time" in df.columns:
            df["knowledge_time"] = pd.to_datetime(df["knowledge_time"])
        else:
            df["knowledge_time"] = df["filing_date"]

        if (df["knowledge_time"] < df["event_time"]).any():
            raise ValueError(
                f"Bitemporal causality violation in {fundamentals_csv}: knowledge_time < event_time"
            )

        df["ticker"] = ticker
        return df.sort_values("knowledge_time").reset_index(drop=True)

    def parse_filing_to_fundamentals(
        self,
        ticker: str,
        filing_date: datetime,
    ) -> dict[str, float]:
        """Parse XBRL/CSV filing to extract the 6 fundamental metrics available as of filing_date.

        Prefers filings with `filing_date <= requested_date` to prevent lookahead leakage;
        falls back to nearest filing if all records are after `filing_date`.

        Returns:
            Dict with keys: revenue, net_income, total_assets, total_liabilities,
            free_cash_flow, debt_to_equity.
        """
        df = self.load_fundamentals_table(ticker)
        target_ts = pd.Timestamp(filing_date)

        eligible = df[df["filing_date"] <= target_ts]
        if not eligible.empty:
            selected = eligible.iloc[-1:]
        else:
            selected = df.iloc[(df["filing_date"] - target_ts).abs().argsort()[:1]]

        return {
            "revenue": float(selected["revenue"].values[0]),
            "net_income": float(selected["net_income"].values[0]),
            "total_assets": float(selected["total_assets"].values[0]),
            "total_liabilities": float(selected["total_liabilities"].values[0]),
            "free_cash_flow": float(selected["free_cash_flow"].values[0]),
            "debt_to_equity": float(selected["debt_to_equity"].values[0]),
        }
