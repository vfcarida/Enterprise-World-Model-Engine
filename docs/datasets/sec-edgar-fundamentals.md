# Dataset Card: SEC EDGAR Bitemporal Fundamentals

**Author:** Vinícius Caridá  
**Ingestion Modules:** `ewm_engine.financial.data.EDGARLoader`, `ewm_engine.financial.data.BitemporalSynchronizer`  
**Maturity Tier:** **Experimental (`Synthetic Schema & Point-in-Time Synchronizer`)**  
**License:** Apache-2.0

> [!WARNING]
> **Scope & Data Availability Notice (`EWM-007` Audit):** This card documents the point-in-time bitemporal schema enforced by `EDGARLoader` and `BitemporalSynchronizer`. Full historical SEC EDGAR bulk mirrors are **not** bundled in this repository; all automated tests run offline against deterministic synthetic `10-Q`/`10-K` filing fixtures.

---

## 1. Dataset Description

- **Domain:** Public US corporate quarterly (`10-Q`) and annual (`10-K`) financial statements retrieved from the SEC EDGAR system.
- **Extracted Fundamental Metrics (6 channels):**
  1. `revenue`: Total quarterly operating revenue.
  2. `net_income`: GAAP net income.
  3. `total_assets`: Total balance-sheet assets.
  4. `total_liabilities`: Total balance-sheet liabilities.
  5. `free_cash_flow`: Operating cash flow minus capital expenditures.
  6. `debt_to_equity`: Total debt divided by shareholders' equity.

## 2. Bitemporal Discipline & Leakage Prevention

- Every filing record carries two distinct timestamps:
  - `event_time`: Fiscal period end date (e.g., `2026-03-31`).
  - `knowledge_time`: Public SEC acceptance/filing timestamp (e.g., `2026-05-08`).
- `BitemporalSynchronizer.synchronize_point_in_time` joins 1Hz market observations with fundamental filings using `pd.merge_asof(..., direction="backward")` strictly on `knowledge_time`, ensuring that accounting numbers are never visible to the model prior to their public filing timestamp.
