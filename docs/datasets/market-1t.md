# Dataset Card: Market-1T High-Frequency Microstructure

**Author:** Vinícius Caridá  
**Ingestion Module:** `ewm_engine.financial.data.Market1TLoader`  
**License:** Apache-2.0

---

## 1. Dataset Description

- **Domain:** US Equity Limit Order Book (LOB) and trade microstructure at 1Hz (one observation per second per ticker).
- **Storage Format:** Date-partitioned Apache Parquet files (`market1t_YYYYMMDD.parquet`).
- **Schema & Features (8 channels):**
  1. `best_bid` (`float64`): Highest displayed bid price.
  2. `best_ask` (`float64`): Lowest displayed ask price (`>= best_bid`).
  3. `bid_size` (`float64`): Aggregate share volume at `best_bid`.
  4. `ask_size` (`float64`): Aggregate share volume at `best_ask`.
  5. `last_price` (`float64`): Most recent execution price.
  6. `volume` (`float64`): Executed share volume within the 1-second bucket.
  7. `vwap` (`float64`): Volume-weighted average price.
  8. `spread_bps` (`float64`): Derived bid-ask spread in basis points:
     $$\text{spread\_bps} = \frac{\text{best\_ask} - \text{best\_bid}}{\text{last\_price}} \times 10{,}000$$

## 2. Preprocessing & Temporal Alignment

- `Market1TLoader.synchronize_to_1hz_grid` resamples asynchronous ticker streams onto a uniform 1-second lattice (`1s`), forward-filling price/quote fields (`best_bid`, `best_ask`, `last_price`, `vwap`) and zero-filling `volume` during seconds without trades.
