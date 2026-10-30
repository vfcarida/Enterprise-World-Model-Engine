# Model Card: Financial JEPA Encoder & Transition Predictor

**Author:** Vinícius Caridá  
**Module:** `ewm_engine.financial.models` (`FinancialJEPAEncoder`, `FinancialJEPAPair`, `FinancialTransitionPredictor`, `FinancialWorldModel`)  
**License:** Apache-2.0

---

## 1. Model Details

- **Architecture:** Joint-Embedding Predictive Architecture (JEPA) combining a Vision Transformer (`ViTModel`) context/target encoder pair with an `EvolveGCN`-augmented Transformer transition predictor (`g_theta(z_t, A_t) -> z_hat_{t+1}`) and an MLP financial metrics decoder (`FinancialMetricsDecoder`).
- **Regularization:** Variance-Invariance-Covariance Regularization (`VICRegLoss`) and Sketched-Isotropic-Gaussian Regularization (`sigreg_loss`), monitored via effective rank (`compute_rankme`).
- **Inputs:**
  - Bitemporally synchronized `(B, T, N, 14)` financial state tensor combining 8 Market-1T 1Hz limit order book microstructure features and 6 point-in-time SEC EDGAR fundamentals.
  - Optional `CausalInterventionVector` ($A_t$) and `ContinuousGraphRegistry` supply-chain/sector topology.
- **Outputs:**
  - Predicted next-step latent representation $\hat{z}_{t+1} \in \mathbb{R}^D$.
  - Asset-level realized volatility $\hat{\sigma}_{t+1}$, bid-ask spread change $\Delta\text{spread}_{\text{bps}}$, implied returns, 90% uncertainty intervals, and portfolio $\text{VaR}_{95}$ / $\text{CVaR}_{95}$.

## 2. Intended Use

- Counterfactual stress testing and systemic contagion simulation across multi-asset equity portfolios under exogenous macro, policy, or supply-chain shocks.
- Comparative research benchmarking state-transition world models against statistical (ARIMA, GARCH), tree-based (XGBoost), and predict-only JEPA baselines.

## 3. Out-of-Scope Use

- Automated high-frequency live order execution without human risk oversight.
- Extrapolation to illiquid over-the-counter derivatives lacking 1Hz order book telemetry or quarterly financial disclosures.

## 4. Evaluation & Metrics

- **Rank Information Coefficient (Rank IC):** Spearman rank correlation between predicted and realized cross-sectional volatility and spread widening.
- **Mean Absolute Error (MAE):** Evaluated on annualized volatility and basis-point bid-ask spread changes across historical stress regimes (`2020_covid_crash`, `2022_fed_tightening`, `bull_expansion`).
- **Interval Calibration:** Empirical coverage against nominal 90% uncertainty bounds.

## 5. Ethical Considerations & Limitations

- All fundamental inputs must pass `BitemporalSynchronizer.verify_no_future_leakage` (`fundamental.knowledge_time <= market.knowledge_time`) to prevent look-ahead bias.
- Exogenous shock vectors extracted from unstructured text depend on upstream Mechanistic Router confidence calibration.
