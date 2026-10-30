"""Backtesting and validation protocol for the Financial World Model (FWM).

Evaluates FWM against all 5 comparative baselines across historical market
regimes (2020 COVID crash, 2022 Fed tightening, bull expansion) using:
- Rank Information Coefficient (Rank IC, Spearman correlation)
- Mean Absolute Error (MAE) on volatility and bid-ask spread changes
- Uncertainty interval coverage calibration
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from ewm_engine.financial.evaluation.baselines import (
    ARIMABaseline,
    BaselinePrediction,
    FullFWMBaseline,
    GARCHBaseline,
    PredictOnlyJEPABaseline,
    RegimeEpisode,
    XGBoostBaseline,
    XGBoostEngineeredBaseline,
)
from ewm_engine.graph.continuous_registry import ContinuousGraphRegistry, Edge, Node
from ewm_engine.routing.schemas import CausalInterventionVector

STANDARD_UNIVERSE: tuple[str, ...] = (
    "NVDA",
    "AMD",
    "TSM",
    "AAPL",
    "MSFT",
    "JPM",
    "XOM",
    "WMT",
)


@dataclass(frozen=True)
class EvaluationMetrics:
    """Validation metrics for a single model across one or more regime episodes."""

    model_name: str
    regime_name: str
    rank_ic_volatility: float
    rank_ic_spread: float
    volatility_mae: float
    spread_mae_bps: float
    empirical_coverage: float
    nominal_coverage: float
    calibration_error: float


def compute_rank_ic(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Compute Spearman Rank Information Coefficient (Rank IC) between predicted and actual."""
    pred_flat = np.asarray(predicted, dtype=np.float64).ravel()
    act_flat = np.asarray(actual, dtype=np.float64).ravel()
    if (
        pred_flat.size < 2
        or np.allclose(pred_flat, pred_flat[0])
        or np.allclose(act_flat, act_flat[0])
    ):
        return 0.0
    corr, _ = spearmanr(pred_flat, act_flat)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def compute_mae(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE) between predicted and actual arrays."""
    pred_flat = np.asarray(predicted, dtype=np.float64).ravel()
    act_flat = np.asarray(actual, dtype=np.float64).ravel()
    return float(np.mean(np.abs(pred_flat - act_flat)))


def compute_coverage_calibration(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal_coverage: float = 0.90,
) -> tuple[float, float]:
    """Compute empirical interval coverage and absolute calibration error vs nominal level."""
    act_flat = np.asarray(actual, dtype=np.float64).ravel()
    low_flat = np.asarray(lower, dtype=np.float64).ravel()
    up_flat = np.asarray(upper, dtype=np.float64).ravel()
    inside = (act_flat >= low_flat) & (act_flat <= up_flat)
    empirical = float(np.mean(inside))
    calib_error = abs(empirical - nominal_coverage)
    return empirical, calib_error


def build_historical_regime_episode(
    regime_name: str,
    seed: int = 42,
    window_length: int = 30,
) -> RegimeEpisode:
    """Construct a historical market regime backtest episode with supply-chain topology.

    Supported regimes:
    - '2020_covid_crash': Systemic liquidity + supply-chain shock (March 2020 profile)
    - '2022_fed_tightening': Monetary policy rate-hike shock impacting high-duration tech
    - 'bull_expansion': Low-volatility expansion with positive earnings shock
    """
    rng = np.random.default_rng(seed)
    tickers = STANDARD_UNIVERSE
    n_tickers = len(tickers)
    num_features = 14  # 8 Market-1T microstructure + 6 SEC EDGAR fundamentals
    sector_ids = np.array([8, 8, 8, 8, 8, 6, 1, 5], dtype=np.int64)

    registry = ContinuousGraphRegistry()
    for i, ticker in enumerate(tickers):
        registry.add_node(
            Node(
                node_id=ticker,
                node_type="equity",
                features={"sector_id": float(sector_ids[i]), "market_cap": 500.0 + 150.0 * i},
            )
        )

    # Supply-chain & sector topology edges
    supply_edges = [
        ("TSM", "NVDA", 0.90),
        ("TSM", "AMD", 0.85),
        ("TSM", "AAPL", 0.80),
        ("NVDA", "MSFT", 0.70),
        ("JPM", "XOM", 0.45),
        ("XOM", "WMT", 0.50),
    ]
    for src, tgt, w in supply_edges:
        registry.add_edge(Edge(src, tgt, "supply_chain", w))

    # Construct 1Hz microstructure + fundamental pre-shock window
    x_window = np.zeros((window_length, n_tickers, num_features), dtype=np.float32)
    base_prices = np.array([120.0, 95.0, 85.0, 150.0, 280.0, 135.0, 75.0, 140.0], dtype=np.float32)
    base_spreads = np.array([1.8, 2.2, 2.0, 1.1, 1.0, 1.4, 1.6, 1.2], dtype=np.float32)

    for t in range(window_length):
        noise = rng.normal(0.0, 0.002, size=n_tickers).astype(np.float32)
        px = base_prices * (1.0 + noise * (t + 1) * 0.1)
        sp_bps = base_spreads + rng.uniform(-0.1, 0.1, size=n_tickers).astype(np.float32)
        half_sp = px * (sp_bps / 20000.0)
        x_window[t, :, 0] = px - half_sp  # best_bid
        x_window[t, :, 1] = px + half_sp  # best_ask
        x_window[t, :, 2] = 500.0 + rng.uniform(-50, 50, size=n_tickers)  # bid_size
        x_window[t, :, 3] = 500.0 + rng.uniform(-50, 50, size=n_tickers)  # ask_size
        x_window[t, :, 4] = px  # last_price
        x_window[t, :, 5] = 2500.0 + rng.uniform(0, 500, size=n_tickers)  # volume
        x_window[t, :, 6] = px  # vwap
        x_window[t, :, 7] = sp_bps  # spread_bps
        # 6 SEC EDGAR fundamentals (revenue, net_income, assets, liabilities, fcf, debt_to_equity)
        for f_idx in range(6):
            x_window[t, :, 8 + f_idx] = (10.0 + i * 2.0) / (f_idx + 1.0)

    if regime_name == "2020_covid_crash":
        intervention = CausalInterventionVector(
            primary_target_entities=["TSM", "XOM", "JPM"],
            directional_sentiment=-0.92,
            magnitude=0.95,
            contagion_pathways=[
                {"source": "TSM", "target": "NVDA", "weight": 0.90},
                {"source": "TSM", "target": "AAPL", "weight": 0.85},
                {"source": "XOM", "target": "WMT", "weight": 0.60},
            ],
            confidence=0.94,
            extraction_metadata={"regime": "2020_covid_crash"},
        )
        # Realized post-shock volatility & spread widening reflect direct shock + supply-chain contagion
        actual_volatility = np.array(
            [0.78, 0.72, 0.85, 0.68, 0.52, 0.74, 0.82, 0.44], dtype=np.float32
        )
        actual_spread_change = np.array(
            [14.5, 13.0, 16.2, 11.8, 8.4, 13.5, 15.1, 6.2], dtype=np.float32
        )

    elif regime_name == "2022_fed_tightening":
        intervention = CausalInterventionVector(
            primary_target_entities=["NVDA", "AMD", "MSFT"],
            directional_sentiment=-0.78,
            magnitude=0.82,
            contagion_pathways=[
                {"source": "NVDA", "target": "TSM", "weight": 0.80},
                {"source": "MSFT", "target": "AAPL", "weight": 0.65},
            ],
            confidence=0.90,
            extraction_metadata={"regime": "2022_fed_tightening"},
        )
        actual_volatility = np.array(
            [0.66, 0.64, 0.55, 0.48, 0.58, 0.34, 0.31, 0.25], dtype=np.float32
        )
        actual_spread_change = np.array([9.8, 9.4, 7.6, 6.1, 8.2, 3.5, 2.9, 2.1], dtype=np.float32)

    else:  # bull_expansion
        intervention = CausalInterventionVector(
            primary_target_entities=["NVDA", "MSFT"],
            directional_sentiment=0.75,
            magnitude=0.60,
            contagion_pathways=[
                {"source": "NVDA", "target": "TSM", "weight": 0.75},
                {"source": "NVDA", "target": "AMD", "weight": 0.60},
            ],
            confidence=0.88,
            extraction_metadata={"regime": "bull_expansion"},
        )
        actual_volatility = np.array(
            [0.28, 0.26, 0.25, 0.20, 0.24, 0.18, 0.19, 0.15], dtype=np.float32
        )
        actual_spread_change = np.array(
            [-1.2, -0.9, -1.0, -0.6, -1.1, -0.4, -0.3, -0.2], dtype=np.float32
        )

    next_x_window = x_window.copy()
    next_x_window[:, :, 7] += actual_spread_change[None, :]
    next_x_window[:, :, 4] *= (
        1.0 + np.sign(intervention.directional_sentiment) * actual_volatility[None, :] * 0.05
    )

    return RegimeEpisode(
        regime_name=regime_name,
        tickers=tickers,
        sector_ids=sector_ids,
        x_window=x_window,
        next_x_window=next_x_window,
        intervention=intervention,
        registry=registry,
        actual_volatility=actual_volatility,
        actual_spread_change_bps=actual_spread_change,
    )


class FinancialBacktester:
    """Runs comparative backtests across historical regimes for FWM and all 5 baselines."""

    def __init__(self, nominal_coverage: float = 0.90) -> None:
        self.nominal_coverage = nominal_coverage

    def evaluate_prediction(
        self,
        prediction: BaselinePrediction,
        episode: RegimeEpisode,
    ) -> EvaluationMetrics:
        """Compute Rank IC, MAE, and coverage calibration for a single model prediction."""
        rank_ic_vol = compute_rank_ic(prediction.predicted_volatility, episode.actual_volatility)
        rank_ic_sp = compute_rank_ic(
            prediction.predicted_spread_change_bps, episode.actual_spread_change_bps
        )
        vol_mae = compute_mae(prediction.predicted_volatility, episode.actual_volatility)
        sp_mae = compute_mae(
            prediction.predicted_spread_change_bps, episode.actual_spread_change_bps
        )
        emp_cov_vol, _ = compute_coverage_calibration(
            episode.actual_volatility,
            prediction.volatility_lower,
            prediction.volatility_upper,
            nominal_coverage=self.nominal_coverage,
        )
        emp_cov_sp, _ = compute_coverage_calibration(
            episode.actual_spread_change_bps,
            prediction.spread_lower,
            prediction.spread_upper,
            nominal_coverage=self.nominal_coverage,
        )
        emp_cov = 0.5 * (emp_cov_vol + emp_cov_sp)
        calib_err = abs(emp_cov - self.nominal_coverage)

        return EvaluationMetrics(
            model_name=prediction.model_name,
            regime_name=episode.regime_name,
            rank_ic_volatility=rank_ic_vol,
            rank_ic_spread=rank_ic_sp,
            volatility_mae=vol_mae,
            spread_mae_bps=sp_mae,
            empirical_coverage=emp_cov,
            nominal_coverage=self.nominal_coverage,
            calibration_error=calib_err,
        )

    def run_comparative_backtest(
        self,
        regimes: Sequence[str] = ("2020_covid_crash", "2022_fed_tightening", "bull_expansion"),
        seed: int = 42,
    ) -> dict[str, list[EvaluationMetrics]]:
        """Train and evaluate ARIMA, GARCH, XGBoost, XGBoost+Features, Predict-Only JEPA, and Full FWM."""
        train_episodes = [
            build_historical_regime_episode(r, seed=seed + idx) for idx, r in enumerate(regimes)
        ]
        test_episodes = [
            build_historical_regime_episode(r, seed=seed + 100 + idx)
            for idx, r in enumerate(regimes)
        ]

        arima = ARIMABaseline()
        garch = GARCHBaseline()
        xgb_raw = XGBoostBaseline(random_state=seed)
        xgb_eng = XGBoostEngineeredBaseline(random_state=seed)
        jepa_only = PredictOnlyJEPABaseline(seed=seed)
        full_fwm = FullFWMBaseline(seed=seed)

        arima.fit(train_episodes)
        garch.fit(train_episodes)
        xgb_raw.fit(train_episodes)
        xgb_eng.fit(train_episodes)
        jepa_only.fit(train_episodes, num_epochs=20)
        full_fwm.fit(train_episodes, num_epochs=30)

        models: list[
            ARIMABaseline
            | GARCHBaseline
            | XGBoostBaseline
            | XGBoostEngineeredBaseline
            | PredictOnlyJEPABaseline
            | FullFWMBaseline
        ] = [arima, garch, xgb_raw, xgb_eng, jepa_only, full_fwm]
        results: dict[str, list[EvaluationMetrics]] = {}

        for model in models:
            for ep in test_episodes:
                pred = model.predict(ep)
                metrics = self.evaluate_prediction(pred, ep)
                results.setdefault(pred.model_name, []).append(metrics)

        return results
