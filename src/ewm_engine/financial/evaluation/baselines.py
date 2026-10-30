"""Comparative baselines for Financial World Model evaluation.

Implements all 5 comparative baselines and the full Financial World Model:
1. ARIMABaseline (statistical ARMA(1,1) on asset series)
2. GARCHBaseline (statistical GARCH(1,1) conditional variance model)
3. XGBoostBaseline (gradient boosting on raw microstructure + fundamental features)
4. XGBoostEngineeredBaseline (gradient boosting + sector dummies + order book imbalance)
5. PredictOnlyJEPABaseline (JEPA without causal intervention conditioning or graph contagion)
6. FullFWMBaseline (Full FWM: JEPA + VICReg + CT-TGNN + CausalInterventionVector)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor

from ewm_engine.financial.models.transition_predictor import FinancialWorldModel
from ewm_engine.graph.continuous_registry import ContinuousGraphRegistry
from ewm_engine.routing.schemas import CausalInterventionVector


@dataclass(frozen=True)
class RegimeEpisode:
    """Single historical regime backtest episode."""

    regime_name: str
    tickers: tuple[str, ...]
    sector_ids: np.ndarray  # (N,) integer GICS sector codes
    x_window: np.ndarray  # (T, N, F) 1Hz market + fundamental tensor
    next_x_window: np.ndarray  # (T, N, F) true future window
    intervention: CausalInterventionVector
    registry: ContinuousGraphRegistry
    actual_volatility: np.ndarray  # (N,) realized post-shock volatility
    actual_spread_change_bps: np.ndarray  # (N,) realized post-shock spread widening


@dataclass(frozen=True)
class BaselinePrediction:
    """Standardized prediction output across all baseline and world-model estimators."""

    model_name: str
    predicted_volatility: np.ndarray  # (N,)
    predicted_spread_change_bps: np.ndarray  # (N,)
    volatility_lower: np.ndarray  # (N,)
    volatility_upper: np.ndarray  # (N,)
    spread_lower: np.ndarray  # (N,)
    spread_upper: np.ndarray  # (N,)


class ARIMABaseline:
    """Statistical ARMA(1,1) / ARIMA(1,0,1) baseline on trailing asset series."""

    def __init__(self, ar_phi: float = 0.65, ma_theta: float = 0.25) -> None:
        self.ar_phi = ar_phi
        self.ma_theta = ma_theta
        self.vol_drift: float = 0.0
        self.spread_drift: float = 0.0
        self.residual_std_vol: float = 0.08
        self.residual_std_spread: float = 2.5

    def fit(self, episodes: Sequence[RegimeEpisode]) -> None:
        """Estimate drift and residual dispersion from training episodes."""
        if not episodes:
            return
        vol_Res: list[np.ndarray] = []
        sp_res: list[np.ndarray] = []
        for ep in episodes:
            hist_vol = np.std(np.diff(ep.x_window[:, :, 4], axis=0), axis=0) + 0.05
            hist_sp = ep.x_window[-1, :, 7] - ep.x_window[0, :, 7]
            vol_Res.append(ep.actual_volatility - hist_vol)
            sp_res.append(ep.actual_spread_change_bps - hist_sp)
        all_v = np.concatenate(vol_Res)
        all_s = np.concatenate(sp_res)
        self.vol_drift = float(np.mean(all_v))
        self.spread_drift = float(np.mean(all_s))
        self.residual_std_vol = float(np.std(all_v)) + 0.02
        self.residual_std_spread = float(np.std(all_s)) + 0.5

    def predict(self, episode: RegimeEpisode) -> BaselinePrediction:
        """Forecast volatility and spread change using ARMA(1,1) on trailing window."""
        prices = episode.x_window[:, :, 4]  # last_price
        spreads = episode.x_window[:, :, 7]  # spread_bps
        returns = np.diff(prices, axis=0) / np.clip(prices[:-1], 1e-4, None)
        trailing_vol = np.std(returns, axis=0) * np.sqrt(252.0) + 0.10
        prev_vol = np.std(returns[:-1], axis=0) * np.sqrt(252.0) + 0.10
        innov_vol = trailing_vol - prev_vol

        pred_vol = np.clip(
            self.vol_drift + self.ar_phi * trailing_vol + self.ma_theta * innov_vol,
            0.01,
            None,
        )

        sp_diff = spreads[-1] - spreads[0]
        sp_innov = spreads[-1] - np.mean(spreads, axis=0)
        pred_sp = self.spread_drift + self.ar_phi * sp_diff + self.ma_theta * sp_innov

        z_90 = 1.645
        return BaselinePrediction(
            model_name="ARIMA",
            predicted_volatility=pred_vol.astype(np.float32),
            predicted_spread_change_bps=pred_sp.astype(np.float32),
            volatility_lower=np.clip(pred_vol - z_90 * self.residual_std_vol, 0.0, None).astype(
                np.float32
            ),
            volatility_upper=(pred_vol + z_90 * self.residual_std_vol).astype(np.float32),
            spread_lower=(pred_sp - z_90 * self.residual_std_spread).astype(np.float32),
            spread_upper=(pred_sp + z_90 * self.residual_std_spread).astype(np.float32),
        )


class GARCHBaseline:
    """Statistical GARCH(1,1) conditional variance baseline."""

    def __init__(
        self,
        omega: float = 0.01,
        alpha: float = 0.25,
        beta: float = 0.65,
    ) -> None:
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.spread_scale: float = 12.0
        self.vol_unc: float = 0.07
        self.spread_unc: float = 2.2

    def fit(self, episodes: Sequence[RegimeEpisode]) -> None:
        """Calibrate GARCH(1,1) unconditional level and spread scaling from episodes."""
        if not episodes:
            return
        avg_var = np.mean([np.mean(ep.actual_volatility**2) for ep in episodes])
        self.omega = float(max(avg_var * (1.0 - self.alpha - self.beta), 1e-4))

    def predict(self, episode: RegimeEpisode) -> BaselinePrediction:
        """Iterate GARCH(1,1) recursion over the 1Hz return window."""
        prices = episode.x_window[:, :, 4]
        returns = np.diff(prices, axis=0) / np.clip(prices[:-1], 1e-4, None)
        scaled_ret = returns * np.sqrt(252.0)

        n_tickers = prices.shape[1]
        sigma2 = np.full(n_tickers, self.omega / max(1.0 - self.alpha - self.beta, 0.05))
        for t in range(scaled_ret.shape[0]):
            eps2 = scaled_ret[t] ** 2
            sigma2 = self.omega + self.alpha * eps2 + self.beta * sigma2

        pred_vol = np.sqrt(np.clip(sigma2, 1e-6, None))
        pred_sp = (pred_vol - np.mean(pred_vol)) * self.spread_scale + 1.5

        z_90 = 1.645
        return BaselinePrediction(
            model_name="GARCH",
            predicted_volatility=pred_vol.astype(np.float32),
            predicted_spread_change_bps=pred_sp.astype(np.float32),
            volatility_lower=np.clip(pred_vol - z_90 * self.vol_unc, 0.0, None).astype(np.float32),
            volatility_upper=(pred_vol + z_90 * self.vol_unc).astype(np.float32),
            spread_lower=(pred_sp - z_90 * self.spread_unc).astype(np.float32),
            spread_upper=(pred_sp + z_90 * self.spread_unc).astype(np.float32),
        )


class XGBoostBaseline:
    """Gradient boosting on raw microstructure and fundamental features."""

    def __init__(self, random_state: int = 42) -> None:
        self.vol_model = GradientBoostingRegressor(
            n_estimators=40, max_depth=3, learning_rate=0.1, random_state=random_state
        )
        self.spread_model = GradientBoostingRegressor(
            n_estimators=40, max_depth=3, learning_rate=0.1, random_state=random_state
        )
        self.vol_unc: float = 0.06
        self.spread_unc: float = 1.8
        self._is_fitted = False

    def _extract_raw_features(self, episode: RegimeEpisode) -> np.ndarray:
        """Extract raw last-step and mean microstructure + fundamental features (N, 2*F)."""
        last_step = episode.x_window[-1]  # (N, F)
        mean_step = np.mean(episode.x_window, axis=0)  # (N, F)
        return np.hstack([last_step, mean_step]).astype(np.float32)

    def fit(self, episodes: Sequence[RegimeEpisode]) -> None:
        """Fit gradient boosting regressors on raw features across episodes."""
        if not episodes:
            return
        x_list = [self._extract_raw_features(ep) for ep in episodes]
        y_vol = np.concatenate([ep.actual_volatility for ep in episodes])
        y_sp = np.concatenate([ep.actual_spread_change_bps for ep in episodes])
        x_mat = np.vstack(x_list)

        self.vol_model.fit(x_mat, y_vol)
        self.spread_model.fit(x_mat, y_sp)
        self.vol_unc = float(np.std(y_vol - self.vol_model.predict(x_mat))) + 0.03
        self.spread_unc = float(np.std(y_sp - self.spread_model.predict(x_mat))) + 0.4
        self._is_fitted = True

    def predict(self, episode: RegimeEpisode) -> BaselinePrediction:
        """Predict asset volatility and spread widening from raw features."""
        feats = self._extract_raw_features(episode)
        if self._is_fitted:
            pred_vol = np.clip(self.vol_model.predict(feats), 0.01, None)
            pred_sp = self.spread_model.predict(feats)
        else:
            pred_vol = np.std(episode.x_window[:, :, 4], axis=0) + 0.2
            pred_sp = np.zeros(len(episode.tickers))

        z_90 = 1.645
        return BaselinePrediction(
            model_name="XGBoost",
            predicted_volatility=pred_vol.astype(np.float32),
            predicted_spread_change_bps=pred_sp.astype(np.float32),
            volatility_lower=np.clip(pred_vol - z_90 * self.vol_unc, 0.0, None).astype(np.float32),
            volatility_upper=(pred_vol + z_90 * self.vol_unc).astype(np.float32),
            spread_lower=(pred_sp - z_90 * self.spread_unc).astype(np.float32),
            spread_upper=(pred_sp + z_90 * self.spread_unc).astype(np.float32),
        )


class XGBoostEngineeredBaseline(XGBoostBaseline):
    """Gradient boosting on raw features + engineered domain features (sector dummies, LOB imbalance)."""

    def _extract_raw_features(self, episode: RegimeEpisode) -> np.ndarray:
        base = super()._extract_raw_features(episode)
        bid_sz = episode.x_window[:, :, 2]
        ask_sz = episode.x_window[:, :, 3]
        lob_imbalance = np.mean((bid_sz - ask_sz) / np.clip(bid_sz + ask_sz, 1.0, None), axis=0)[
            :, None
        ]

        prices = episode.x_window[:, :, 4]
        ret_std = np.std(np.diff(prices, axis=0), axis=0)[:, None]
        spread_series = episode.x_window[:, :, 7]
        spread_momentum = (spread_series[-1] - np.mean(spread_series, axis=0))[:, None]

        # Sector one-hot encoding (12 GICS sectors)
        n_tickers = len(episode.tickers)
        sector_dummies = np.zeros((n_tickers, 12), dtype=np.float32)
        for i, sec_id in enumerate(episode.sector_ids):
            sector_dummies[i, int(sec_id) % 12] = 1.0

        return np.hstack([base, lob_imbalance, ret_std, spread_momentum, sector_dummies]).astype(
            np.float32
        )

    def predict(self, episode: RegimeEpisode) -> BaselinePrediction:
        pred = super().predict(episode)
        return BaselinePrediction(
            model_name="XGBoost+Features",
            predicted_volatility=pred.predicted_volatility,
            predicted_spread_change_bps=pred.predicted_spread_change_bps,
            volatility_lower=pred.volatility_lower,
            volatility_upper=pred.volatility_upper,
            spread_lower=pred.spread_lower,
            spread_upper=pred.spread_upper,
        )


class PredictOnlyJEPABaseline:
    """JEPA encoder + predictor without exogenous causal intervention or graph contagion."""

    def __init__(
        self,
        num_tickers: int = 8,
        num_features: int = 14,
        latent_dim: int = 64,
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        self.fwm = FinancialWorldModel(
            num_tickers=num_tickers,
            num_features=num_features,
            latent_dim=latent_dim,
            node_dim=16,
        )

    def fit(self, episodes: Sequence[RegimeEpisode], num_epochs: int = 25) -> None:
        """Train JEPA in predict-only mode (intervention=None, registry=None)."""
        if not episodes:
            return
        self.fwm.tickers = list(episodes[0].tickers)
        optimizer = torch.optim.Adam(self.fwm.parameters(), lr=0.005)
        self.fwm.train()

        for _ in range(num_epochs):
            for ep in episodes:
                x_t = torch.from_numpy(ep.x_window).unsqueeze(0)
                x_t1 = torch.from_numpy(ep.next_x_window).unsqueeze(0)
                y_vol = torch.from_numpy(ep.actual_volatility).unsqueeze(0)
                y_sp = torch.from_numpy(ep.actual_spread_change_bps).unsqueeze(0)

                loss, _ = self.fwm.compute_training_loss(
                    x_t=x_t,
                    x_t1=x_t1,
                    target_volatility=y_vol,
                    target_spread_bps=y_sp,
                    intervention=None,
                    registry=None,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.fwm.jepa_pair.update_target_encoder()

    def predict(self, episode: RegimeEpisode) -> BaselinePrediction:
        """Predict next-state metrics without intervention conditioning."""
        self.fwm.eval()
        self.fwm.tickers = list(episode.tickers)
        with torch.no_grad():
            x_t = torch.from_numpy(episode.x_window).unsqueeze(0)
            _, out = self.fwm(x_t, intervention=None, registry=None)

        return BaselinePrediction(
            model_name="Predict-Only JEPA",
            predicted_volatility=out.predicted_volatility.squeeze(0).cpu().numpy(),
            predicted_spread_change_bps=out.predicted_spread_change_bps.squeeze(0).cpu().numpy(),
            volatility_lower=out.volatility_lower_bound.squeeze(0).cpu().numpy(),
            volatility_upper=out.volatility_upper_bound.squeeze(0).cpu().numpy(),
            spread_lower=out.spread_lower_bound.squeeze(0).cpu().numpy(),
            spread_upper=out.spread_upper_bound.squeeze(0).cpu().numpy(),
        )


class FullFWMBaseline:
    """Full Financial World Model (JEPA + VICReg + CT-TGNN + CausalInterventionVector)."""

    def __init__(
        self,
        num_tickers: int = 8,
        num_features: int = 14,
        latent_dim: int = 64,
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        self.fwm = FinancialWorldModel(
            num_tickers=num_tickers,
            num_features=num_features,
            latent_dim=latent_dim,
            node_dim=16,
        )

    def fit(self, episodes: Sequence[RegimeEpisode], num_epochs: int = 35) -> None:
        """Train full FWM with causal intervention vectors and CT-TGNN topology."""
        if not episodes:
            return
        self.fwm.tickers = list(episodes[0].tickers)
        optimizer = torch.optim.Adam(self.fwm.parameters(), lr=0.008)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, num_epochs), eta_min=5e-4
        )
        self.fwm.train()

        for epoch_idx in range(num_epochs):
            optimizer.zero_grad()
            for ep_idx, ep in enumerate(episodes):
                self.fwm.tickers = list(ep.tickers)
                state_ep = episodes[(ep_idx + epoch_idx) % len(episodes)]
                x_t = torch.from_numpy(state_ep.x_window).unsqueeze(0)
                x_t1 = torch.from_numpy(ep.next_x_window).unsqueeze(0)
                y_vol = torch.from_numpy(ep.actual_volatility).unsqueeze(0)
                y_sp = torch.from_numpy(ep.actual_spread_change_bps).unsqueeze(0)

                loss, _ = self.fwm.compute_training_loss(
                    x_t=x_t,
                    x_t1=x_t1,
                    target_volatility=y_vol,
                    target_spread_bps=y_sp,
                    intervention=ep.intervention,
                    registry=ep.registry,
                )
                (loss / len(episodes)).backward()
            optimizer.step()
            scheduler.step()
            self.fwm.jepa_pair.update_target_encoder()

    def predict(self, episode: RegimeEpisode) -> BaselinePrediction:
        """Simulate counterfactual transition S_t + A_t -> S_{t+1} with graph contagion."""
        self.fwm.eval()
        self.fwm.tickers = list(episode.tickers)
        with torch.no_grad():
            x_t = torch.from_numpy(episode.x_window).unsqueeze(0)
            _, out = self.fwm(
                x_t,
                intervention=episode.intervention,
                registry=episode.registry,
            )

        return BaselinePrediction(
            model_name="Full FWM",
            predicted_volatility=out.predicted_volatility.squeeze(0).cpu().numpy(),
            predicted_spread_change_bps=out.predicted_spread_change_bps.squeeze(0).cpu().numpy(),
            volatility_lower=out.volatility_lower_bound.squeeze(0).cpu().numpy(),
            volatility_upper=out.volatility_upper_bound.squeeze(0).cpu().numpy(),
            spread_lower=out.spread_lower_bound.squeeze(0).cpu().numpy(),
            spread_upper=out.spread_upper_bound.squeeze(0).cpu().numpy(),
        )
