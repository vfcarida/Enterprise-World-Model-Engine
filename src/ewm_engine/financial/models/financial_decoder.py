"""MLP projection head decoding JEPA latent states into measurable financial metrics.

Projects predicted future latent states z_hat_{t+1} and CT-TGNN per-asset graph
embeddings into asset-level volatility (sigma_hat), bid-ask spread changes
(Delta_spread_bps), implied returns, calibrated uncertainty intervals, and
portfolio-level VaR / CVaR risk rebalancing magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FinancialPredictionOutput:
    """Decoded measurable financial predictions and risk intervals."""

    predicted_volatility: torch.Tensor  # (B, N) strictly positive asset volatility
    predicted_spread_change_bps: torch.Tensor  # (B, N) bid-ask spread change in bps
    predicted_returns: torch.Tensor  # (B, N) implied asset returns
    portfolio_var_95: torch.Tensor  # (B,) 95% Value-at-Risk magnitude
    portfolio_cvar_95: torch.Tensor  # (B,) 95% Conditional VaR (>= VaR_95)
    volatility_lower_bound: torch.Tensor  # (B, N) lower interval bound
    volatility_upper_bound: torch.Tensor  # (B, N) upper interval bound
    spread_lower_bound: torch.Tensor  # (B, N) lower interval bound
    spread_upper_bound: torch.Tensor  # (B, N) upper interval bound


class FinancialMetricsDecoder(nn.Module):
    """MLP decoder mapping predicted latent state z_hat_{t+1} to financial metrics."""

    def __init__(
        self,
        latent_dim: int = 256,
        num_tickers: int = 8,
        node_embedding_dim: int = 32,
        hidden_dim: int = 128,
        interval_z_score: float = 1.645,  # 90% nominal Gaussian/conformal multiplier
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_tickers = num_tickers
        self.node_embedding_dim = node_embedding_dim
        self.hidden_dim = hidden_dim
        self.interval_z_score = interval_z_score

        self.global_to_ticker = nn.Linear(latent_dim, num_tickers * node_embedding_dim)
        combined_ticker_dim = node_embedding_dim * 2

        self.ticker_mlp = nn.Sequential(
            nn.Linear(combined_ticker_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Output heads per ticker: [raw_vol, spread_delta, implied_return, raw_vol_unc, raw_spread_unc]
        self.ticker_head = nn.Linear(hidden_dim, 5)
        self.ticker_residual = nn.Linear(combined_ticker_dim, 5)

        # Portfolio-level risk rebalancing head (VaR, tail excess for CVaR)
        self.portfolio_mlp = nn.Sequential(
            nn.Linear(latent_dim + num_tickers * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        z_hat: torch.Tensor,
        node_embeddings: torch.Tensor | None = None,
    ) -> FinancialPredictionOutput:
        """Decode predicted latent state and graph node embeddings into financial metrics.

        Args:
            z_hat: Predicted global latent state (B, latent_dim).
            node_embeddings: Optional CT-TGNN node embeddings (N, node_dim) or (B, N, node_dim).

        Returns:
            FinancialPredictionOutput containing point predictions, VaR/CVaR, and intervals.
        """
        if z_hat.ndim == 1:
            z_hat = z_hat.unsqueeze(0)
        b_sz = z_hat.size(0)

        global_per_ticker = self.global_to_ticker(z_hat).view(
            b_sz, self.num_tickers, self.node_embedding_dim
        )

        if node_embeddings is None:
            graph_per_ticker = torch.zeros_like(global_per_ticker)
        elif node_embeddings.ndim == 2:
            graph_per_ticker = node_embeddings.unsqueeze(0).expand(
                b_sz, self.num_tickers, self.node_embedding_dim
            )
        else:
            graph_per_ticker = node_embeddings

        combined = torch.cat([global_per_ticker, graph_per_ticker], dim=-1)
        hidden = self.ticker_mlp(combined)
        raw_out = self.ticker_head(hidden) + self.ticker_residual(combined)  # (B, N, 5)

        pred_vol = F.softplus(raw_out[:, :, 0]) + 1e-4
        pred_spread = raw_out[:, :, 1]
        pred_ret = torch.tanh(raw_out[:, :, 2])

        vol_half_width = (F.softplus(raw_out[:, :, 3]) + 0.02) * self.interval_z_score
        spread_half_width = (F.softplus(raw_out[:, :, 4]) + 0.5) * self.interval_z_score

        vol_lower = torch.clamp(pred_vol - vol_half_width, min=0.0)
        vol_upper = pred_vol + vol_half_width
        spread_lower = pred_spread - spread_half_width
        spread_upper = pred_spread + spread_half_width

        portfolio_in = torch.cat(
            [
                z_hat,
                pred_vol.reshape(b_sz, -1),
                pred_spread.reshape(b_sz, -1),
                pred_ret.reshape(b_sz, -1),
            ],
            dim=-1,
        )
        port_raw = self.portfolio_mlp(portfolio_in)
        var_95 = F.softplus(port_raw[:, 0]) + 0.01 * pred_vol.mean(dim=1)
        cvar_95 = var_95 + F.softplus(port_raw[:, 1]) + 0.005 * pred_vol.max(dim=1).values

        return FinancialPredictionOutput(
            predicted_volatility=pred_vol,
            predicted_spread_change_bps=pred_spread,
            predicted_returns=pred_ret,
            portfolio_var_95=var_95,
            portfolio_cvar_95=cvar_95,
            volatility_lower_bound=vol_lower,
            volatility_upper_bound=vol_upper,
            spread_lower_bound=spread_lower,
            spread_upper_bound=spread_upper,
        )
