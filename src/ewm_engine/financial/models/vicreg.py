"""VICReg, SIGReg, and RankMe regularization for JEPA latent representations.

Prevents representational collapse in Joint-Embedding Predictive Architectures
without requiring negative samples or raw observation decoders.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ewm_engine.routing.semantic_router import compute_rankme


@dataclass(frozen=True)
class VICRegLossComponents:
    """Decomposed loss terms from VICReg evaluation."""

    total_loss: torch.Tensor
    invariance_loss: torch.Tensor
    variance_loss: torch.Tensor
    covariance_loss: torch.Tensor


class VICRegLoss(nn.Module):
    """Variance-Invariance-Covariance Regularization (Bardes et al., ICLR 2022).

    Enforces:
    1. Invariance: MSE between predicted future embedding z_hat_{t+1} and target z_{t+1}.
    2. Variance: Hinge penalty keeping per-dimension batch standard deviation >= gamma.
    3. Covariance: Off-diagonal covariance penalty decorrelating feature dimensions.
    """

    def __init__(
        self,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.eps = eps

    def _variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        if z.size(0) <= 1:
            return torch.zeros((), dtype=z.dtype, device=z.device)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + self.eps)
        return torch.mean(F.relu(self.gamma - std))

    def _covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, num_features = z.shape
        if batch_size <= 1 or num_features <= 1:
            return torch.zeros((), dtype=z.dtype, device=z.device)
        z_centered = z - z.mean(dim=0, keepdim=True)
        cov = (z_centered.T @ z_centered) / float(batch_size - 1)
        diag_mask = ~torch.eye(num_features, dtype=torch.bool, device=z.device)
        off_diag_sq = cov[diag_mask].pow(2).sum()
        return off_diag_sq / float(num_features)

    def forward_components(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
    ) -> VICRegLossComponents:
        """Compute total VICReg loss along with individual invariance, variance, and covariance terms."""
        inv_loss = F.mse_loss(z_pred, z_target)
        var_loss = self._variance_loss(z_pred) + self._variance_loss(z_target)
        cov_loss = self._covariance_loss(z_pred) + self._covariance_loss(z_target)
        total = self.sim_coeff * inv_loss + self.std_coeff * var_loss + self.cov_coeff * cov_loss
        return VICRegLossComponents(
            total_loss=total,
            invariance_loss=inv_loss,
            variance_loss=var_loss,
            covariance_loss=cov_loss,
        )

    def forward(self, z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        """Compute scalar VICReg loss."""
        return self.forward_components(z_pred, z_target).total_loss


def sigreg_loss(
    z: torch.Tensor,
    num_projections: int = 32,
    seed: int = 42,
) -> torch.Tensor:
    """Sketched-Isotropic-Gaussian Regularization (SIGReg).

    Projects embeddings onto random unit directions and penalizes deviations from
    standard normal moments (zero mean, unit variance, zero skewness, zero excess kurtosis).
    """
    if z.size(0) <= 1:
        return torch.zeros((), dtype=z.dtype, device=z.device)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    directions = torch.randn(z.size(1), num_projections, generator=gen).to(
        device=z.device, dtype=z.dtype
    )
    directions = F.normalize(directions, p=2, dim=0)

    projected = z @ directions  # (B, M)
    mean = projected.mean(dim=0)
    centered = projected - mean.unsqueeze(0)
    var = centered.pow(2).mean(dim=0)
    std = torch.sqrt(var + 1e-6)
    normed = centered / std.unsqueeze(0)

    skewness = normed.pow(3).mean(dim=0)
    excess_kurtosis = normed.pow(4).mean(dim=0) - 3.0

    moment_penalty = (
        mean.pow(2).mean()
        + (std - 1.0).pow(2).mean()
        + 0.1 * skewness.pow(2).mean()
        + 0.05 * excess_kurtosis.pow(2).mean()
    )
    return moment_penalty


__all__ = [
    "VICRegLoss",
    "VICRegLossComponents",
    "compute_rankme",
    "sigreg_loss",
]
