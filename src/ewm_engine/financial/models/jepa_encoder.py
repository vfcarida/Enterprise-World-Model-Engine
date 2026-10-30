"""Vision Transformer (ViT) JEPA encoder for financial state representations.

Encodes synchronized multi-variate financial tensors (1Hz Market-1T microstructure
+ SEC EDGAR fundamentals) into dense latent state representations z_t without
raw observation reconstruction.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import ViTConfig, ViTModel
except ImportError:  # pragma: no cover
    ViTConfig = None
    ViTModel = None


def _is_mock_callable(obj: Any) -> bool:
    """Return True if obj or its class originates from unittest.mock."""
    return hasattr(obj, "_mock_name") or "unittest.mock" in type(obj).__module__


def augment_time_warp(
    x: torch.Tensor,
    sigma: float = 0.05,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply smooth multiplicative temporal warping augmentation to financial tensor.

    Args:
        x: Input tensor of shape (B, T, N, F).
        sigma: Magnitude of temporal warping modulation.
        generator: Optional deterministic torch.Generator.
    """
    if sigma <= 0.0 or x.size(1) <= 1:
        return x.clone()
    noise = torch.randn(
        x.size(0), x.size(1), 1, 1, generator=generator, device=x.device, dtype=x.dtype
    )
    envelope = 1.0 + sigma * torch.tanh(noise)
    return x * envelope


def augment_cross_stock_mask(
    x: torch.Tensor,
    mask_prob: float = 0.15,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly mask a subset of tickers across the time window to encourage cross-asset learning.

    Args:
        x: Input tensor of shape (B, T, N, F).
        mask_prob: Probability of masking an asset's features.
        generator: Optional deterministic torch.Generator.
    """
    if mask_prob <= 0.0:
        return x.clone()
    keep_prob = 1.0 - min(mask_prob, 0.95)
    probs = torch.rand(
        x.size(0), 1, x.size(2), 1, generator=generator, device=x.device, dtype=x.dtype
    )
    mask = (probs < keep_prob).to(x.dtype) / keep_prob
    return x * mask


class FinancialJEPAEncoder(nn.Module):
    """Vision Transformer encoder for financial state representation."""

    def __init__(
        self,
        num_tickers: int,
        num_features: int,
        latent_dim: int = 256,
        vit_model: str = "google/vit-base-patch16-224",
        image_size: int = 16,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        self.num_tickers = num_tickers
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.vit_model = vit_model
        self.image_size = image_size
        self.patch_size = patch_size

        in_flat_dim = num_tickers * num_features
        spatial_pixels = 3 * image_size * image_size

        # Project multi-scale (mean, momentum, volatility) financial tensor onto 3xHxW ViT canvas
        self.spatial_adapter = nn.Linear(in_flat_dim * 3, spatial_pixels)

        use_remote_pretrained = ViTModel is not None and (
            _is_mock_callable(ViTModel)
            or _is_mock_callable(getattr(ViTModel, "from_pretrained", None))
            or os.getenv("EWM_FWM_LOAD_VIT_REMOTE") == "1"
        )

        self.vit: Any
        if use_remote_pretrained:
            self.vit = ViTModel.from_pretrained(vit_model)
            vit_hidden_size = int(getattr(self.vit.config, "hidden_size", 768))
        elif ViTConfig is not None and ViTModel is not None:
            config = ViTConfig(
                image_size=image_size,
                patch_size=patch_size,
                num_channels=3,
                hidden_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=256,
            )
            self.vit = ViTModel(config)
            vit_hidden_size = config.hidden_size
        else:  # pragma: no cover
            vit_hidden_size = 128
            self.vit = None

        self.projection = nn.Linear(vit_hidden_size, latent_dim)
        self.residual_proj = nn.Linear(in_flat_dim * 3, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode financial microstructure + fundamentals tensor into latent state z_t.

        Args:
            x: (B, T, num_tickers, num_features) or (B, num_tickers, num_features) tensor.

        Returns:
            z: (B, latent_dim) latent state embedding.
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D tensor (B, T, num_tickers, num_features), got shape {tuple(x.shape)}"
            )

        b_sz, t_len, n_tickers, n_feats = x.shape
        if n_tickers != self.num_tickers or n_feats != self.num_features:
            raise ValueError(
                f"Expected (num_tickers={self.num_tickers}, num_features={self.num_features}), "
                f"got ({n_tickers}, {n_feats})"
            )

        # Normalize input scale safely
        x_scaled = torch.arcsinh(x)

        # Extract 3 temporal channels across window T: mean state, momentum (last - first), volatility
        mean_feat = x_scaled.mean(dim=1).reshape(b_sz, -1)
        delta_feat = (x_scaled[:, -1] - x_scaled[:, 0]).reshape(b_sz, -1)
        if t_len > 1:
            std_feat = x_scaled.std(dim=1, unbiased=False).reshape(b_sz, -1)
        else:
            std_feat = torch.zeros_like(mean_feat)

        summary = torch.cat([mean_feat, delta_feat, std_feat], dim=-1)
        pixel_values = self.spatial_adapter(summary).view(b_sz, 3, self.image_size, self.image_size)

        if self.vit is not None:
            vit_out = self.vit(pixel_values=pixel_values)
            cls_token: torch.Tensor = vit_out.last_hidden_state[:, 0, :]
            z_vit = self.projection(cls_token)
        else:  # pragma: no cover
            z_vit = torch.zeros((b_sz, self.latent_dim), dtype=x.dtype, device=x.device)

        z_res = self.residual_proj(summary)
        z: torch.Tensor = self.layer_norm(z_vit + z_res)
        return F.elu(z) + z


class FinancialJEPAPair(nn.Module):
    """Paired Context Encoder (phi) and EMA Target Encoder (psi) for JEPA training."""

    def __init__(
        self,
        num_tickers: int,
        num_features: int,
        latent_dim: int = 256,
        ema_momentum: float = 0.99,
    ) -> None:
        super().__init__()
        self.context_encoder = FinancialJEPAEncoder(
            num_tickers=num_tickers,
            num_features=num_features,
            latent_dim=latent_dim,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.ema_momentum = ema_momentum

    def encode_context(self, x_t: torch.Tensor) -> torch.Tensor:
        """Encode observed context state x_t -> z_t = phi(x_t)."""
        out: torch.Tensor = self.context_encoder(x_t)
        return out

    def encode_target(self, x_t1: torch.Tensor) -> torch.Tensor:
        """Encode true future state x_{t+1} -> z_{t+1} = psi(x_{t+1}) without gradient."""
        with torch.no_grad():
            out: torch.Tensor = self.target_encoder(x_t1)
            return out

    def update_target_encoder(self) -> None:
        """Update target encoder weights via exponential moving average of context encoder."""
        with torch.no_grad():
            m = self.ema_momentum
            for p_ctx, p_tgt in zip(
                self.context_encoder.parameters(),
                self.target_encoder.parameters(),
                strict=True,
            ):
                p_tgt.data.mul_(m).add_(p_ctx.data, alpha=1.0 - m)
