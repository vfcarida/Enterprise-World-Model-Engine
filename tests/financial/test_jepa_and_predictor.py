"""Unit and property tests for FinancialJEPAEncoder, VICReg, TransitionPredictor, and Decoder."""
# ruff: noqa: E402

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st

torch = pytest.importorskip("torch")

from ewm_engine.financial.models import (
    FinancialJEPAEncoder,
    FinancialJEPAPair,
    FinancialMetricsDecoder,
    FinancialTransitionPredictor,
    FinancialWorldModel,
    VICRegLoss,
    augment_cross_stock_mask,
    augment_time_warp,
    compute_rankme,
    sigreg_loss,
)
from ewm_engine.graph.continuous_registry import ContinuousGraphRegistry, Edge, Node
from ewm_engine.routing.schemas import CausalInterventionVector


@pytest.mark.unit
def test_financial_jepa_encoder_and_augmentations() -> None:
    """Verify ViT-based FinancialJEPAEncoder, EMA target pair, and financial augmentations."""
    torch.manual_seed(42)
    encoder = FinancialJEPAEncoder(num_tickers=6, num_features=14, latent_dim=64)
    x = torch.randn(4, 15, 6, 14)

    gen = torch.Generator().manual_seed(42)
    x_warped = augment_time_warp(x, sigma=0.05, generator=gen)
    x_masked = augment_cross_stock_mask(x, mask_prob=0.2, generator=gen)
    assert x_warped.shape == x.shape
    assert x_masked.shape == x.shape

    z = encoder(x)
    assert z.shape == (4, 64)
    assert not torch.isnan(z).any()

    # 3D input (B, N, F) is also accepted
    z_3d = encoder(x[:, 0])
    assert z_3d.shape == (4, 64)

    # Invalid dimensions raise ValueError
    with pytest.raises(ValueError, match="Expected 4D tensor"):
        encoder(torch.randn(4, 14))
    with pytest.raises(ValueError, match="Expected \\(num_tickers=6"):
        encoder(torch.randn(4, 10, 5, 14))

    # Verify FinancialJEPAPair EMA update
    pair = FinancialJEPAPair(num_tickers=6, num_features=14, latent_dim=64, ema_momentum=0.5)
    for p in pair.context_encoder.parameters():
        p.data.add_(0.5)
    z_target_before = pair.encode_target(x).clone()
    pair.update_target_encoder()
    z_target_after = pair.encode_target(x)
    assert not torch.allclose(z_target_before, z_target_after)


@pytest.mark.unit
def test_financial_jepa_encoder_with_mocked_vit_pretrained() -> None:
    """Verify FinancialJEPAEncoder calls ViTModel.from_pretrained when mocked."""

    @dataclass
    class _DummyViTOut:
        last_hidden_state: torch.Tensor

    mock_vit = MagicMock()
    mock_vit.config = MagicMock(hidden_size=768)
    mock_vit.return_value = _DummyViTOut(
        last_hidden_state=torch.ones((2, 197, 768), dtype=torch.float32)
    )

    with patch(
        "ewm_engine.financial.models.jepa_encoder.ViTModel.from_pretrained",
        return_value=mock_vit,
    ) as mock_fp:
        enc = FinancialJEPAEncoder(
            num_tickers=4,
            num_features=10,
            latent_dim=32,
            vit_model="google/vit-base-patch16-224",
        )
        out = enc(torch.randn(2, 5, 4, 10))
        assert out.shape == (2, 32)
        mock_fp.assert_called_once_with("google/vit-base-patch16-224")


@pytest.mark.unit
def test_vicreg_and_sigreg_prevent_collapse() -> None:
    """Verify VICReg variance and covariance terms penalize collapsed and correlated embeddings."""
    vicreg = VICRegLoss(sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0, gamma=1.0)

    # Diverse isotropic embeddings have low variance and covariance loss
    torch.manual_seed(10)
    z_diverse_1 = torch.randn(32, 64) * 1.2
    z_diverse_2 = z_diverse_1 + torch.randn(32, 64) * 0.05
    comp_diverse = vicreg.forward_components(z_diverse_1, z_diverse_2)

    # Collapsed constant embeddings have maximum variance penalty
    z_collapsed = torch.zeros(32, 64)
    comp_collapsed = vicreg.forward_components(z_collapsed, z_collapsed)
    assert comp_collapsed.variance_loss > comp_diverse.variance_loss
    assert comp_collapsed.variance_loss > 1.5

    # Highly correlated feature dimensions trigger high covariance loss
    base_col = torch.randn(32, 1)
    z_correlated = base_col.repeat(1, 64)
    comp_corr = vicreg.forward_components(z_correlated, z_correlated)
    assert comp_corr.covariance_loss > comp_diverse.covariance_loss

    # SIGReg loss and RankMe check
    sig_val = sigreg_loss(z_diverse_1)
    assert sig_val.item() >= 0.0
    assert compute_rankme(z_diverse_1) > 15.0


@pytest.mark.unit
def test_transition_predictor_ct_tgnn_contagion_and_decoder() -> None:
    """Verify TransitionPredictor propagates causal shocks across supply-chain graph and decodes risk."""
    torch.manual_seed(42)
    tickers = ["TSM", "NVDA", "AMD", "AAPL"]
    registry = ContinuousGraphRegistry()
    for t in tickers:
        registry.add_node(Node(t, "equity", {"market_cap": 1000.0}))
    registry.add_edge(Edge("TSM", "NVDA", "supply_chain", 0.95))
    registry.add_edge(Edge("TSM", "AMD", "supply_chain", 0.85))

    predictor = FinancialTransitionPredictor(latent_dim=64, num_tickers=4, node_dim=16)
    decoder = FinancialMetricsDecoder(latent_dim=64, num_tickers=4, node_embedding_dim=16)

    z_t = torch.randn(1, 64)

    # Compare baseline transition (no shock) vs supply-chain shock on TSM
    predictor.evolvegcn.reset_hidden_states()
    z_no_shock, nodes_no_shock = predictor(
        z_t, intervention=None, tickers=tickers, registry=registry
    )

    shock = CausalInterventionVector(
        primary_target_entities=["TSM"],
        directional_sentiment=-0.95,
        magnitude=0.90,
        contagion_pathways=[{"source": "TSM", "target": "NVDA", "weight": 0.95}],
        confidence=0.92,
    )
    predictor.evolvegcn.reset_hidden_states()
    z_shock, nodes_shock = predictor(z_t, intervention=shock, tickers=tickers, registry=registry)

    assert not torch.allclose(z_no_shock, z_shock)
    # Contagion propagates to NVDA (index 1) even though TSM (index 0) was the primary target
    assert not torch.allclose(nodes_no_shock[0, 1], nodes_shock[0, 1])

    decoded = decoder(z_shock, nodes_shock)
    assert decoded.predicted_volatility.shape == (1, 4)
    assert torch.all(decoded.predicted_volatility > 0.0)
    assert torch.all(decoded.portfolio_cvar_95 >= decoded.portfolio_var_95)
    assert torch.all(decoded.volatility_upper_bound >= decoded.volatility_lower_bound)
    assert torch.all(decoded.spread_upper_bound >= decoded.spread_lower_bound)


@pytest.mark.property
@given(seed=st.integers(min_value=1, max_value=10000))
def test_property_jepa_deterministic_latent_transition(seed: int) -> None:
    """Property: Identical random seed and input produce bit-identical latent transition and metrics."""
    x = torch.ones((1, 4, 4, 14), dtype=torch.float32) * 0.5
    shock = CausalInterventionVector(
        primary_target_entities=["TICKER_0"],
        directional_sentiment=-0.5,
        magnitude=0.7,
        confidence=0.9,
    )

    torch.manual_seed(seed)
    m1 = FinancialWorldModel(num_tickers=4, num_features=14, latent_dim=32, node_dim=8)
    m1.eval()
    with torch.no_grad():
        z1, out1 = m1(x, intervention=shock)

    torch.manual_seed(seed)
    m2 = FinancialWorldModel(num_tickers=4, num_features=14, latent_dim=32, node_dim=8)
    m2.eval()
    with torch.no_grad():
        z2, out2 = m2(x, intervention=shock)

    assert torch.allclose(z1, z2, atol=1e-6)
    assert torch.allclose(out1.predicted_volatility, out2.predicted_volatility, atol=1e-6)
