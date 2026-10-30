"""JEPA Transition Predictor + CT-TGNN (EvolveGCN) contagion propagation engine.

Implements S_t + A_t -> S_{t+1} in latent representation space:
1. Embeds structured CausalInterventionVector (A_t) from the Mechanistic Router.
2. Applies direct perturbations to primary target entities and propagates systemic
   contagion across supply-chain and sector edges via EvolveGCN.
3. Predicts future latent embedding z_hat_{t+1} = g_theta(z_t, A_t) via Transformer.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from ewm_engine.financial.models.financial_decoder import (
    FinancialMetricsDecoder,
    FinancialPredictionOutput,
)
from ewm_engine.financial.models.jepa_encoder import FinancialJEPAPair
from ewm_engine.financial.models.vicreg import VICRegLoss, VICRegLossComponents
from ewm_engine.graph.continuous_registry import ContinuousGraphRegistry
from ewm_engine.graph.evolvegcn import EvolveGCN
from ewm_engine.routing.schemas import CausalInterventionVector


class CausalInterventionEmbedder(nn.Module):
    """Embeds a structured CausalInterventionVector A_t into global and per-node tensors."""

    def __init__(
        self,
        num_tickers: int,
        action_dim: int = 64,
        node_feature_dim: int = 16,
    ) -> None:
        super().__init__()
        self.num_tickers = num_tickers
        self.action_dim = action_dim
        self.node_feature_dim = node_feature_dim

        # Input vector: [sentiment, magnitude, confidence, active_flag] + per-ticker target mask (N) + per-ticker incoming contagion (N)
        raw_dim = 4 + 2 * num_tickers
        self.global_proj = nn.Sequential(
            nn.Linear(raw_dim, action_dim),
            nn.GELU(),
            nn.LayerNorm(action_dim),
            nn.Linear(action_dim, action_dim),
        )
        # Per-node perturbation projection from 7 causal shock features
        self.node_shock_proj = nn.Linear(7, node_feature_dim)
        self.ticker_emb = nn.Parameter(torch.zeros(num_tickers, node_feature_dim))

    def encode_intervention(
        self,
        intervention: CausalInterventionVector | None,
        tickers: Sequence[str],
        batch_size: int = 1,
        device: torch.device | str = "cpu",
        registry: ContinuousGraphRegistry | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int, float]]]:
        """Convert CausalInterventionVector into global action embedding and node perturbations.

        Returns:
            Tuple of:
                action_emb: (B, action_dim) global intervention embedding
                node_perturbations: (N, node_feature_dim) direct node-level shock tensor
                extra_edges: List of (src_idx, tgt_idx, weight) from contagion_pathways
        """
        n_tickers = self.num_tickers
        ticker_to_idx = {t: idx for idx, t in enumerate(tickers[:n_tickers])}

        if intervention is None:
            raw_vec = torch.zeros(
                (batch_size, 4 + 2 * n_tickers), dtype=torch.float32, device=device
            )
            node_raw = torch.zeros((n_tickers, 7), dtype=torch.float32, device=device)
            return self.global_proj(raw_vec), self.node_shock_proj(node_raw) + self.ticker_emb, []

        sentiment = float(intervention.directional_sentiment)
        magnitude = float(intervention.magnitude)
        confidence = float(intervention.confidence)

        target_mask = torch.zeros(n_tickers, dtype=torch.float32, device=device)
        for entity in intervention.primary_target_entities:
            if entity in ticker_to_idx:
                target_mask[ticker_to_idx[entity]] = 1.0

        contagion_in = torch.zeros(n_tickers, dtype=torch.float32, device=device)
        contagion_out = torch.zeros(n_tickers, dtype=torch.float32, device=device)
        extra_edges: list[tuple[int, int, float]] = []
        for pathway in intervention.contagion_pathways:
            src_name = str(pathway.get("source", ""))
            tgt_name = str(pathway.get("target", ""))
            weight = float(pathway.get("weight", 0.5))
            if tgt_name in ticker_to_idx:
                t_idx = ticker_to_idx[tgt_name]
                contagion_in[t_idx] += weight * magnitude * abs(sentiment)
                if src_name in ticker_to_idx:
                    s_idx = ticker_to_idx[src_name]
                    contagion_out[s_idx] += weight * magnitude * abs(sentiment)
                    extra_edges.append((s_idx, t_idx, weight))

        if registry is not None:
            base_shock = target_mask * magnitude * abs(sentiment) + contagion_in
            topo_contagion = torch.zeros(n_tickers, dtype=torch.float32, device=device)
            for (src_id, tgt_id), edge in registry.edges.items():
                if src_id in ticker_to_idx and tgt_id in ticker_to_idx:
                    s_idx = ticker_to_idx[src_id]
                    t_idx = ticker_to_idx[tgt_id]
                    if s_idx != t_idx and base_shock[s_idx] > 0:
                        flow = float(edge.weight) * base_shock[s_idx] * 0.5
                        topo_contagion[t_idx] += flow
                        contagion_out[s_idx] += flow * 0.5
            contagion_in = contagion_in + topo_contagion

        scalars = torch.tensor(
            [sentiment, magnitude, confidence, 1.0], dtype=torch.float32, device=device
        )
        single_raw = torch.cat([scalars, target_mask, contagion_in], dim=0)
        batch_raw = single_raw.unsqueeze(0).expand(batch_size, -1)

        global_signed = torch.full(
            (n_tickers,), sentiment * magnitude, dtype=torch.float32, device=device
        )
        global_unsigned = torch.full(
            (n_tickers,), magnitude * confidence, dtype=torch.float32, device=device
        )

        node_raw = torch.stack(
            [
                target_mask * magnitude * confidence,
                target_mask * sentiment * magnitude,
                contagion_in,
                contagion_in * sentiment,
                contagion_out,
                global_signed,
                global_unsigned,
            ],
            dim=-1,
        )
        return (
            self.global_proj(batch_raw),
            self.node_shock_proj(node_raw) + self.ticker_emb,
            extra_edges,
        )


class FinancialTransitionPredictor(nn.Module):
    """Transformer-based JEPA predictor g_theta(z_t, A_t) -> z_hat_{t+1} with EvolveGCN contagion."""

    def __init__(
        self,
        latent_dim: int = 256,
        num_tickers: int = 8,
        action_dim: int = 64,
        node_dim: int = 32,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_tickers = num_tickers
        self.action_dim = action_dim
        self.node_dim = node_dim

        self.intervention_embedder = CausalInterventionEmbedder(
            num_tickers=num_tickers,
            action_dim=action_dim,
            node_feature_dim=node_dim,
        )

        # Project z_t to per-node initial features before EvolveGCN propagation
        self.latent_to_nodes = nn.Linear(latent_dim, num_tickers * node_dim)
        self.evolvegcn = EvolveGCN(
            in_channels=node_dim,
            hidden_channels=node_dim,
            out_channels=node_dim,
            num_layers=2,
            recurrent_type="gru",
        )

        # Transformer predictor tokens: [z_token, action_token, graph_summary_token]
        self.action_to_latent = nn.Linear(action_dim, latent_dim)
        self.graph_to_latent = nn.Linear(num_tickers * node_dim, latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.out_norm = nn.LayerNorm(latent_dim)

    def _build_edge_index(
        self,
        registry: ContinuousGraphRegistry | None,
        tickers: Sequence[str],
        extra_edges: Sequence[tuple[int, int, float]],
        device: torch.device | str,
    ) -> torch.Tensor:
        ticker_to_idx = {t: idx for idx, t in enumerate(tickers[: self.num_tickers])}
        edges: set[tuple[int, int]] = set()

        if registry is not None:
            for src_id, tgt_id in registry.edges:
                if src_id in ticker_to_idx and tgt_id in ticker_to_idx:
                    edges.add((ticker_to_idx[src_id], ticker_to_idx[tgt_id]))

        for s_idx, t_idx, _ in extra_edges:
            if 0 <= s_idx < self.num_tickers and 0 <= t_idx < self.num_tickers:
                edges.add((s_idx, t_idx))

        if not edges:
            return torch.zeros((2, 0), dtype=torch.long, device=device)

        src_list = [e[0] for e in sorted(edges)]
        tgt_list = [e[1] for e in sorted(edges)]
        return torch.tensor([src_list, tgt_list], dtype=torch.long, device=device)

    def forward(
        self,
        z_t: torch.Tensor,
        intervention: CausalInterventionVector | None = None,
        tickers: Sequence[str] | None = None,
        registry: ContinuousGraphRegistry | None = None,
        edge_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict future latent embedding z_hat_{t+1} and propagated node embeddings H_{t+1}.

        Args:
            z_t: Context latent embedding (B, latent_dim).
            intervention: Optional CausalInterventionVector A_t.
            tickers: Ordered ticker universe of length num_tickers.
            registry: Optional ContinuousGraphRegistry holding supply-chain/sector topology.
            edge_index: Optional explicit (2, E) edge_index tensor.

        Returns:
            Tuple of:
                z_hat_t1: (B, latent_dim) predicted next-state latent embedding
                propagated_nodes: (B, num_tickers, node_dim) CT-TGNN propagated node states
        """
        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        b_sz = z_t.size(0)
        device = z_t.device
        active_tickers = (
            list(tickers)
            if tickers is not None
            else [f"TICKER_{i}" for i in range(self.num_tickers)]
        )

        action_emb, node_perturbation, extra_edges = self.intervention_embedder.encode_intervention(
            intervention=intervention,
            tickers=active_tickers,
            batch_size=b_sz,
            device=device,
            registry=registry,
        )

        if edge_index is None:
            edge_index = self._build_edge_index(
                registry=registry,
                tickers=active_tickers,
                extra_edges=extra_edges,
                device=device,
            )

        base_nodes = self.latent_to_nodes(z_t).view(b_sz, self.num_tickers, self.node_dim)
        perturbed_nodes = base_nodes + node_perturbation.unsqueeze(0)

        propagated_list: list[torch.Tensor] = []
        for b_idx in range(b_sz):
            h_nodes = self.evolvegcn(perturbed_nodes[b_idx], edge_index)
            propagated_list.append(h_nodes)
        propagated_nodes = perturbed_nodes + 0.5 * torch.stack(
            propagated_list, dim=0
        )  # (B, N, node_dim)

        a_token = self.action_to_latent(action_emb)
        g_token = self.graph_to_latent(propagated_nodes.reshape(b_sz, -1))

        token_seq = torch.stack([z_t, a_token, g_token], dim=1)  # (B, 3, latent_dim)
        transformed = self.transformer(token_seq)
        delta_z = transformed[:, 0, :]
        z_hat_t1: torch.Tensor = self.out_norm(z_t + delta_z)

        return z_hat_t1, propagated_nodes


class FinancialWorldModel(nn.Module):
    """Full Financial World Model combining JEPA, VICReg, CT-TGNN, and MLP Decoder."""

    def __init__(
        self,
        num_tickers: int = 8,
        num_features: int = 14,
        latent_dim: int = 128,
        node_dim: int = 32,
        tickers: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.num_tickers = num_tickers
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.node_dim = node_dim
        self.tickers = (
            list(tickers) if tickers is not None else [f"TICKER_{i}" for i in range(num_tickers)]
        )

        self.jepa_pair = FinancialJEPAPair(
            num_tickers=num_tickers,
            num_features=num_features,
            latent_dim=latent_dim,
        )
        self.predictor = FinancialTransitionPredictor(
            latent_dim=latent_dim,
            num_tickers=num_tickers,
            node_dim=node_dim,
        )
        self.decoder = FinancialMetricsDecoder(
            latent_dim=latent_dim,
            num_tickers=num_tickers,
            node_embedding_dim=node_dim,
        )
        self.vicreg = VICRegLoss()

    def forward(
        self,
        x_t: torch.Tensor,
        intervention: CausalInterventionVector | None = None,
        registry: ContinuousGraphRegistry | None = None,
        edge_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, FinancialPredictionOutput]:
        """Simulate state transition S_t + A_t -> S_{t+1} and decode financial metrics."""
        z_t = self.jepa_pair.encode_context(x_t)
        self.predictor.evolvegcn.reset_hidden_states()
        z_hat_t1, node_embeddings = self.predictor(
            z_t=z_t,
            intervention=intervention,
            tickers=self.tickers,
            registry=registry,
            edge_index=edge_index,
        )
        metrics = self.decoder(z_hat_t1, node_embeddings)
        return z_hat_t1, metrics

    def compute_training_loss(
        self,
        x_t: torch.Tensor,
        x_t1: torch.Tensor,
        target_volatility: torch.Tensor,
        target_spread_bps: torch.Tensor,
        intervention: CausalInterventionVector | None = None,
        registry: ContinuousGraphRegistry | None = None,
    ) -> tuple[torch.Tensor, VICRegLossComponents]:
        """Compute joint JEPA VICReg latent loss + supervised financial metric decoding loss."""
        z_t = self.jepa_pair.encode_context(x_t)
        z_target = self.jepa_pair.encode_target(x_t1)

        self.predictor.evolvegcn.reset_hidden_states()
        z_hat_t1, node_embeddings = self.predictor(
            z_t=z_t,
            intervention=intervention,
            tickers=self.tickers,
            registry=registry,
        )
        decoded = self.decoder(z_hat_t1, node_embeddings)

        vicreg_components = self.vicreg.forward_components(z_hat_t1, z_target)
        vol_loss = nn.functional.l1_loss(decoded.predicted_volatility, target_volatility)
        spread_loss = nn.functional.l1_loss(decoded.predicted_spread_change_bps, target_spread_bps)
        # Interval calibration loss (pinball / coverage width alignment)
        err_vol = torch.abs(target_volatility - decoded.predicted_volatility).detach()
        width_vol = decoded.volatility_upper_bound - decoded.predicted_volatility
        calib_loss = nn.functional.mse_loss(width_vol, err_vol * 1.8 + 0.05)

        total_loss = (
            0.05 * vicreg_components.total_loss
            + 25.0 * vol_loss
            + 2.0 * spread_loss
            + 5.0 * calib_loss
        )
        return total_loss, vicreg_components
