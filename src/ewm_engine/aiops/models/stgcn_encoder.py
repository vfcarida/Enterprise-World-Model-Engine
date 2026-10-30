"""Spatial-Temporal Graph Convolutional Network (STGCN) encoder for AIOps telemetry.

Combines EvolveGCN (spatial microservice call-graph topology) with a multi-layer
GRU (temporal telemetry evolution across 1-minute steps) to map multi-node
telemetry tensors `(B, T, N, F)` into latent operational states `S_t`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ewm_engine.graph.evolvegcn import EvolveGCN


class STGCNEncoder(nn.Module):
    """Spatial-Temporal Graph Convolutional Network for telemetry state representation."""

    def __init__(
        self,
        in_channels: int = 5,  # (cpu, memory, error_rate, latency, connections)
        spatial_hidden: int = 64,
        temporal_hidden: int = 64,
        out_channels: int = 32,
        num_gcn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.spatial_hidden = spatial_hidden
        self.temporal_hidden = temporal_hidden
        self.out_channels = out_channels
        self.num_gcn_layers = num_gcn_layers

        # Spatial component: EvolveGCN
        self.evolvegcn = EvolveGCN(
            in_channels=in_channels,
            hidden_channels=spatial_hidden,
            out_channels=out_channels,
            num_layers=num_gcn_layers,
        )

        # Node-level residual projection preserving raw per-service telemetry signal
        self.node_input_proj = nn.Linear(in_channels, out_channels)

        # Temporal component: GRU over pooled graph sequence
        self.gru = nn.GRU(
            input_size=out_channels,
            hidden_size=temporal_hidden,
            num_layers=2,
            batch_first=True,
        )

        # Node-level temporal summary projection from [last_step, mean_step, slope_step, spatial_last]
        self.node_temporal_proj = nn.Sequential(
            nn.Linear(in_channels * 3 + out_channels, temporal_hidden),
            nn.GELU(),
            nn.Linear(temporal_hidden, out_channels),
        )

        self.fc_out = nn.Linear(temporal_hidden, out_channels)

    def encode_with_node_states(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode telemetry sequence into global latent state `S_t` and per-node states `H_t`.

        Args:
            x: Telemetry tensor of shape `(B, T, N, F)` or `(T, N, F)`.
            edge_index: Graph connectivity `(2, E)`.

        Returns:
            Tuple of:
                state_embedding: `(B, out_channels)` global operational state `S_t`
                node_embeddings: `(B, N, out_channels)` per-service spatial-temporal state `H_t`
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor (B, T, N, F), got shape {tuple(x.shape)}")

        b_sz, t_len, n_nodes, n_feats = x.shape
        if n_feats != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {n_feats}")
        if t_len == 0 or n_nodes == 0:
            raise ValueError("Telemetry tensor must have T > 0 and N > 0")

        x_norm = torch.arcsinh(x)

        batch_spatial_seqs: list[torch.Tensor] = []
        batch_last_spatial: list[torch.Tensor] = []

        for b_idx in range(b_sz):
            self.evolvegcn.reset_hidden_states()
            step_pooled: list[torch.Tensor] = []
            last_h = torch.zeros((n_nodes, self.out_channels), dtype=x.dtype, device=x.device)

            for t in range(t_len):
                x_bt = x_norm[b_idx, t]  # (N, F)
                h_spatial = self.evolvegcn(x_bt, edge_index)
                h_combined = self.node_input_proj(x_bt) + 0.5 * h_spatial
                last_h = h_combined
                step_pooled.append(h_combined.mean(dim=0))

            batch_spatial_seqs.append(torch.stack(step_pooled, dim=0))  # (T, out_channels)
            batch_last_spatial.append(last_h)  # (N, out_channels)

        spatial_sequence = torch.stack(batch_spatial_seqs, dim=0)  # (B, T, out_channels)
        last_spatial_nodes = torch.stack(batch_last_spatial, dim=0)  # (B, N, out_channels)

        gru_out, _ = self.gru(spatial_sequence)  # (B, T, temporal_hidden)
        final_state: torch.Tensor = self.fc_out(gru_out[:, -1, :])  # (B, out_channels)

        # Per-node temporal features: last state, temporal mean, and temporal drift/slope
        last_feat = x_norm[:, -1, :, :]
        mean_feat = x_norm.mean(dim=1)
        slope_feat = x_norm[:, -1, :, :] - x_norm[:, 0, :, :]
        node_summary = torch.cat(
            [last_feat, mean_feat, slope_feat, last_spatial_nodes],
            dim=-1,
        )
        node_states: torch.Tensor = self.node_temporal_proj(node_summary)  # (B, N, out_channels)

        return final_state, node_states

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Encode telemetry tensor `(B, T, N, F)` into final state representation `(B, out_channels)`."""
        final_state, _ = self.encode_with_node_states(x, edge_index)
        return final_state
