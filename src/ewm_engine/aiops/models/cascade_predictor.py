"""Cascading microservice failure predictor and AIOps World Model.

Forecasts downstream microservice cascade failures 30-60 minutes before user-facing
SLO breach by simulating state transitions `S_t + A_t -> S_{t+1}` across the
microservice call graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ewm_engine.aiops.models.stgcn_encoder import STGCNEncoder
from ewm_engine.routing.schemas import CausalInterventionVector


@dataclass(frozen=True)
class CascadePredictionOutput:
    """Predicted cascading failure probabilities, lead times, and next latent state."""

    failure_probabilities: torch.Tensor  # (B, N) per-service failure probability in [0, 1]
    predicted_lead_time_minutes: torch.Tensor  # (B, N) advance warning lead time in minutes
    predicted_slo_degradation: torch.Tensor  # (B,) system-wide SLO impact in [0, 1]
    next_latent_state: torch.Tensor  # (B, latent_dim) predicted future latent state S_hat_{t+1}


class CascadeFailurePredictor(nn.Module):
    """Predicts cascading microservice failures 30-60 minutes in advance."""

    def __init__(
        self,
        num_services: int = 6,
        in_channels: int = 5,
        latent_dim: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.num_services = num_services
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Per-service anomaly & contagion feature encoder:
        # [direct_anomaly, drift_score, upstream_contagion, downstream_backpressure, intervention_shock]
        self.contagion_proj = nn.Linear(in_channels * 2 + 3, latent_dim)

        self.transition_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Per-service prediction head: [failure_logit, lead_time_raw]
        combined_node_dim = latent_dim * 3
        self.service_mlp = nn.Sequential(
            nn.Linear(combined_node_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2),
        )
        self.service_residual = nn.Linear(combined_node_dim, 2)

        # System-wide SLO degradation head
        self.slo_head = nn.Sequential(
            nn.Linear(latent_dim + num_services, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _compute_call_graph_propagation(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        services: Sequence[str],
        intervention: CausalInterventionVector | None,
    ) -> torch.Tensor:
        """Compute per-service telemetry drift, call-graph backpressure, and intervention shock."""
        b_sz, t_len, n_nodes, _ = x.shape
        device = x.device
        svc_to_idx = {s: idx for idx, s in enumerate(services[:n_nodes])}

        x_norm = torch.arcsinh(x)
        last_vals = x_norm[:, -1, :, :]  # (B, N, F)
        first_vals = x_norm[:, 0, :, :] if t_len > 1 else last_vals
        drift_vals = last_vals - first_vals  # Positive drift = rising CPU/mem/error/latency

        # Incipient anomaly intensity per node from positive drift and level
        node_stress = F.relu(drift_vals).mean(dim=-1) + 0.3 * F.relu(last_vals).mean(
            dim=-1
        )  # (B, N)

        intervention_shock = torch.zeros((b_sz, n_nodes), dtype=x.dtype, device=device)
        if intervention is not None:
            mag = float(intervention.magnitude) * abs(float(intervention.directional_sentiment))
            for entity in intervention.primary_target_entities:
                if entity in svc_to_idx:
                    intervention_shock[:, svc_to_idx[entity]] += mag
            for pathway in intervention.contagion_pathways:
                tgt = str(pathway.get("target", ""))
                w = float(pathway.get("weight", 0.5))
                if tgt in svc_to_idx:
                    intervention_shock[:, svc_to_idx[tgt]] += mag * w

        total_source_stress = node_stress + intervention_shock
        downstream_prop = torch.zeros_like(total_source_stress)
        upstream_backpressure = torch.zeros_like(total_source_stress)

        if edge_index.numel() > 0:
            src_indices = edge_index[0].tolist()
            dst_indices = edge_index[1].tolist()
            # 2-hop propagation across call graph:
            # caller -> callee (downstream load) and callee -> caller (upstream timeout backpressure)
            curr_stress = total_source_stress.clone()
            for hop in range(2):
                decay = 0.75 if hop == 0 else 0.45
                next_stress = curr_stress.clone()
                for u, v in zip(src_indices, dst_indices, strict=False):
                    if 0 <= u < n_nodes and 0 <= v < n_nodes and u != v:
                        downstream_prop[:, v] += decay * curr_stress[:, u]
                        upstream_backpressure[:, u] += decay * curr_stress[:, v]
                        next_stress[:, u] = torch.maximum(
                            next_stress[:, u], decay * curr_stress[:, v]
                        )
                        next_stress[:, v] = torch.maximum(
                            next_stress[:, v], decay * curr_stress[:, u]
                        )
                curr_stress = next_stress

        extra_channels = torch.stack(
            [downstream_prop, upstream_backpressure, intervention_shock],
            dim=-1,
        )  # (B, N, 3)
        raw_features = torch.cat([last_vals, drift_vals, extra_channels], dim=-1)  # (B, N, 2F + 3)
        proj: torch.Tensor = self.contagion_proj(raw_features)
        return proj

    def forward(
        self,
        s_t: torch.Tensor,
        node_states: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        services: Sequence[str] | None = None,
        intervention: CausalInterventionVector | None = None,
    ) -> CascadePredictionOutput:
        """Predict cascading microservice failures and lead time in minutes."""
        if x.ndim == 3:
            x = x.unsqueeze(0)
        b_sz, _, n_nodes, _ = x.shape
        active_services = (
            list(services) if services is not None else [f"service-{i}" for i in range(n_nodes)]
        )

        contagion_emb = self._compute_call_graph_propagation(
            x=x,
            edge_index=edge_index,
            services=active_services,
            intervention=intervention,
        )  # (B, N, latent_dim)

        global_contagion = contagion_emb.mean(dim=1)  # (B, latent_dim)
        s_hat_t1 = s_t + self.transition_mlp(torch.cat([s_t, global_contagion], dim=-1))

        s_expanded = s_hat_t1.unsqueeze(1).expand(b_sz, n_nodes, self.latent_dim)
        combined = torch.cat(
            [s_expanded, node_states, contagion_emb], dim=-1
        )  # (B, N, 3*latent_dim)

        raw_out = self.service_mlp(combined) + self.service_residual(combined)  # (B, N, 2)
        failure_probs = torch.sigmoid(raw_out[:, :, 0])

        # Map second head to [30.0, 60.0] minute advance warning range for predicted failures
        lead_time_minutes = 30.0 + 30.0 * torch.sigmoid(raw_out[:, :, 1])

        slo_in = torch.cat([s_hat_t1, failure_probs], dim=-1)
        slo_deg = torch.sigmoid(self.slo_head(slo_in)).squeeze(-1)

        return CascadePredictionOutput(
            failure_probabilities=failure_probs,
            predicted_lead_time_minutes=lead_time_minutes,
            predicted_slo_degradation=slo_deg,
            next_latent_state=s_hat_t1,
        )


class AIOpsWorldModel(nn.Module):
    """Complete STGCN + CascadeFailurePredictor state-transition engine for AIOps."""

    def __init__(
        self,
        num_services: int = 6,
        in_channels: int = 5,
        latent_dim: int = 32,
        services: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.num_services = num_services
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.services = (
            list(services)
            if services is not None
            else [f"service-{i}" for i in range(num_services)]
        )

        self.encoder = STGCNEncoder(
            in_channels=in_channels,
            spatial_hidden=64,
            temporal_hidden=64,
            out_channels=latent_dim,
            num_gcn_layers=2,
        )
        self.predictor = CascadeFailurePredictor(
            num_services=num_services,
            in_channels=in_channels,
            latent_dim=latent_dim,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        edge_index: torch.Tensor,
        intervention: CausalInterventionVector | None = None,
    ) -> tuple[torch.Tensor, CascadePredictionOutput]:
        """Encode telemetry window `x_t` and predict downstream cascade failures."""
        s_t, node_states = self.encoder.encode_with_node_states(x_t, edge_index)
        prediction = self.predictor(
            s_t=s_t,
            node_states=node_states,
            x=x_t,
            edge_index=edge_index,
            services=self.services,
            intervention=intervention,
        )
        return s_t, prediction

    def compute_training_loss(
        self,
        x_t: torch.Tensor,
        x_t1: torch.Tensor,
        edge_index: torch.Tensor,
        target_failures: torch.Tensor,
        target_lead_minutes: torch.Tensor,
        target_slo_degradation: torch.Tensor,
        intervention: CausalInterventionVector | None = None,
    ) -> torch.Tensor:
        """Compute joint state-transition consistency + cascade failure supervision loss."""
        _, pred = self.forward(x_t, edge_index, intervention=intervention)
        with torch.no_grad():
            s_target, _ = self.encoder.encode_with_node_states(x_t1, edge_index)

        latent_loss = F.mse_loss(pred.next_latent_state, s_target)
        bce_loss = F.binary_cross_entropy(pred.failure_probabilities, target_failures)
        lead_loss = F.l1_loss(pred.predicted_lead_time_minutes, target_lead_minutes)
        slo_loss = F.mse_loss(pred.predicted_slo_degradation, target_slo_degradation)

        total: torch.Tensor = 10.0 * bce_loss + 0.2 * lead_loss + 5.0 * slo_loss + 0.5 * latent_loss
        return total
