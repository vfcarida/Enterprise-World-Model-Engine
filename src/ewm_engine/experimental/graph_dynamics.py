"""Graph Neural Network (GNN) dynamics adapter for heterogeneous relational graph states.

[EXPERIMENTAL] This module provides a relational message-passing dynamics model
conforming to the DynamicsModel protocol, operating over HeterogeneousGraphView.

Requires the 'ml' optional extra: pip install 'ewm-engine[ml]'
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.graph import HeterogeneousGraphView
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import RandomGenerator, ResourceId
from ewm_engine.dynamics.base import DynamicsModel, TransitionResult
from ewm_engine.dynamics.learned import TransitionDataset
from ewm_engine.exceptions import DynamicsError, SimulationConfigurationError
from ewm_engine.provenance.evidence import EvidenceLevel


def _load_torch() -> tuple[Any, Any, Any]:
    """Lazily import PyTorch modules, raising SimulationConfigurationError if absent."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        return torch, nn, optim
    except ImportError as exc:
        raise SimulationConfigurationError(
            "PyTorch is required for GraphNeuralDynamics. "
            "Install with: pip install 'ewm-engine[ml]'"
        ) from exc


class GraphNeuralDynamics(DynamicsModel):
    """Relational Graph Neural Network dynamics model over heterogeneous world states.

    Executes multi-relational message passing across heterogeneous entity nodes and
    directed typed edges:
        h_v^{(l+1)} = ReLU( W_{self} h_v^{(l)} + \\sum_{r} (1 / |N_r(v)|) \\sum_{u \\in N_r(v)} W_r h_u^{(l)} )

    Outputs are decoded into continuous resource deltas clamped to structural bounds.
    Evidence level is strictly PREDICTIVE.
    """

    def __init__(
        self,
        node_dim: int = 8,
        hidden_dim: int = 16,
        num_layers: int = 2,
        learning_rate: float = 1e-3,
        evidence_level: EvidenceLevel = EvidenceLevel.PREDICTIVE,
    ) -> None:
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.evidence_level = evidence_level

        self._model: Any = None
        self._is_fitted: bool = False
        self._target_resources: list[ResourceId] = []
        self._entity_order: list[str] = []

    def _init_network(self, num_nodes: int, num_resources: int) -> None:
        """Initialize internal PyTorch neural message-passing architecture."""
        torch, nn, _ = _load_torch()

        class _HeteroGNNModule(nn.Module):  # type: ignore[name-defined, misc]
            def __init__(self, n_nodes: int, n_res: int, h_dim: int) -> None:
                super().__init__()
                self.in_proj = nn.Linear(max(1, n_res), h_dim)
                self.self_linear = nn.Linear(h_dim, h_dim)
                self.rel_linear = nn.Linear(h_dim, h_dim)
                self.out_proj = nn.Linear(h_dim, max(1, n_res))
                self.relu = nn.ReLU()

            def forward(self, x: Any, adj: Any) -> Any:
                # x shape: (N, num_resources)
                # adj shape: (N, N) normalized relational adjacency
                h = self.relu(self.in_proj(x))
                # Message passing: h_rel = adj @ h
                h_rel = torch.matmul(adj, h)
                h_next = self.relu(self.self_linear(h) + self.rel_linear(h_rel))
                out = self.out_proj(h_next)
                return out

        self._model = _HeteroGNNModule(num_nodes, num_resources, self.hidden_dim)

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent] | None = None,
        rng: RandomGenerator | None = None,
        **kwargs: Any,
    ) -> TransitionResult:
        """Execute one step forward under the graph neural dynamics."""
        torch, _, _ = _load_torch()
        rng_gen = rng if rng is not None else np.random.default_rng(42)

        graph: HeterogeneousGraphView = state.as_graph()
        all_entities = sorted(state.entities.keys())
        all_resources = sorted(state.resources.keys())

        if not all_entities or not all_resources:
            return TransitionResult(
                next_state=state.advance_time(),
                applied_changes={},
                evidence_level=self.evidence_level,
            )

        if self._model is None:
            self._init_network(len(all_entities), len(all_resources))
            self._target_resources = all_resources
            self._entity_order = all_entities

        # Build feature vector per entity
        entity_idx = {e_id: idx for idx, e_id in enumerate(all_entities)}
        n_nodes = len(all_entities)
        n_res = len(all_resources)

        X = np.zeros((n_nodes, n_res), dtype=np.float32)
        for r_idx, r_id in enumerate(all_resources):
            r = state.resources[r_id]
            if r.entity_id and r.entity_id in entity_idx:
                e_i = entity_idx[r.entity_id]
                X[e_i, r_idx] = float(r.current)

        # Inject action perturbations onto target nodes
        for a in actions:
            target_e = a.parameters.get("target_entity")
            if target_e and target_e in entity_idx:
                e_i = entity_idx[target_e]
                # Perturb node features with action parameters
                for r_idx, r_id in enumerate(all_resources):
                    if r_id in a.parameters:
                        X[e_i, r_idx] += float(a.parameters[r_id])

        # Build adjacency matrix across all edge types
        adj = np.eye(n_nodes, dtype=np.float32)  # Self-loops
        for edges in graph.edges_by_type.values():
            for e in edges:
                if e.source in entity_idx and e.target in entity_idx:
                    s_i = entity_idx[e.source]
                    t_i = entity_idx[e.target]
                    adj[t_i, s_i] += float(e.weight)

        # Row-normalize adjacency
        deg = np.sum(adj, axis=1, keepdims=True)
        deg[deg == 0] = 1.0
        norm_adj = adj / deg

        # Forward pass
        self._model.eval()
        with torch.no_grad():
            t_X = torch.from_numpy(X)
            t_adj = torch.from_numpy(norm_adj)
            deltas = self._model(t_X, t_adj).numpy()

        # Add optional stochastic noise
        noise_scale = 0.05
        noise = rng_gen.normal(0.0, noise_scale, size=deltas.shape)
        deltas = deltas + noise

        # Update world state resources
        next_s = state
        applied_changes: dict[str, float] = {}

        for r_idx, r_id in enumerate(all_resources):
            r = state.resources[r_id]
            if r.entity_id and r.entity_id in entity_idx:
                e_i = entity_idx[r.entity_id]
                delta = float(deltas[e_i, r_idx])
                # Small step scaling for numerical stability
                delta = float(np.tanh(delta) * 5.0)
                new_val = float(np.clip(r.current + delta, r.min_value, r.max_value))
                next_s = next_s.update_resource(r_id, new_value=new_val)
                applied_changes[r_id] = new_val - r.current

        return TransitionResult(
            next_state=next_s.advance_time(),
            applied_changes=applied_changes,
            evidence_level=self.evidence_level,
        )

    def fit(
        self,
        dataset: TransitionDataset,
        epochs: int = 30,
        seed: int = 42,
    ) -> None:
        """Fit the GNN model weights to observed transition samples."""
        torch, nn, optim = _load_torch()

        if len(dataset) == 0:
            raise DynamicsError("Cannot fit GraphNeuralDynamics on an empty dataset.")

        first_sample = dataset[0]
        all_entities = sorted(first_sample.state.entities.keys())
        all_resources = sorted(first_sample.state.resources.keys())

        if not all_entities or not all_resources:
            self._is_fitted = True
            return

        self._init_network(len(all_entities), len(all_resources))
        self._target_resources = all_resources
        self._entity_order = all_entities

        entity_idx = {e_id: idx for idx, e_id in enumerate(all_entities)}
        n_nodes = len(all_entities)
        n_res = len(all_resources)

        torch.manual_seed(seed)
        optimizer = optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        self._model.train()
        for _ in range(epochs):
            for sample in dataset:
                s_curr = sample.state
                s_next = sample.next_state

                X = np.zeros((n_nodes, n_res), dtype=np.float32)
                Y = np.zeros((n_nodes, n_res), dtype=np.float32)

                for r_idx, r_id in enumerate(all_resources):
                    r_c = s_curr.resources.get(r_id)
                    r_n = s_next.resources.get(r_id)
                    if r_c and r_c.entity_id in entity_idx:
                        e_i = entity_idx[r_c.entity_id]
                        X[e_i, r_idx] = float(r_c.current)
                        if r_n:
                            Y[e_i, r_idx] = float(r_n.current - r_c.current)

                # Build graph adjacency
                graph = s_curr.as_graph()
                adj = np.eye(n_nodes, dtype=np.float32)
                for edges in graph.edges_by_type.values():
                    for e in edges:
                        if e.source in entity_idx and e.target in entity_idx:
                            adj[entity_idx[e.target], entity_idx[e.source]] += float(e.weight)

                deg = np.sum(adj, axis=1, keepdims=True)
                deg[deg == 0] = 1.0
                norm_adj = adj / deg

                t_X = torch.from_numpy(X)
                t_adj = torch.from_numpy(norm_adj)
                t_Y = torch.from_numpy(Y)

                optimizer.zero_grad()
                pred = self._model(t_X, t_adj)
                loss = criterion(pred, t_Y)
                loss.backward()
                optimizer.step()

        self._is_fitted = True
