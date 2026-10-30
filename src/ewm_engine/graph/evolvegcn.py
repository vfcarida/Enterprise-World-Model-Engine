"""Continuous-Time Temporal Graph Neural Network (CT-TGNN) via EvolveGCN.

Evolves Graph Convolutional Network (GCN) weights dynamically over time using
recurrent units (GRU or LSTM) to model mutating enterprise topologies without
static adjacency matrix recomputation overhead.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNConv(nn.Module):
    """Symmetric Kipf & Welling Graph Convolutional layer with functional weight support.

    Compatible with `torch_geometric.nn.GCNConv` attribute layout (`self.lin.weight`,
    `self.bias`) while supporting differentiable recurrent weight evolution.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize linear projection weights and zero bias."""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply symmetrically normalized graph convolution.

        Args:
            x: Node feature matrix of shape (N, in_channels).
            edge_index: Graph connectivity in COO format of shape (2, E) or (E, 2).
            weight: Optional evolved weight tensor of shape (out_channels, in_channels).

        Returns:
            Updated node representations of shape (N, out_channels).
        """
        num_nodes = x.size(0)
        w = weight if weight is not None else self.lin.weight
        support: torch.Tensor = F.linear(x, w)

        if num_nodes == 0:
            return support + self.bias

        loop_idx = torch.arange(num_nodes, dtype=torch.long, device=x.device)
        if edge_index.numel() > 0:
            if edge_index.ndim == 2 and edge_index.size(0) != 2 and edge_index.size(1) == 2:
                edge_index = edge_index.t().contiguous()
            src = edge_index[0].to(dtype=torch.long, device=x.device)
            dst = edge_index[1].to(dtype=torch.long, device=x.device)
            non_self = src != dst
            row = torch.cat([src[non_self], loop_idx], dim=0)
            col = torch.cat([dst[non_self], loop_idx], dim=0)
        else:
            row = loop_idx
            col = loop_idx

        deg = torch.zeros(num_nodes, dtype=x.dtype, device=x.device)
        ones = torch.ones(row.size(0), dtype=x.dtype, device=x.device)
        deg.index_add_(0, col, ones)
        deg.index_add_(0, row, ones)
        deg = deg * 0.5
        deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        messages = support[row] * norm.unsqueeze(-1)
        out = torch.zeros((num_nodes, self.out_channels), dtype=x.dtype, device=x.device)
        out.index_add_(0, col, messages)
        return out + self.bias


class EvolveGCNLayer(nn.Module):
    """Single EvolveGCN layer with recurrent weight evolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        recurrent_type: str = "gru",  # 'gru' or 'lstm'
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # GCN layer (weights will be evolved by RNN)
        self.gcn = GCNConv(in_channels, out_channels)
        self.base_weight = nn.Parameter(self.gcn.lin.weight.detach().clone())

        # Recurrent unit to evolve GCN weights
        gcn_weight_dim = in_channels * out_channels
        self.recurrent_unit: nn.GRUCell | nn.LSTMCell
        if recurrent_type == "gru":
            self.recurrent_unit = nn.GRUCell(gcn_weight_dim, gcn_weight_dim)
        elif recurrent_type == "lstm":
            self.recurrent_unit = nn.LSTMCell(gcn_weight_dim, gcn_weight_dim)
        else:
            raise ValueError(f"Unknown recurrent_type: {recurrent_type}")

        self.recurrent_type = recurrent_type
        self.hidden_state: Any = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        hidden_state: Any = None,
    ) -> tuple[torch.Tensor, Any]:
        """Evolve GCN weights via recurrent cell and compute updated node embeddings.

        Args:
            x: Node feature matrix (N, in_channels).
            edge_index: Graph connectivity in COO format (2, E).
            hidden_state: Previous recurrent hidden state (optional).

        Returns:
            Tuple of (node_embeddings of shape (N, out_channels), new_hidden_state).
        """
        if hidden_state is None:
            current_weights = self.base_weight.detach().reshape(-1)
        else:
            current_weights = self.gcn.lin.weight.detach().reshape(-1)

        new_hidden: Any
        if self.recurrent_type == "gru":
            assert isinstance(self.recurrent_unit, nn.GRUCell)
            h_prev: torch.Tensor | None = None
            if isinstance(hidden_state, torch.Tensor):
                h_prev = (
                    hidden_state.detach().unsqueeze(0)
                    if hidden_state.ndim == 1
                    else hidden_state.detach()
                )
            new_h_gru: torch.Tensor = self.recurrent_unit(
                current_weights.unsqueeze(0), h_prev
            ).squeeze(0)
            new_hidden = new_h_gru
            evolved_weights = new_h_gru.view(self.out_channels, self.in_channels)
        else:
            assert isinstance(self.recurrent_unit, nn.LSTMCell)
            lstm_state: tuple[torch.Tensor, torch.Tensor] | None = None
            if (
                isinstance(hidden_state, tuple)
                and len(hidden_state) == 2
                and isinstance(hidden_state[0], torch.Tensor)
                and isinstance(hidden_state[1], torch.Tensor)
            ):
                h_in = (
                    hidden_state[0].detach().unsqueeze(0)
                    if hidden_state[0].ndim == 1
                    else hidden_state[0].detach()
                )
                c_in = (
                    hidden_state[1].detach().unsqueeze(0)
                    if hidden_state[1].ndim == 1
                    else hidden_state[1].detach()
                )
                lstm_state = (h_in, c_in)
            new_h_lstm, new_c_lstm = self.recurrent_unit(current_weights.unsqueeze(0), lstm_state)
            new_hidden = (new_h_lstm.squeeze(0), new_c_lstm.squeeze(0))
            evolved_weights = new_hidden[0].view(self.out_channels, self.in_channels)

        self.hidden_state = new_hidden
        self.gcn.lin.weight.data.copy_(evolved_weights.detach())

        # Preserve gradient flow to both recurrent unit and base GCN linear weight
        effective_weights = (
            evolved_weights
            + (self.base_weight - self.base_weight.detach())
            + (self.gcn.lin.weight - self.gcn.lin.weight.detach())
        )
        node_embeddings: torch.Tensor = self.gcn(x, edge_index, weight=effective_weights)

        return node_embeddings, new_hidden

    def reset_state(self) -> None:
        """Reset recurrent hidden state and restore GCN weights to learned base weights."""
        self.hidden_state = None
        self.gcn.lin.weight.data.copy_(self.base_weight.detach())


class EvolveGCN(nn.Module):
    """Multi-layer EvolveGCN for temporal graph learning."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        recurrent_type: str = "gru",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.recurrent_type = recurrent_type

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            self.layers.append(EvolveGCNLayer(in_dim, out_dim, recurrent_type))

        self.hidden_states: list[Any] = [None] * num_layers

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute contextualized temporal node embeddings.

        Args:
            x: (N, in_channels) node features.
            edge_index: (2, E) edge connectivity.

        Returns:
            (N, out_channels) final node embeddings.
        """
        h = x
        for i, layer in enumerate(self.layers):
            h, self.hidden_states[i] = layer(h, edge_index, self.hidden_states[i])
        return h

    def reset_hidden_states(self) -> None:
        """Reset all recurrent hidden states (call at start of new sequence)."""
        self.hidden_states = [None] * self.num_layers
        for layer in self.layers:
            if isinstance(layer, EvolveGCNLayer):
                layer.reset_state()
