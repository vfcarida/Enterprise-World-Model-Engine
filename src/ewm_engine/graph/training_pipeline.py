"""Training pipeline connecting ContinuousGraphRegistry to EvolveGCN."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

from ewm_engine.graph.continuous_registry import ContinuousGraphRegistry
from ewm_engine.graph.evolvegcn import EvolveGCN

logger = logging.getLogger("ewm_engine.graph.training_pipeline")


class EvolveGCNTrainer:
    """Trains EvolveGCN on temporal graph snapshots from ContinuousGraphRegistry."""

    def __init__(
        self,
        registry: ContinuousGraphRegistry,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 32,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        recurrent_type: str = "gru",
    ) -> None:
        self.registry = registry
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.model = EvolveGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            recurrent_type=recurrent_type,
        )
        self.projection_head = nn.Linear(out_channels, in_channels)
        self.optimizer = torch.optim.Adam(
            [*self.model.parameters(), *self.projection_head.parameters()],
            lr=learning_rate,
        )
        self.loss_fn = nn.MSELoss()

    def train_on_snapshot_sequence(
        self,
        num_epochs: int = 10,
        sequence_length: int = 10,
    ) -> list[float]:
        """Train EvolveGCN on a temporal sequence of graph snapshots.

        Args:
            num_epochs: Number of optimization epochs over the snapshot sequence.
            sequence_length: Maximum number of trailing snapshots to train on.

        Returns:
            List of average MSE losses per epoch.
        """
        self.model.train()
        self.projection_head.train()
        epoch_losses: list[float] = []

        snapshots = self.registry.snapshot_history[-sequence_length:]
        if len(snapshots) < 2:
            return epoch_losses

        for epoch in range(num_epochs):
            self.model.reset_hidden_states()
            total_loss = 0.0
            steps = 0

            for i in range(len(snapshots) - 1):
                _, x_t_arr, a_t_arr = snapshots[i]
                _, x_t1_arr, _ = snapshots[i + 1]

                if x_t_arr.shape[0] == 0 or x_t1_arr.shape[0] == 0:
                    continue

                x_t = torch.from_numpy(np.ascontiguousarray(x_t_arr, dtype=np.float32))
                nonzero_rc = np.asarray(a_t_arr.nonzero(), dtype=np.int64)
                edge_index_t = torch.from_numpy(nonzero_rc).to(dtype=torch.long)

                x_t1 = torch.from_numpy(np.ascontiguousarray(x_t1_arr, dtype=np.float32))

                embeddings_t = self.model(x_t, edge_index_t)
                predicted_x_t1 = self.projection_head(embeddings_t)

                common_nodes = min(predicted_x_t1.size(0), x_t1.size(0))
                loss = self.loss_fn(predicted_x_t1[:common_nodes], x_t1[:common_nodes])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss.item())
                steps += 1

            avg_loss = total_loss / max(steps, 1)
            epoch_losses.append(avg_loss)
            logger.info("Epoch %d/%d, Loss: %.4f", epoch + 1, num_epochs, avg_loss)

        return epoch_losses
