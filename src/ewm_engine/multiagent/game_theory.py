"""Game-theoretic analysis tools (Nash equilibria, replicator dynamics) using Nashpy.

Quarantined behind the [game] extra.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Sequence

import numpy as np

from ewm_engine.exceptions import SimulationConfigurationError


def is_nashpy_available() -> bool:
    """Check if Nashpy 2-player game-theory library is installed."""
    return importlib.util.find_spec("nashpy") is not None


class NashpyGameSolver:
    """Solver for 2-player strategic-form games and population dynamics."""

    def __init__(self) -> None:
        if not is_nashpy_available():
            raise SimulationConfigurationError(
                "Nashpy is required for NashpyGameSolver. Install via `pip install ewm-engine[game]`."
            )

    def solve_nash_equilibria(
        self,
        payoff_matrix_row: Sequence[Sequence[float]] | np.ndarray,
        payoff_matrix_col: Sequence[Sequence[float]] | np.ndarray,
        method: str = "support_enumeration",
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Compute Nash equilibria of a 2-player bi-matrix game.

        Args:
            payoff_matrix_row: Payoff matrix for row player (A).
            payoff_matrix_col: Payoff matrix for column player (B).
            method: Equilibrium algorithm ('support_enumeration' or 'vertex_enumeration').

        Returns:
            List of (sigma_row, sigma_col) mixed strategy probability vectors.
        """
        nash = importlib.import_module("nashpy")
        a = np.asarray(payoff_matrix_row, dtype=np.float64)
        b = np.asarray(payoff_matrix_col, dtype=np.float64)

        game = nash.Game(a, b)
        if method == "vertex_enumeration":
            eqs = list(game.vertex_enumeration())
        else:
            eqs = list(game.support_enumeration())

        return [
            (np.asarray(s1, dtype=np.float64), np.asarray(s2, dtype=np.float64)) for s1, s2 in eqs
        ]

    def compute_replicator_dynamics(
        self,
        payoff_matrix: Sequence[Sequence[float]] | np.ndarray,
        y0: Sequence[float] | np.ndarray,
        time_points: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Compute evolutionary replicator dynamics trajectory for a symmetric population.

        Args:
            payoff_matrix: Symmetric population payoff matrix.
            y0: Initial strategy population distribution (sums to 1.0).
            time_points: Monotonic array of continuous time points.

        Returns:
            Array of shape (len(time_points), num_strategies) population trajectory.
        """
        nash = importlib.import_module("nashpy")
        a = np.asarray(payoff_matrix, dtype=np.float64)
        game = nash.Game(a)
        y0_arr = np.asarray(y0, dtype=np.float64)
        t_arr = np.asarray(time_points, dtype=np.float64)

        trajectory = game.replicator_dynamics(y0=y0_arr, timepoints=t_arr)
        return np.asarray(trajectory, dtype=np.float64)
