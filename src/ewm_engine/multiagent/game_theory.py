"""Game-theoretic analysis tools (Nash equilibria, replicator dynamics) using Nashpy.

Quarantined behind the [game] extra.
"""

from __future__ import annotations

import importlib
import importlib.util
import itertools
from collections.abc import Sequence

import numpy as np

from ewm_engine.exceptions import SimulationConfigurationError


def is_nashpy_available() -> bool:
    """Check if Nashpy 2-player game-theory library is installed."""
    return importlib.util.find_spec("nashpy") is not None


class NashpyGameSolver:
    """Solver for 2-player strategic-form games and population dynamics."""

    def __init__(self, require_extra: bool = False) -> None:
        if require_extra and not is_nashpy_available():
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
        a = np.asarray(payoff_matrix_row, dtype=np.float64)
        b = np.asarray(payoff_matrix_col, dtype=np.float64)

        if is_nashpy_available():
            nash = importlib.import_module("nashpy")
            game = nash.Game(a, b)
            if method == "vertex_enumeration":
                eqs = list(game.vertex_enumeration())
            else:
                eqs = list(game.support_enumeration())

            return [
                (np.asarray(s1, dtype=np.float64), np.asarray(s2, dtype=np.float64))
                for s1, s2 in eqs
            ]

        return self._solve_nash_numpy(a, b)

    def _solve_nash_numpy(
        self, a: np.ndarray, b: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Pure-NumPy support enumeration solver for 2-player bi-matrix games."""
        m, n = a.shape
        equilibria: list[tuple[np.ndarray, np.ndarray]] = []

        for k in range(1, min(m, n) + 1):
            for supp_r in itertools.combinations(range(m), k):
                for supp_c in itertools.combinations(range(n), k):
                    sr = list(supp_r)
                    sc = list(supp_c)

                    # Solve for column strategy sigma_c on support sc making row indifferent on sr
                    mat_r = np.ones((k + 1, k + 1), dtype=np.float64)
                    mat_r[:k, :k] = a[np.ix_(sr, sc)]
                    mat_r[:k, k] = -1.0
                    mat_r[k, k] = 0.0
                    rhs = np.zeros(k + 1, dtype=np.float64)
                    rhs[k] = 1.0

                    # Solve for row strategy sigma_r on support sr making col indifferent on sc
                    mat_c = np.ones((k + 1, k + 1), dtype=np.float64)
                    mat_c[:k, :k] = b[np.ix_(sr, sc)].T
                    mat_c[:k, k] = -1.0
                    mat_c[k, k] = 0.0

                    try:
                        sol_c = np.linalg.solve(mat_r, rhs)[:k]
                        sol_r = np.linalg.solve(mat_c, rhs)[:k]
                    except np.linalg.LinAlgError:
                        continue

                    if np.any(sol_r < -1e-7) or np.any(sol_c < -1e-7):
                        continue

                    sol_r = np.maximum(sol_r, 0.0)
                    sol_c = np.maximum(sol_c, 0.0)
                    if sol_r.sum() <= 1e-12 or sol_c.sum() <= 1e-12:
                        continue
                    sol_r /= sol_r.sum()
                    sol_c /= sol_c.sum()

                    sigma_r = np.zeros(m, dtype=np.float64)
                    sigma_c = np.zeros(n, dtype=np.float64)
                    sigma_r[sr] = sol_r
                    sigma_c[sc] = sol_c

                    # Best-response verification across all pure strategies
                    row_payoffs = a @ sigma_c
                    col_payoffs = b.T @ sigma_r
                    expected_r = float(sigma_r @ row_payoffs)
                    expected_c = float(sigma_c @ col_payoffs)

                    if (
                        float(np.max(row_payoffs)) <= expected_r + 1e-6
                        and float(np.max(col_payoffs)) <= expected_c + 1e-6
                    ):
                        is_dup = any(
                            np.allclose(sigma_r, er, atol=1e-5)
                            and np.allclose(sigma_c, ec, atol=1e-5)
                            for er, ec in equilibria
                        )
                        if not is_dup:
                            equilibria.append((sigma_r, sigma_c))

        return equilibria

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
        a = np.asarray(payoff_matrix, dtype=np.float64)
        y0_arr = np.asarray(y0, dtype=np.float64)
        t_arr = np.asarray(time_points, dtype=np.float64)

        if is_nashpy_available():
            nash = importlib.import_module("nashpy")
            game = nash.Game(a)
            trajectory = game.replicator_dynamics(y0=y0_arr, timepoints=t_arr)
            return np.asarray(trajectory, dtype=np.float64)

        return self._replicator_dynamics_numpy(a, y0_arr, t_arr)

    def _replicator_dynamics_numpy(
        self, a: np.ndarray, y0: np.ndarray, time_points: np.ndarray
    ) -> np.ndarray:
        """Pure-NumPy 4th-order Runge-Kutta integrator for symmetric replicator dynamics."""
        n_steps = len(time_points)
        n_strat = len(y0)
        traj = np.zeros((n_steps, n_strat), dtype=np.float64)
        y = y0.copy()
        y = np.maximum(y, 0.0)
        y /= y.sum()
        traj[0] = y

        def rhs(state: np.ndarray) -> np.ndarray:
            fitness = a @ state
            avg_fitness = float(state @ fitness)
            return np.asarray(state * (fitness - avg_fitness), dtype=np.float64)

        substeps = 10
        for idx in range(1, n_steps):
            dt_total = float(time_points[idx] - time_points[idx - 1])
            dt = dt_total / substeps
            for _ in range(substeps):
                k1 = rhs(y)
                k2 = rhs(y + 0.5 * dt * k1)
                k3 = rhs(y + 0.5 * dt * k2)
                k4 = rhs(y + dt * k3)
                y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                y = np.maximum(y, 0.0)
                total = float(y.sum())
                if total > 0.0:
                    y /= total
            traj[idx] = y

        return traj
