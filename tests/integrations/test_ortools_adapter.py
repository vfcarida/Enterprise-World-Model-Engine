"""Unit tests for ORToolsAllocationAdapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.integrations.ortools import ORToolsAllocationAdapter


def test_ortools_missing_dependency_raises_configuration_error() -> None:
    """Ensure attempting to optimize without OR-Tools raises informative error."""
    adapter = ORToolsAllocationAdapter()
    adapter._pywraplp = None  # Ensure mock missing state

    state = WorldState(
        resources=[
            Resource(id="stock_a", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="stock_b", current=0.0, min_value=0.0, max_value=200.0),
        ]
    )

    with pytest.raises(SimulationConfigurationError) as exc_info:
        adapter.optimize_transfers(
            state=state,
            sources=["stock_a"],
            destinations=["stock_b"],
            demands={"stock_b": 50.0},
        )
    assert "Google OR-Tools is required" in str(exc_info.value)


def test_ortools_optimization_with_mock_solver() -> None:
    """Ensure adapter correctly builds MILP variables and translates solution to Actions."""
    mock_pywraplp = MagicMock()
    mock_solver = MagicMock()
    mock_pywraplp.Solver.CreateSolver.return_value = mock_solver
    mock_pywraplp.Solver.OPTIMAL = 0

    mock_var = MagicMock()
    mock_var.solution_value.return_value = 45.0
    mock_solver.NumVar.return_value = mock_var
    mock_solver.Sum.return_value.__le__.return_value = MagicMock()
    mock_solver.Sum.return_value.__ge__.return_value = MagicMock()
    mock_solver.Solve.return_value = 0

    adapter = ORToolsAllocationAdapter()
    adapter._pywraplp = mock_pywraplp

    state = WorldState(
        resources=[
            Resource(id="stock_a", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="stock_b", current=0.0, min_value=0.0, max_value=200.0),
        ]
    )

    actions = adapter.optimize_transfers(
        state=state,
        sources=["stock_a"],
        destinations=["stock_b"],
        demands={"stock_b": 45.0},
    )

    assert len(actions) == 1
    assert actions[0].type == "transfer_resource"
    assert actions[0].parameters["source_resource"] == "stock_a"
    assert actions[0].parameters["target_resource"] == "stock_b"
    assert actions[0].parameters["quantity"] == 45.0
