"""Property tests ensuring zero dependency on global random or numpy.random state."""

from __future__ import annotations

import ast
import random
from pathlib import Path

import numpy as np
import pytest

from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.stochastic import StochasticDemandDynamics
from ewm_engine.simulation.scenario import Scenario

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = REPO_ROOT / "src" / "ewm_engine"

DISALLOWED_GLOBAL_RNG_CALLS = {
    "random.random",
    "random.randint",
    "random.choice",
    "random.uniform",
    "random.gauss",
    "random.seed",
    "np.random.rand",
    "np.random.randn",
    "np.random.randint",
    "np.random.choice",
    "np.random.uniform",
    "np.random.normal",
    "np.random.seed",
}


@pytest.mark.property
def test_ast_scan_no_global_rng_calls_in_core_modules() -> None:
    """AC-004 & AC-009: Statically assert that core simulation and dynamics do not invoke global RNG functions."""
    core_packages = ["simulation", "dynamics", "constraints", "core", "provenance"]

    for pkg_name in core_packages:
        pkg_dir = SRC_ROOT / pkg_name
        for py_file in pkg_dir.rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Check func like random.choice or np.random.normal
                    call_name = ""
                    if isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name):
                            call_name = f"{node.func.value.id}.{node.func.attr}"
                        elif isinstance(node.func.value, ast.Attribute) and isinstance(
                            node.func.value.value, ast.Name
                        ):
                            call_name = f"{node.func.value.value.id}.{node.func.value.attr}.{node.func.attr}"

                    if call_name in DISALLOWED_GLOBAL_RNG_CALLS:
                        pytest.fail(
                            f"Forbidden global RNG call '{call_name}' in {py_file.relative_to(REPO_ROOT)}:{node.lineno}! "
                            f"Components must accept and use explicit np.random.Generator (rng) instances."
                        )


@pytest.mark.property
def test_external_global_rng_seeding_does_not_affect_simulation_results() -> None:
    """AC-004: Externally seeding random or np.random has zero effect on engine reproducibility."""
    base_state = WorldState(
        resources=[Resource(id="inventory", current=250.0, min_value=0.0, max_value=500.0)],
    )
    world = World(
        state=base_state,
        dynamics=StochasticDemandDynamics(
            resource_id="inventory", mean_demand=15.0, std_demand=5.0
        ),
    )
    scenario = Scenario(horizon=10, samples=3, seed=777)

    # Run 1: with one global seed state
    random.seed(11111)
    np.random.seed(22222)
    _ = [random.random() for _ in range(100)]
    _ = np.random.rand(100)
    res_1 = world.simulate(scenario)

    # Run 2: with completely different global seed and state
    random.seed(99999)
    np.random.seed(88888)
    _ = [random.random() for _ in range(500)]
    _ = np.random.rand(500)
    res_2 = world.simulate(scenario)

    # Trajectories must be identical
    assert len(res_1.trajectories) == len(res_2.trajectories) == 3
    for t1, t2 in zip(res_1.trajectories, res_2.trajectories, strict=True):
        assert t1.seed == t2.seed
        assert t1.final_state.fingerprint == t2.final_state.fingerprint
        assert t1.metric_series("resource_inventory") == t2.metric_series("resource_inventory")
