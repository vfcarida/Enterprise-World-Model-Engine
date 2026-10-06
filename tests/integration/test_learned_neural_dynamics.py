"""Integration tests for learned dynamics models, dataset ergonomics, and neural baselines."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from ewm_engine.core import Action, Resource, WorldState
from ewm_engine.dynamics.learned import LinearResidualDynamics, TransitionDataset, TransitionSample
from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.experimental.dynamics_eval import (
    collect_transition_dataset,
    evaluate_dynamics_model,
    evaluate_interventional_shift,
)
from ewm_engine.experimental.dynamics_torch import TorchNeuralResidualDynamics, _load_torch
from ewm_engine.simulation.scenario import Scenario, ScheduledAction
from examples.minimal_warehouse.world import create_warehouse_world


@pytest.mark.integration
def test_transition_dataset_collection_and_split() -> None:
    """Test generating a TransitionDataset from a simulation world and splitting it."""
    world = create_warehouse_world()

    # Scenario with transfer actions across steps
    actions = [
        ScheduledAction(
            step=1,
            action=Action(
                id="act_transfer_1",
                type="transfer_resource",
                parameters={"value": 15.0, "amount": 15.0},
            ),
        ),
        ScheduledAction(
            step=2,
            action=Action(
                id="act_transfer_2",
                type="transfer_resource",
                parameters={"value": 10.0, "amount": 10.0},
            ),
        ),
    ]

    scenario = Scenario(
        scenario_id="warehouse-collection",
        name="Collection Scenario",
        horizon=3,
        samples=4,
        seed=100,
        scheduled_actions=tuple(actions),
    )

    dataset = collect_transition_dataset(world, scenario)
    assert len(dataset) > 0

    train_ds, test_ds = dataset.split(train_ratio=0.75, seed=42)
    assert len(train_ds) + len(test_ds) == len(dataset)
    assert len(train_ds) > len(test_ds)

    # Test mini-batching
    batches = list(train_ds.batch(batch_size=2, shuffle=True, seed=42))
    assert len(batches) > 0
    total_batched = sum(len(b) for b in batches)
    assert total_batched == len(train_ds)


@pytest.mark.integration
def test_torch_neural_baseline_training_and_invariant_gate() -> None:
    """Train TorchNeuralResidualDynamics and verify invariant gate and predictive accuracy."""
    rng = np.random.default_rng(42)

    # Create synthetic linear inventory dynamics: inventory_{t+1} = inventory_t + 2.0 * restock - 10.0
    samples: list[TransitionSample] = []
    for _ in range(80):
        inv_curr = float(rng.uniform(100.0, 300.0))
        restock_val = float(rng.uniform(5.0, 20.0))

        delta = (2.0 * restock_val) - 10.0
        inv_next = max(0.0, min(500.0, inv_curr + delta))

        s1 = WorldState(
            resources={
                "inventory": Resource(
                    id="inventory", current=inv_curr, min_value=0.0, max_value=500.0
                )
            }
        )
        s2 = WorldState(
            resources={
                "inventory": Resource(
                    id="inventory", current=inv_next, min_value=0.0, max_value=500.0
                )
            }
        )
        act = Action(id=f"act_{len(samples)}", type="restock", parameters={"value": restock_val})
        samples.append(TransitionSample(state=s1, actions=(act,), next_state=s2))

    dataset = TransitionDataset(samples)
    train_ds, test_ds = dataset.split(train_ratio=0.8, seed=42)

    # Train Neural Baseline
    model = TorchNeuralResidualDynamics(
        target_resources=["inventory"],
        action_types=["restock"],
        action_param="value",
        hidden_dims=(32, 32),
        learning_rate=0.01,
        epochs=80,
        batch_size=16,
        use_symlog=True,
        seed=42,
    )
    model.fit(train_ds)
    assert model.is_fitted

    # Evaluate via full harness
    report = evaluate_dynamics_model(model=model, dataset=test_ds)

    assert report.invariants.passed
    assert report.overall_valid
    assert report.invariants.bounds_violations == 0
    assert report.invariants.non_negativity_violations == 0
    # In-distribution error should be low
    assert report.one_step.mae < 10.0


@pytest.mark.integration
def test_harness_detects_interventional_distribution_shift() -> None:
    """Harness must detect significant error degradation under out-of-distribution interventional actions."""
    rng = np.random.default_rng(123)

    # In-distribution: restock in [2.0, 8.0], linear effect: delta = 2.0 * restock
    in_samples: list[TransitionSample] = []
    for _ in range(50):
        inv_curr = float(rng.uniform(50.0, 200.0))
        restock_val = float(rng.uniform(2.0, 8.0))
        delta = 2.0 * restock_val

        s1 = WorldState(
            resources={
                "inventory": Resource(
                    id="inventory", current=inv_curr, min_value=0.0, max_value=1000.0
                )
            }
        )
        s2 = WorldState(
            resources={
                "inventory": Resource(
                    id="inventory", current=inv_curr + delta, min_value=0.0, max_value=1000.0
                )
            }
        )
        act = Action(id="a", type="restock", parameters={"value": restock_val})
        in_samples.append(TransitionSample(state=s1, actions=(act,), next_state=s2))

    # Out-of-distribution intervention: restock in [40.0, 80.0], saturated diminishing returns: delta = 20.0 + log(restock)
    out_samples: list[TransitionSample] = []
    for _ in range(50):
        inv_curr = float(rng.uniform(50.0, 200.0))
        restock_val = float(rng.uniform(40.0, 80.0))
        delta = 20.0 + float(np.log(restock_val))  # Severe saturation (non-linear shift)

        s1 = WorldState(
            resources={
                "inventory": Resource(
                    id="inventory", current=inv_curr, min_value=0.0, max_value=1000.0
                )
            }
        )
        s2 = WorldState(
            resources={
                "inventory": Resource(
                    id="inventory", current=inv_curr + delta, min_value=0.0, max_value=1000.0
                )
            }
        )
        act = Action(id="a", type="restock", parameters={"value": restock_val})
        out_samples.append(TransitionSample(state=s1, actions=(act,), next_state=s2))

    in_ds = TransitionDataset(in_samples)
    out_ds = TransitionDataset(out_samples)

    # Train linear baseline on in-distribution data
    linear_model = LinearResidualDynamics(target_resource="inventory", action_type="restock")
    linear_model.fit(in_ds)

    # Evaluate shift
    shift_res = evaluate_interventional_shift(linear_model, in_ds, out_ds)

    assert shift_res.shift_detected
    assert shift_res.interventional_mae > shift_res.in_distribution_mae
    assert shift_res.gap_ratio > 1.15


@pytest.mark.integration
def test_missing_torch_raises_simulation_configuration_error() -> None:
    """When PyTorch is absent, _load_torch must raise SimulationConfigurationError with guidance."""
    with patch.dict("sys.modules", {"torch": None}):
        with pytest.raises(SimulationConfigurationError, match="pip install 'ewm-engine\\[ml\\]'"):
            _load_torch()
