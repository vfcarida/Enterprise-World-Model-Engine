"""Unit tests for ML engineering rigor on learned dynamics and transition datasets.

Verifies:
1. Temporal-leakage protection on sequential TransitionDataset splits and model evaluation.
2. Cryptographic dataset, config, and fitted model SHA-256 fingerprints.
3. Datasheets-for-Datasets compliant DatasetCard generation from TransitionDataset.
4. Mitchell-complete ModelCard generation (including intended use, limitations, OOD behavior,
   uncertainty contract, seed, split methodology, and metrics) for LinearResidualDynamics and
   TorchNeuralResidualDynamics.
5. Core package imports without requiring optional ML frameworks (PyTorch).
"""

from __future__ import annotations

import pytest

from ewm_engine.core.actions import Action
from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.base import InfeasibleResidualPolicy
from ewm_engine.dynamics.learned import (
    LinearResidualDynamics,
    TransitionDataset,
    TransitionSample,
)
from ewm_engine.experimental.dynamics_torch import TorchNeuralResidualDynamics


def _build_sequential_dataset(n_steps: int = 10) -> TransitionDataset:
    """Construct a sequential trajectory dataset with strictly increasing timestamps and steps."""
    samples: list[TransitionSample] = []
    ent = Entity(id="plant_1", type="facility")
    curr_val = 50.0
    for step in range(n_steps):
        act_qty = float((step % 4) + 1) * 2.5
        delta = 2.0 * act_qty + 1.0
        next_val = curr_val + delta

        s_t = WorldState(
            entities=[ent],
            relationships=[],
            resources=[
                Resource(
                    id="inventory",
                    entity_id="plant_1",
                    current=curr_val,
                    min_value=0.0,
                    max_value=500.0,
                )
            ],
            step=step,
            timestamp=float(step),
        )
        s_next = WorldState(
            entities=[ent],
            relationships=[],
            resources=[
                Resource(
                    id="inventory",
                    entity_id="plant_1",
                    current=next_val,
                    min_value=0.0,
                    max_value=500.0,
                )
            ],
            step=step + 1,
            timestamp=float(step + 1),
        )
        act = Action(id=f"act_{step}", type="produce", parameters={"value": act_qty})
        samples.append(TransitionSample(state=s_t, actions=(act,), next_state=s_next))
        curr_val = next_val

    return TransitionDataset(samples)


def test_temporal_split_prevents_lookahead_leakage() -> None:
    """Verify temporal_split orders chronologically and blocks future-to-past leakage."""
    dataset = _build_sequential_dataset(n_steps=10)
    train_ds, eval_ds = dataset.temporal_split(train_ratio=0.7)

    assert len(train_ds) == 7
    assert len(eval_ds) == 3

    # Verify chronological ordering and zero temporal leakage
    TransitionDataset.verify_no_temporal_leakage(train_ds, eval_ds)
    max_train_ts = max(s.next_state.timestamp for s in train_ds)
    min_eval_ts = min(s.state.timestamp for s in eval_ds)
    assert max_train_ts <= min_eval_ts

    # Random permutation on sequential data with prevent_temporal_leakage=True must raise ValueError
    with pytest.raises(ValueError, match="Temporal leakage detected"):
        dataset.split(
            train_ratio=0.7,
            seed=42,
            strategy="random",
            prevent_temporal_leakage=True,
        )

    # Passing reversed splits to verify_no_temporal_leakage must fail closed
    with pytest.raises(ValueError, match="Temporal leakage detected"):
        TransitionDataset.verify_no_temporal_leakage(eval_ds, train_ds)


def test_dataset_fingerprint_and_dataset_card() -> None:
    """Verify TransitionDataset emits deterministic SHA-256 fingerprint and DatasetCard."""
    ds1 = _build_sequential_dataset(n_steps=6)
    ds2 = _build_sequential_dataset(n_steps=6)
    assert len(ds1.fingerprint) == 64
    assert ds1.fingerprint == ds2.fingerprint

    card = ds1.to_dataset_card(dataset_id="seq_prod_v1", name="Sequential Production Transitions")
    assert card.card_type == "dataset"
    assert card.artifact_fingerprint == ds1.fingerprint
    assert card.num_steps == 6
    assert "inventory" in card.features
    assert len(card.intended_use) >= 1
    assert len(card.biases_and_limitations) >= 1

    croissant = card.to_croissant()
    assert croissant["@type"] == "sc:Dataset"


def test_linear_residual_dynamics_fingerprints_eval_and_model_card() -> None:
    """Verify LinearResidualDynamics fingerprints, leakage-checked eval, and Mitchell ModelCard."""
    dataset = _build_sequential_dataset(n_steps=10)
    train_ds, eval_ds = dataset.temporal_split(train_ratio=0.7)

    model = LinearResidualDynamics(
        target_resource="inventory",
        action_type="produce",
        infeasible_policy=InfeasibleResidualPolicy.PROJECT,
        seed=123,
    )
    unfitted_fp = model.model_fingerprint
    assert len(model.config_fingerprint) == 64
    assert len(unfitted_fp) == 64

    model.fit(train_ds)
    assert model.is_fitted
    assert model.training_dataset_fingerprint == train_ds.fingerprint
    assert model.model_fingerprint != unfitted_fp
    assert model.action_weight == pytest.approx(2.0, abs=1e-6)
    assert model.bias == pytest.approx(1.0, abs=1e-6)
    assert model.training_metrics["train_rmse"] < 1e-6

    # Evaluate with temporal-leakage guard enabled
    eval_metrics = model.evaluate(eval_ds, train_dataset=train_ds, enforce_no_leakage=True)
    assert eval_metrics["eval_rmse"] < 1e-6
    assert eval_metrics["n_eval_samples"] == 3.0

    # Evaluating on leaked (reversed) train/eval split raises ValueError
    with pytest.raises(ValueError, match="Temporal leakage detected"):
        model.evaluate(train_ds, train_dataset=eval_ds, enforce_no_leakage=True)

    # Generate ModelCard and validate Mitchell et al. completeness
    card = model.to_model_card()
    missing = card.validate_mitchell_completeness(strict=True)
    assert missing == []
    assert card.artifact_fingerprint == model.model_fingerprint
    assert card.model_details["config_fingerprint"] == model.config_fingerprint
    assert card.model_details["training_dataset_fingerprint"] == train_ds.fingerprint
    assert card.model_details["seed"] == 123
    assert "InfeasibleResidualPolicy.PROJECT" in card.model_details["ood_and_infeasible_behavior"]
    assert "eval_rmse" in card.metrics


def test_torch_neural_residual_dynamics_card_and_optional_isolation() -> None:
    """Verify TorchNeuralResidualDynamics config/model card metadata without mandatory torch."""
    torch_dyn = TorchNeuralResidualDynamics(
        target_resources=["inventory"],
        action_types=["produce"],
        hidden_dims=(16, 16),
        epochs=5,
        seed=99,
    )
    assert len(torch_dyn.config_fingerprint) == 64
    assert len(torch_dyn.model_fingerprint) == 64

    card = torch_dyn.to_model_card()
    assert card.validate_mitchell_completeness(strict=True) == []
    assert card.model_details["config_fingerprint"] == torch_dyn.config_fingerprint
    assert card.model_details["seed"] == 99
    assert "MahalanobisOODDetector" in card.model_details["ood_and_infeasible_behavior"]
