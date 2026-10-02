"""Unit tests for deterministic, stochastic, composite, and learned dynamics."""

from __future__ import annotations

import numpy as np

from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.composite import CompositeDynamics
from ewm_engine.dynamics.deterministic import (
    DeterministicTransferDynamics,
)
from ewm_engine.dynamics.learned import (
    LinearResidualDynamics,
    TransitionDataset,
    TransitionSample,
)
from ewm_engine.dynamics.stochastic import StochasticDemandDynamics
from ewm_engine.provenance.evidence import EvidenceLevel


def test_deterministic_transfer_dynamics(sample_world_state: WorldState) -> None:
    """Test conservation-preserving deterministic resource transfer."""
    dynamics = DeterministicTransferDynamics()
    rng = np.random.default_rng(42)

    action = Action(
        id="act_xfer",
        type="transfer_resource",
        parameters={
            "source_resource": "stock_wh1",
            "target_resource": "stock_wh2",
            "quantity": 30.0,
        },
    )

    result = dynamics.transition(
        state=sample_world_state,
        actions=[action],
        exogenous_events=[],
        rng=rng,
    )

    assert result.evidence_level == EvidenceLevel.STRUCTURAL
    assert result.next_state.get_resource("stock_wh1").current == 70.0
    assert result.next_state.get_resource("stock_wh2").current == 80.0
    assert result.applied_changes["transfers_executed"] == 1
    assert result.applied_changes["total_transferred_volume"] == 30.0


def test_stochastic_demand_dynamics(sample_world_state: WorldState) -> None:
    """Test stochastic demand sampling and memory updates."""
    dynamics = StochasticDemandDynamics(
        resource_id="stock_wh1",
        mean_demand=20.0,
        std_demand=0.0,  # Zero variance for deterministic validation
    )
    rng = np.random.default_rng(42)

    result = dynamics.transition(
        state=sample_world_state,
        actions=[],
        exogenous_events=[],
        rng=rng,
    )

    assert result.next_state.get_resource("stock_wh1").current == 80.0
    assert result.applied_changes["fulfilled_demand"] == 20.0
    assert result.applied_changes["unmet_demand"] == 0.0
    assert result.next_state.memory["last_demand_stock_wh1"] == 20.0


def test_composite_dynamics_pipeline(sample_world_state: WorldState) -> None:
    """Test sequential composition of deterministic flow and stochastic demand."""
    composite = CompositeDynamics(
        [
            DeterministicTransferDynamics(),
            StochasticDemandDynamics(resource_id="stock_wh2", mean_demand=10.0, std_demand=0.0),
        ]
    )
    rng = np.random.default_rng(42)

    action = Action(
        id="act_xfer",
        type="transfer_resource",
        parameters={
            "source_resource": "stock_wh1",
            "target_resource": "stock_wh2",
            "quantity": 20.0,
        },
    )

    result = composite.transition(
        state=sample_world_state,
        actions=[action],
        exogenous_events=[],
        rng=rng,
    )

    # Initial stock: wh1=100, wh2=50
    # Transfer 20: wh1=80, wh2=70
    # Demand on wh2=10: wh2=60
    assert result.next_state.get_resource("stock_wh1").current == 80.0
    assert result.next_state.get_resource("stock_wh2").current == 60.0
    # Composite of STRUCTURAL + PREDICTIVE should equal PREDICTIVE
    assert result.evidence_level == EvidenceLevel.PREDICTIVE


def test_learned_dynamics_baseline(sample_world_state: WorldState) -> None:
    """Test LinearResidualDynamics fitting and transition execution."""
    learned_model = LinearResidualDynamics(
        target_resource="stock_wh1",
        action_type="replenish_action",
    )

    # Synthetic dataset
    dataset = TransitionDataset(
        [
            TransitionSample(
                state=sample_world_state,
                actions=(Action(id="a1", type="replenish_action", parameters={"value": 10.0}),),
                next_state=sample_world_state.update_resource("stock_wh1", delta=20.0),
            ),
            TransitionSample(
                state=sample_world_state,
                actions=(Action(id="a2", type="replenish_action", parameters={"value": 20.0}),),
                next_state=sample_world_state.update_resource("stock_wh1", delta=40.0),
            ),
        ]
    )

    learned_model.fit(dataset)
    assert learned_model.is_fitted
    assert learned_model.action_weight == 2.0

    rng = np.random.default_rng(42)
    test_action = Action(id="a_test", type="replenish_action", parameters={"value": 15.0})
    result = learned_model.transition(
        state=sample_world_state,
        actions=[test_action],
        exogenous_events=[],
        rng=rng,
    )
    # Expected delta = 2.0 * 15 = 30 -> 100 + 30 = 130
    assert result.next_state.get_resource("stock_wh1").current == 130.0
    assert result.evidence_level == EvidenceLevel.PREDICTIVE


def test_learned_dynamics_ols_with_intercept_and_constant_x(
    sample_world_state: WorldState,
) -> None:
    """Test LinearResidualDynamics with non-zero intercept and constant x fallback."""
    # Data following y = 3 * x + 5:
    # x=1 -> y=8
    # x=2 -> y=11
    # x=3 -> y=14
    dataset = TransitionDataset(
        [
            TransitionSample(
                state=sample_world_state,
                actions=(Action(id="a1", type="x_act", parameters={"value": 1.0}),),
                next_state=sample_world_state.update_resource("stock_wh1", delta=8.0),
            ),
            TransitionSample(
                state=sample_world_state,
                actions=(Action(id="a2", type="x_act", parameters={"value": 2.0}),),
                next_state=sample_world_state.update_resource("stock_wh1", delta=11.0),
            ),
            TransitionSample(
                state=sample_world_state,
                actions=(Action(id="a3", type="x_act", parameters={"value": 3.0}),),
                next_state=sample_world_state.update_resource("stock_wh1", delta=14.0),
            ),
        ]
    )
    model = LinearResidualDynamics(target_resource="stock_wh1", action_type="x_act")
    model.fit(dataset)
    assert model.is_fitted
    assert round(model.action_weight, 4) == 3.0
    assert round(model.bias, 4) == 5.0

    # Constant x dataset fallback: x=0, y=10
    dataset_const = TransitionDataset(
        [
            TransitionSample(
                state=sample_world_state,
                actions=(Action(id="c1", type="x_act", parameters={"value": 0.0}),),
                next_state=sample_world_state.update_resource("stock_wh1", delta=10.0),
            ),
            TransitionSample(
                state=sample_world_state,
                actions=(Action(id="c2", type="x_act", parameters={"value": 0.0}),),
                next_state=sample_world_state.update_resource("stock_wh1", delta=12.0),
            ),
        ]
    )
    model_const = LinearResidualDynamics(target_resource="stock_wh1", action_type="x_act")
    model_const.fit(dataset_const)
    assert model_const.is_fitted
    assert model_const.action_weight == 0.0
    assert round(model_const.bias, 4) == 11.0
