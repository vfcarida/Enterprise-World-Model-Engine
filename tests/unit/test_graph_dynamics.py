"""Unit tests for GraphNeuralDynamics adapter (P10 - Experimental)."""

from __future__ import annotations

import numpy as np
import pytest

from ewm_engine.core.actions import Action
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.dynamics.learned import TransitionDataset, TransitionSample
from ewm_engine.experimental.dynamics_eval import evaluate_one_step
from ewm_engine.experimental.graph_dynamics import GraphNeuralDynamics
from ewm_engine.provenance.evidence import EvidenceLevel


def _build_network_world_state() -> WorldState:
    """Build a connected graph state for GNN dynamics testing."""
    entities = [
        Entity(id="node_a", type="terminal", attributes={"capacity": 50.0}),
        Entity(id="node_b", type="terminal", attributes={"capacity": 50.0}),
    ]
    resources = [
        Resource(id="flow_a", entity_id="node_a", current=20.0, min_value=0.0, max_value=100.0),
        Resource(id="flow_b", entity_id="node_b", current=15.0, min_value=0.0, max_value=100.0),
    ]
    relationships = [
        Relationship(
            source="node_a", target="node_b", type="connected_to", attributes={"weight": 1.0}
        ),
        Relationship(
            source="node_b", target="node_a", type="connected_to", attributes={"weight": 1.0}
        ),
    ]
    return WorldState(
        step=0,
        timestamp=0.0,
        schema_version="2.0.0",
        entities=entities,
        resources=resources,
        relationships=relationships,
    )


def test_graph_neural_dynamics_step() -> None:
    """Assert GraphNeuralDynamics executes step transitions and returns PREDICTIVE evidence."""
    pytest.importorskip("torch")

    model = GraphNeuralDynamics(hidden_dim=16, num_layers=2)
    state = _build_network_world_state()
    rng = np.random.default_rng(42)

    result = model.transition(state, [], [], rng)

    assert isinstance(result, TransitionResult)
    assert result.evidence_level == EvidenceLevel.PREDICTIVE
    assert result.next_state.step == 1
    assert result.next_state.timestamp == 1.0

    # Resource bounds respected
    for _r_id, r in result.next_state.resources.items():
        assert r.min_value <= r.current <= r.max_value


def test_graph_neural_dynamics_action_perturbation() -> None:
    """Assert actions targeting nodes induce feature perturbations."""
    pytest.importorskip("torch")

    model = GraphNeuralDynamics(hidden_dim=16)
    state = _build_network_world_state()

    act = Action(
        id="act_boost",
        type="boost_flow",
        parameters={"target_entity": "node_a", "flow_a": 10.0},
    )

    result_no_action = model.transition(state, [], [], np.random.default_rng(42))
    result_with_action = model.transition(state, [act], [], np.random.default_rng(42))

    # Next state flow should reflect action influence
    assert (
        result_with_action.next_state.resources["flow_a"].current
        != result_no_action.next_state.resources["flow_a"].current
    )


def test_graph_neural_dynamics_fit_and_eval_harness() -> None:
    """Assert GraphNeuralDynamics fits a dataset and evaluates under the P06 harness."""
    pytest.importorskip("torch")

    state = _build_network_world_state()

    # Generate synthetic training transitions
    samples: list[TransitionSample] = []
    curr = state
    for _ in range(25):
        flow_a = float(curr.resources["flow_a"].current)
        flow_b = float(curr.resources["flow_b"].current)
        # Structural ground truth: mutual diffusion
        next_flow_a = float(np.clip(flow_a - 0.1 * flow_a + 0.1 * flow_b, 0.0, 100.0))
        next_flow_b = float(np.clip(flow_b - 0.1 * flow_b + 0.1 * flow_a, 0.0, 100.0))

        next_s = (
            curr.update_resource("flow_a", new_value=next_flow_a)
            .update_resource("flow_b", new_value=next_flow_b)
            .advance_time()
        )

        samples.append(
            TransitionSample(
                state=curr,
                actions=(),
                events=(),
                next_state=next_s,
            )
        )
        curr = next_s

    dataset = TransitionDataset(samples=samples)

    model = GraphNeuralDynamics(hidden_dim=16, learning_rate=0.01)
    model.fit(dataset, epochs=15, seed=42)

    assert model._is_fitted is True

    # P06 evaluation harness compatibility
    one_step, _ = evaluate_one_step(model, dataset)
    assert one_step.mae >= 0.0
    assert one_step.rmse >= 0.0
    assert len(one_step.per_resource_mae) == 2
