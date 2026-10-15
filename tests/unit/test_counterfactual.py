"""Unit tests for native Pearl three-step counterfactual engine and Twin-Rollout noise coupling."""

from __future__ import annotations

from typing import Any

import pytest

from ewm_engine.core.actions import Intervention
from ewm_engine.core.entities import Entity
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.resources import Resource
from ewm_engine.core.scm import NonUniqueAbductionWarning
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.simulation.counterfactual import CounterfactualResult
from ewm_engine.simulation.scenario import Scenario


class DeterministicStepShockSource:
    """Deterministic event source injecting demand shocks."""

    def __init__(self, shock_amount: float = 20.0) -> None:
        self.name = "StepDemandShockSource"
        self.shock_amount = shock_amount

    def sample(self, state: WorldState, step: int, rng: Any) -> list[ExogenousEvent]:
        return [
            ExogenousEvent(
                id=f"demand_shock_{step}",
                type="demand_shock",
                severity=1.0,
                parameters={"amount": self.shock_amount},
            )
        ]


def _build_logistics_world() -> World:
    state = WorldState(
        resources=[
            Resource(id="warehouse_stock", current=100.0, min_value=0.0, max_value=500.0),
            Resource(id="retail_stock", current=50.0, min_value=0.0, max_value=500.0),
            Resource(id="unrelated_fund", current=5000.0, min_value=0.0, max_value=100000.0),
        ],
        entities=[
            Entity(id="hub_central", type="hub", attributes={"region": "central"}),
        ],
    )
    dyn = DeterministicTransferDynamics(
        action_type="transfer_resource",
        name="LogisticsTransfer",
    )
    world = World(
        state=state,
        dynamics=dyn,
        event_sources=[DeterministicStepShockSource(shock_amount=20.0)],
    )
    return world


def test_native_counterfactual_rollout_coupled_noise() -> None:
    world = _build_logistics_world()

    # 1. Run factual rollout
    scenario = Scenario(name="FactualRun", horizon=4, seed=12345)
    factual_res = world.simulate(scenario)
    factual_traj = factual_res.trajectories[0]

    # Verify exogenous noise was recorded during factual rollout
    assert len(factual_traj.exogenous_noise) > 0
    assert len(factual_traj.exogenous_noise_fingerprint) == 64

    # 2. Execute counterfactual query under causal surgery do(warehouse_stock = 300)
    cf_result: CounterfactualResult = world.counterfactual(
        observed=factual_traj,
        intervention={"warehouse_stock": 300.0},
        horizon=4,
        seed=12345,
    )

    # 3. Assertions on Pearl's three-step result
    assert isinstance(cf_result, CounterfactualResult)
    assert cf_result.evidence_level == EvidenceLevel.COUNTERFACTUAL
    assert "Model-counterfactual" in cf_result.disclaimer
    assert "simulated SCM" in cf_result.disclaimer
    assert cf_result.is_unique is True
    assert cf_result.is_bit_reproducible is True

    # Counterfactual stock should diverge from factual stock
    assert cf_result.target_divergences["warehouse_stock"] == 200.0
    assert (
        cf_result.counterfactual_trajectory.final_state.resources["warehouse_stock"].current
        > factual_traj.final_state.resources["warehouse_stock"].current
    )

    # Unrelated fund (no causal path) must remain bit-identical (structural locality)
    assert cf_result.off_target_divergences.get("unrelated_fund", 0.0) == 0.0
    assert (
        cf_result.counterfactual_trajectory.final_state.resources["unrelated_fund"].current
        == factual_traj.final_state.resources["unrelated_fund"].current
    )
    assert cf_result.locality_preserved is True


def test_counterfactual_bit_reproducibility() -> None:
    """Invariant 5: Counterfactual rollout must be strictly bit-reproducible across reruns."""
    world = _build_logistics_world()
    scenario = Scenario(name="FactualRun", horizon=3, seed=999)
    factual_traj = world.simulate(scenario).trajectories[0]

    cf1 = world.counterfactual(
        observed=factual_traj,
        intervention={"warehouse_stock": 250.0},
        horizon=3,
        seed=999,
    )
    cf2 = world.counterfactual(
        observed=factual_traj,
        intervention={"warehouse_stock": 250.0},
        horizon=3,
        seed=999,
    )

    # Bit-identical final state fingerprints
    assert cf1.final_state.fingerprint == cf2.final_state.fingerprint
    # Bit-identical abduced noise fingerprints
    assert cf1.abduction.fingerprint == cf2.abduction.fingerprint
    # Bit-identical counterfactual trajectory fingerprints
    assert (
        cf1.counterfactual_trajectory.exogenous_noise_fingerprint
        == cf2.counterfactual_trajectory.exogenous_noise_fingerprint
    )


def test_counterfactual_with_intervention_object() -> None:
    world = _build_logistics_world()
    scenario = Scenario(name="FactualRun", horizon=2, seed=42)
    factual_traj = world.simulate(scenario).trajectories[0]

    intervention = Intervention(
        id="replenish_stock",
        description="Emergency replenishment of warehouse stock",
        parameters={"warehouse_stock": 400.0},
    )

    cf_res = world.counterfactual(
        observed=factual_traj,
        intervention=intervention,
        horizon=2,
        seed=42,
    )

    assert cf_res.target_divergences["warehouse_stock"] == 300.0
    assert cf_res.counterfactual_trajectory.final_state.resources["warehouse_stock"].current >= 400.0


def test_counterfactual_trace_diff_and_mermaid() -> None:
    world = _build_logistics_world()
    factual_traj = world.simulate(Scenario(name="Fact", horizon=2, seed=777)).trajectories[0]

    cf_res = world.counterfactual(
        observed=factual_traj,
        intervention={"warehouse_stock": 350.0},
        horizon=2,
        seed=777,
    )

    trace_diff = cf_res.trace_diff
    assert trace_diff.is_noise_coupled is True
    assert len(trace_diff.coupled_noise_ids) > 0
    assert len(trace_diff.divergent_noise_ids) == 0

    mermaid_diagram = trace_diff.to_mermaid()
    assert "flowchart TD" in mermaid_diagram
    assert "classDef invariant" in mermaid_diagram
    assert "classDef divergent" in mermaid_diagram


def test_non_unique_abduction_guard() -> None:
    """Non-unique abduction must warn, set is_unique=False, and provide posterior samples."""
    world = _build_logistics_world()
    factual_traj = world.simulate(Scenario(name="Fact", horizon=2, seed=888)).trajectories[0]

    # Artificially insert a non-invertible exogenous draw in the factual trace
    factual_traj.systemic_trace.record_exogenous_draw(
        noise_id="u:stochastic_non_invertible_mechanism",
        step=0,
        mechanism="StochasticPooling",
        variable="pooled_demand",
        value=10.0,
        is_invertible=False,
    )

    with pytest.warns(NonUniqueAbductionWarning, match="Non-unique abduction detected"):
        cf_res = world.counterfactual(
            observed=factual_traj,
            intervention={"warehouse_stock": 200.0},
            horizon=2,
            seed=888,
        )

    assert cf_res.is_unique is False
    assert len(cf_res.posterior_sample_trajectories) >= 2
    # Disclaimer still present
    assert "Model-counterfactual" in cf_res.disclaimer
