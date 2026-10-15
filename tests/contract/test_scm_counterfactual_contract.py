"""Contract tests for SCM compilation, Pearl's three-step counterfactuals, and epistemic honesty."""

from __future__ import annotations

from typing import Any

import pytest

from ewm_engine.core.entities import Entity
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.trust import UnqualifiedCausalClaimError, validate_causal_claim
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.provenance.trace import TraceEdge
from ewm_engine.simulation.counterfactual import CounterfactualResult
from ewm_engine.simulation.scenario import Scenario


class MockStochasticDemandShock:
    """Event source that injects demand shocks."""

    def __init__(self, rate: float = 25.0) -> None:
        self.name = "StochasticDemand"
        self.rate = rate

    def sample(self, state: WorldState, step: int, rng: Any) -> list[ExogenousEvent]:
        val = float(rng.poisson(self.rate)) if hasattr(rng, "poisson") else self.rate
        return [
            ExogenousEvent(
                id=f"demand_shock_{step}",
                type="demand_shock",
                severity=1.0,
                parameters={"rate": val},
            )
        ]


def _build_contract_world() -> World:
    state = WorldState(
        resources=[
            Resource(id="stock_main", current=200.0, min_value=0.0, max_value=1000.0),
            Resource(id="stock_secondary", current=100.0, min_value=0.0, max_value=1000.0),
            Resource(id="isolated_reserve", current=500.0, min_value=0.0, max_value=5000.0),
        ],
        entities=[
            Entity(id="depot_1", type="depot", attributes={"sqft": 10000}),
        ],
    )
    dyn = DeterministicTransferDynamics(
        action_type="transfer_resource",
        name="ContractTransferDynamics",
    )
    return World(
        state=state,
        dynamics=dyn,
        event_sources=[MockStochasticDemandShock(rate=30.0)],
    )


@pytest.mark.contract
def test_contract_fingerprint_equivalence_under_coupled_noise() -> None:
    """S00 Invariant 5: Twin-Rollout counterfactual produces bit-identical states and fingerprints."""
    world = _build_contract_world()
    scenario = Scenario(name="ContractFactual", horizon=5, seed=12345)
    factual_traj = world.simulate(scenario).trajectories[0]

    # Two independent executions of the counterfactual query
    cf1: CounterfactualResult = world.counterfactual(
        observed=factual_traj,
        intervention={"stock_main": 450.0},
        horizon=5,
        seed=12345,
    )
    cf2: CounterfactualResult = world.counterfactual(
        observed=factual_traj,
        intervention={"stock_main": 450.0},
        horizon=5,
        seed=12345,
    )

    # 1. Bit-identical final state fingerprints
    assert cf1.final_state.fingerprint == cf2.final_state.fingerprint
    # 2. Bit-identical trajectory exogenous noise fingerprints
    assert (
        cf1.counterfactual_trajectory.exogenous_noise_fingerprint
        == cf2.counterfactual_trajectory.exogenous_noise_fingerprint
    )
    # 3. Bit-identical abduced noise fingerprints
    assert cf1.abduction.fingerprint == cf2.abduction.fingerprint
    # 4. Bit-identical state values
    for r_id in ("stock_main", "stock_secondary", "isolated_reserve"):
        assert cf1.final_state.resources[r_id].current == cf2.final_state.resources[r_id].current


@pytest.mark.contract
def test_contract_structural_locality_on_unconnected_subgraphs() -> None:
    """Counterfactual intervention on stock_main leaves uncoupled isolated_reserve bit-identical."""
    world = _build_contract_world()
    scenario = Scenario(name="ContractFactual", horizon=4, seed=54321)
    factual_traj = world.simulate(scenario).trajectories[0]

    cf: CounterfactualResult = world.counterfactual(
        observed=factual_traj,
        intervention={"stock_main": 500.0},
        horizon=4,
        seed=54321,
    )

    # Target variable diverged
    assert cf.target_divergences["stock_main"] == 300.0

    # Off-target isolated reserve experienced zero divergence
    assert cf.off_target_divergences.get("isolated_reserve", 0.0) == 0.0
    assert (
        cf.final_state.resources["isolated_reserve"].current
        == factual_traj.final_state.resources["isolated_reserve"].current
    )
    assert cf.locality_preserved is True


@pytest.mark.contract
def test_contract_epistemic_scoping_and_disclaimer() -> None:
    """Rung-3 outputs carry EvidenceLevel.COUNTERFACTUAL and mandatory model-counterfactual disclaimer."""
    world = _build_contract_world()
    factual_traj = world.simulate(Scenario(name="Fact", horizon=2, seed=77)).trajectories[0]

    cf = world.counterfactual(
        observed=factual_traj,
        intervention={"stock_main": 300.0},
        horizon=2,
        seed=77,
    )

    # Scoped evidence level
    assert cf.evidence_level == EvidenceLevel.COUNTERFACTUAL
    # Required epistemic disclaimer
    assert "Model-counterfactual" in cf.disclaimer
    assert "NOT a validated real-world counterfactual" in cf.disclaimer
    # No causes relation in any trace edge
    for edge in cf.counterfactual_trajectory.systemic_trace.edges:
        assert edge.relation.lower() != "causes"


@pytest.mark.contract
def test_contract_causes_relation_strictly_rejected() -> None:
    """TraceEdge validator forbids 'causes' under all circumstances."""
    with pytest.raises(ValueError, match="relation 'causes' is forbidden"):
        TraceEdge(source="intervention_a", target="node_b", relation="causes")


@pytest.mark.contract
def test_contract_validate_causal_claim_guardrails() -> None:
    """validate_causal_claim refuses unidentifiable counterfactual or interventional standing."""
    # COUNTERFACTUAL requires identifiability
    with pytest.raises(UnqualifiedCausalClaimError, match="Unqualified causal claim"):
        validate_causal_claim(
            claimed_level=EvidenceLevel.COUNTERFACTUAL,
            is_identifiable=False,
            overlap_satisfied=True,
            strict=True,
        )

    # Downgrades to PREDICTIVE with warning when strict=False
    from ewm_engine.core.trust import InsufficientCausalEvidenceWarning

    with pytest.warns(
        InsufficientCausalEvidenceWarning, match="Automatically downgrading to PREDICTIVE"
    ):
        downgraded = validate_causal_claim(
            claimed_level=EvidenceLevel.COUNTERFACTUAL,
            is_identifiable=False,
            overlap_satisfied=True,
            strict=False,
        )
    assert downgraded == EvidenceLevel.PREDICTIVE
