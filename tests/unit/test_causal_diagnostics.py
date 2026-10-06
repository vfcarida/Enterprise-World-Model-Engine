"""Unit tests for causal diagnostics, Ladder-of-Causation identifiability, and Twin Rollouts (P09)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from ewm_engine.core.actions import Action
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.base import DynamicsModel, TransitionResult
from ewm_engine.dynamics.learned import TransitionDataset, TransitionSample
from ewm_engine.experimental.causal import (
    CausalGraph,
    NotIdentifiableResult,
    check_backdoor_identifiability,
    check_positivity_overlap,
    query_interventional,
    query_observational,
    report_confounding_sensitivity,
    twin_rollout_counterfactual,
)
from ewm_engine.provenance.evidence import EvidenceLevel


def test_causal_graph_backdoor_identifiability_with_unobserved_confounder() -> None:
    """Detects open backdoor paths and refuses to emit INTERVENTIONAL evidence level."""
    # Graph: U (unobserved confounder) -> X (treatment), U -> Y (outcome), X -> Y
    dag = CausalGraph()
    dag.add_node("U", is_unobserved=True)
    dag.add_node("X")
    dag.add_node("Y")

    dag.add_edge("U", "X")
    dag.add_edge("U", "Y")
    dag.add_edge("X", "Y")

    # 1. Without conditioning: backdoor path X <- U -> Y is open
    res_empty = check_backdoor_identifiability(dag, treatment="X", outcome="Y", conditioning_set=())
    assert not res_empty.is_identifiable
    assert res_empty.causal_level == EvidenceLevel.PREDICTIVE
    assert len(res_empty.open_backdoor_paths) == 1
    assert res_empty.open_backdoor_paths[0] == ("X", "U", "Y")
    assert "U" in res_empty.unobserved_confounders

    # 2. Even if user attempts to condition on an observed proxy Z that does not block U
    dag.add_node("Z")
    dag.add_edge("X", "Z")
    res_z = check_backdoor_identifiability(dag, treatment="X", outcome="Y", conditioning_set=["Z"])
    assert not res_z.is_identifiable
    assert res_z.causal_level == EvidenceLevel.PREDICTIVE

    # 3. Graph with observed confounder W: W -> X, W -> Y, X -> Y
    dag_clean = CausalGraph()
    dag_clean.add_node("W", is_unobserved=False)
    dag_clean.add_node("X")
    dag_clean.add_node("Y")
    dag_clean.add_edge("W", "X")
    dag_clean.add_edge("W", "Y")
    dag_clean.add_edge("X", "Y")

    # Without conditioning on W -> fails
    assert not check_backdoor_identifiability(dag_clean, "X", "Y", ()).is_identifiable

    # Controlling for W blocks the backdoor -> IDENTIFIABLE
    res_clean = check_backdoor_identifiability(dag_clean, "X", "Y", ("W",))
    assert res_clean.is_identifiable
    assert res_clean.causal_level == EvidenceLevel.INTERVENTIONAL
    assert len(res_clean.open_backdoor_paths) == 0


def test_backdoor_identifiability_forbids_conditioning_on_descendants() -> None:
    """Conditioning on a descendant of treatment induces selection bias and invalidates identifiability."""
    dag = CausalGraph()
    dag.add_node("X")
    dag.add_node("Y")
    dag.add_node("M")  # Mediator / descendant: X -> M -> Y
    dag.add_edge("X", "M")
    dag.add_edge("M", "Y")

    res = check_backdoor_identifiability(dag, treatment="X", outcome="Y", conditioning_set=["M"])
    assert not res.is_identifiable
    assert "descendants of treatment" in res.diagnostic_message


def _create_synthetic_confounded_dataset(n_samples: int = 100) -> TransitionDataset:
    """Generate transitions where treatment is confounded with market demand."""
    samples: list[TransitionSample] = []
    rng = np.random.default_rng(123)

    for i in range(n_samples):
        # Confounder U (market boom)
        market_boom = bool(rng.uniform(0.0, 1.0) > 0.5)

        # Treatment probability depends heavily on market boom
        prob_treat = 0.8 if market_boom else 0.2
        is_treated = bool(rng.uniform(0.0, 1.0) < prob_treat)

        act = Action(id=f"act_{i}", type="promo" if is_treated else "idle", parameters={})
        actions = (act,)

        s0 = WorldState(
            resources={
                "sales": Resource(id="sales", current=0.0, min_value=0.0, max_value=1000.0),
                "market_indicator": Resource(
                    id="market_indicator",
                    current=1.0 if market_boom else 0.0,
                    min_value=0.0,
                    max_value=1.0,
                ),
            }
        )

        # True sales: boom adds 50.0, promo adds 10.0
        final_sales = (
            (50.0 if market_boom else 0.0) + (10.0 if is_treated else 0.0) + rng.normal(0.0, 1.0)
        )
        s1 = s0.update_resource("sales", new_value=float(final_sales), clamp=True)

        samples.append(TransitionSample(state=s0, actions=actions, next_state=s1))

    return TransitionDataset(samples)


def test_ladder_of_causation_observational_vs_interventional_query() -> None:
    """Demonstrates that observational P(Y|X) != interventional P(Y|do(X)) under confounding."""
    ds = _create_synthetic_confounded_dataset(150)

    # 1. Observational query: naive association overestimates promo impact due to boom confounding!
    obs_res = query_observational(ds, treatment_action="promo", outcome_resource="sales")
    assert obs_res.causal_level == EvidenceLevel.PREDICTIVE
    # Observational association is biased upward (around 30-40 instead of true 10.0)
    assert obs_res.observational_association > 15.0

    # 2. Interventional query with open confounder: returns NotIdentifiableResult
    dag_unblocked = CausalGraph()
    dag_unblocked.add_node("promo")
    dag_unblocked.add_node("sales")
    dag_unblocked.add_node("market_boom", is_unobserved=True)
    dag_unblocked.add_edge("market_boom", "promo")
    dag_unblocked.add_edge("market_boom", "sales")
    dag_unblocked.add_edge("promo", "sales")

    query_fail = query_interventional(ds, "promo", "sales", dag_unblocked, conditioning_set=())
    assert isinstance(query_fail, NotIdentifiableResult)
    assert query_fail.causal_level == EvidenceLevel.PREDICTIVE

    # 3. Interventional query with controlled confounder (market_indicator)
    dag_blocked = CausalGraph()
    dag_blocked.add_node("promo")
    dag_blocked.add_node("sales")
    dag_blocked.add_node("market_indicator", is_unobserved=False)
    dag_blocked.add_edge("market_indicator", "promo")
    dag_blocked.add_edge("market_indicator", "sales")
    dag_blocked.add_edge("promo", "sales")

    query_success = query_interventional(
        ds, "promo", "sales", dag_blocked, conditioning_set=["market_indicator"]
    )
    assert not isinstance(query_success, NotIdentifiableResult)
    assert query_success.causal_level == EvidenceLevel.INTERVENTIONAL
    # Adjusted effect recovers the true effect (approx 10.0) rather than naive 35.0!
    assert 5.0 <= query_success.adjusted_effect <= 15.0


def test_positivity_overlap_diagnostic() -> None:
    """Positivity diagnostic flags common support breakdown and zero-support regions."""
    # 1. Common support dataset
    ds_good = _create_synthetic_confounded_dataset(60)
    rep_good = check_positivity_overlap(ds_good, "promo", ["market_indicator"], min_propensity=0.05)
    assert rep_good.treated_count > 0
    assert rep_good.control_count > 0
    assert rep_good.overlap_index > 0.3

    # 2. Extreme non-overlap dataset (treatment only applied when cash is strictly > 500)
    separated_samples: list[TransitionSample] = []
    for i in range(40):
        is_treated = i >= 20
        cash = 800.0 if is_treated else 100.0
        act = Action(id=f"a_{i}", type="big_order" if is_treated else "idle", parameters={})
        s0 = WorldState(
            resources={"cash": Resource(id="cash", current=cash, min_value=0.0, max_value=1000.0)}
        )
        s1 = s0.update_resource("cash", delta=0.0)
        separated_samples.append(TransitionSample(state=s0, actions=(act,), next_state=s1))

    ds_bad = TransitionDataset(separated_samples)
    rep_bad = check_positivity_overlap(ds_bad, "big_order", ["cash"], min_propensity=0.05)
    assert not rep_bad.positivity_satisfied
    assert rep_bad.violation_rate > 0.5


def test_confounding_sensitivity_rosenbaum_bounds() -> None:
    """Rosenbaum bounds schedule correctly identifies critical Gamma breakdown threshold."""
    # Treatment group has outcome ~ 25, control has outcome ~ 15 (delta = 10.0)
    treated = [25.0 + float(i % 3) for i in range(30)]
    control = [15.0 + float(i % 3) for i in range(30)]

    report = report_confounding_sensitivity(treated, control, gammas=[1.0, 1.25, 1.5, 2.0, 3.0])

    assert report.point_estimate == pytest.approx(10.0)
    assert len(report.sensitivity_schedule) == 5
    assert report.gamma_breakdown >= 1.0
    assert "Sensitivity Analysis" in report.diagnostic_message


class SyntheticCoupledDynamics(DynamicsModel):
    """Dynamics with one targeted resource and one independent resource."""

    def __init__(self, name: str = "SyntheticCoupledDynamics") -> None:
        self.name = name

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[Any],
        rng: Any,
    ) -> TransitionResult:
        stock_add = sum(
            float(a.parameters.get("qty", 0.0)) for a in actions if a.type == "add_stock"
        )
        noise = float(rng.normal(0.0, 0.1))

        s1 = state.update_resource("stock", delta=(stock_add + noise), clamp=True)
        # Independent resource only fluctuates with random noise
        s2 = s1.update_resource("isolated_metric", delta=noise, clamp=True)

        return TransitionResult(
            next_state=s2,
            applied_changes={},
            evidence_level=EvidenceLevel.STRUCTURAL,
            model_name=self.name,
        )


def test_twin_rollout_counterfactual_noise_coupling() -> None:
    """Twin rollouts preserve exact noise coupling, isolating target effect from off-target noise."""
    s0 = WorldState(
        resources={
            "stock": Resource(id="stock", current=50.0, min_value=0.0, max_value=500.0),
            "isolated_metric": Resource(
                id="isolated_metric", current=100.0, min_value=0.0, max_value=500.0
            ),
        }
    )
    world = World(state=s0, dynamics=SyntheticCoupledDynamics())

    act_factual = [Action(id="f", type="idle", parameters={})]
    act_counterfactual = [Action(id="cf", type="add_stock", parameters={"qty": 25.0})]

    res = twin_rollout_counterfactual(
        world=world,
        initial_state=s0,
        factual_actions=act_factual,
        counterfactual_actions=act_counterfactual,
        target_resources=["stock"],
        horizon=2,
        seed=42,
    )

    # 1. Target effect on stock reflects the 25.0 intervention
    assert res.target_effects["stock"] == pytest.approx(25.0, abs=1e-4)

    # 2. Exact noise coupling ensures off-target isolated_metric experiences ZERO divergence
    assert res.off_target_divergences["isolated_metric"] == pytest.approx(0.0, abs=1e-4)
    assert res.locality_preserved
    assert "Twin rollout successful" in res.diagnostic_message
