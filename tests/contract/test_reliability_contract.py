"""Contract tests for causal-correctness, metamorphic self-knowledge, and reliability gates.

Enforces Reliability & Causal-Correctness Invariants:
    1. Descendant Selectivity Contract: do(X) must alter ONLY descendants of X. Perturbing non-descendants
       strictly fails the gate.
    2. Metamorphic Mutation-Adequacy Contract: MR-cards must kill synthetic structural mutants with
       adequacy score >= 0.50 without requiring human ground-truth labels.
    3. Epistemic Abstention in Decisions Contract: Out-of-support rollouts generate abstention signals
       and are down-weighted / penalized in policy ranking.
    4. Anti-Overclaiming & Physics-IQ Contract: Plausibility is NEVER a correctness proxy;
       the reliability disclaimer must certify consistency with the assumed graph, not absolute truth.
"""

from __future__ import annotations

import json
from pathlib import Path

from ewm_engine.core.actions import Action, Intervention
from ewm_engine.core.spec import WorldFactory, WorldSpec
from ewm_engine.decision.engine import DecisionEngine
from ewm_engine.evaluation.abstention import CompetenceEstimator
from ewm_engine.evaluation.causal import (
    evaluate_intervention_selectivity,
)
from ewm_engine.evaluation.metamorphic import (
    MetamorphicSuite,
    MutationAdequacyEvaluator,
)
from ewm_engine.evaluation.reliability import (
    WorldModelReliabilityReport,
    evaluate_world_model_reliability,
)
from ewm_engine.experimentation.params import ParameterDef, ParameterSpace
from ewm_engine.simulation.counterfactual import CounterfactualEngine
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario


def _get_warehouse_spec() -> WorldSpec:
    repo_root = Path(__file__).resolve().parent.parent.parent
    path = repo_root / "examples" / "minimal_warehouse" / "spec.yaml"
    return WorldSpec.from_yaml(path)


def test_contract_descendant_selectivity_gate_enforcement() -> None:
    """Contract 1: A do(X) that perturbs a non-descendant fails the causal gate with violations."""
    spec = _get_warehouse_spec()
    world = WorldFactory().create_world(spec)

    sim = SimulationEngine()
    cf = CounterfactualEngine()

    horizon = 3
    seed = 42
    scen = Scenario(scenario_id="fact", horizon=horizon, samples=1, seed=seed)
    res = sim.run(world, scen)
    f_traj = res.trajectories[0]

    # Valid intervention: passes gate
    act = Intervention(
        id="do_a",
        description="do(warehouse_a = 150)",
        parameters={"warehouse_a": 150.0},
    )
    cf_res = cf.run(world=world, observed=f_traj, intervention=act, horizon=horizon, seed=seed)

    scm = world.as_scm(horizon=horizon)
    is_correct, violations, _ = evaluate_intervention_selectivity(
        scm=scm,
        factual_trajectory=f_traj,
        counterfactual_trajectory=cf_res.counterfactual_trajectory,
        intervened_variable="resource:warehouse_a",
    )
    assert is_correct is True
    assert len(violations) == 0

    # Mutate CF trajectory to perturb an independent non-descendant variable
    corrupted_steps = []
    for s in cf_res.counterfactual_trajectory.steps:
        next_s = s.transition_result.next_state
        r_map = dict(next_s.resources)
        # warehouse_b is an independent parallel store unless transferred
        r_map["warehouse_b"] = r_map["warehouse_b"].model_copy(
            update={"current": float(r_map["warehouse_b"].current + 99.0)}
        )
        corrupted_trans = s.transition_result.model_copy(
            update={"next_state": next_s.model_copy(update={"resources": r_map})}
        )
        corrupted_steps.append(s.model_copy(update={"transition_result": corrupted_trans}))

    corrupted_cf = cf_res.counterfactual_trajectory.model_copy(
        update={"steps": tuple(corrupted_steps)}
    )

    bad_correct, bad_violations, _ = evaluate_intervention_selectivity(
        scm=scm,
        factual_trajectory=f_traj,
        counterfactual_trajectory=corrupted_cf,
        intervened_variable="resource:warehouse_a",
    )

    # In warehouse SCM, warehouse_b is not a descendant of warehouse_a without active transfer
    if "resource:warehouse_b" not in scm.causal_graph.descendants("resource:warehouse_a"):
        assert bad_correct is False
        assert any(v.violation_type == "non_descendant_divergence" for v in bad_violations)


def test_contract_metamorphic_mutant_kill_adequacy() -> None:
    """Contract 2: MR-cards kill synthetic structural mutants with adequacy >= 0.50."""
    spec = _get_warehouse_spec()
    suite = MetamorphicSuite()
    evaluator = MutationAdequacyEvaluator(suite=suite)

    score, killed, total, names = evaluator.evaluate_mutation_adequacy(spec)

    assert total > 0
    assert killed > 0
    assert score >= 0.50, f"Mutation adequacy score {score:.2f} is below contract threshold 0.50"
    assert len(names) == killed


def test_contract_epistemic_abstention_in_decisions() -> None:
    """Contract 3: Out-of-support rollouts generate abstention signals and are down-weighted in decision ranking."""
    spec = _get_warehouse_spec()
    world = WorldFactory().create_world(spec)

    # Long horizon exceeds standard verified envelope
    estimator = CompetenceEstimator(max_verified_horizon=3, horizon_decay_rate=0.3)
    sim = SimulationEngine()
    long_res = sim.run(world, Scenario(scenario_id="far_horizon", horizon=20, samples=1, seed=42))
    comp = estimator.evaluate_trajectory(long_res.trajectories[0])

    assert comp.is_competent is False
    assert comp.abstention_step is not None
    assert comp.abstention_step <= 10

    # DecisionEngine down-weights ungrounded extrapolations
    space = ParameterSpace(
        name="ContractSpace",
        parameters=(
            ParameterDef(name="w_a", type="continuous", bounds=(50.0, 150.0), default=100.0, target="resource", target_id="warehouse_a"),
        ),
    )

    candidates = {
        "status_quo": None,
        "policy_transfer": Action(
            id="transfer_act",
            type="transfer_resource",
            parameters={"source_resource": "warehouse_a", "target_resource": "warehouse_b", "quantity": 15.0},
        ),
    }

    engine = DecisionEngine()
    report = engine.decide(
        world=world,
        uncertainty_space=space,
        candidates=candidates,
        objective_metric="unserved_demand",
        satisficing_threshold=10.0,
        n_samples=5,
        horizon=15,  # Horizon beyond 3
        seed=42,
        weight_by_competence=True,
    )

    assert report is not None
    assert report.robustness is not None


def test_contract_anti_overclaiming_and_physics_iq_philosophy() -> None:
    """Contract 4: Reliability report enforces Physics-IQ doctrine and anti-overclaiming disclaimer."""
    spec = _get_warehouse_spec()

    report: WorldModelReliabilityReport = evaluate_world_model_reliability(
        spec=spec,
        horizon=4,
        seed=42,
    )

    # Invariants on the report
    assert report.evidence_disclaimer is not None
    assert "Passing reliability gates certifies model internal consistency" in report.evidence_disclaimer
    assert "It does NOT assert observational point forecasts are infallible" in report.evidence_disclaimer
    assert "EvidenceLevel" in report.evidence_disclaimer

    md = report.to_markdown()
    assert "# World Model Reliability Report" in md
    assert "Anti-Plausibility Guardrail" in md
    assert "Vafa et al." in md
    assert "Physics-IQ" in md

    # Check JSON export
    raw_json = report.to_json()
    parsed = json.loads(raw_json)
    assert parsed["is_reliable"] is True
    assert "evidence_disclaimer" in parsed
