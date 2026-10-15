"""Unit tests for epistemic competence, abstention, and Physics-IQ guardrails."""

from __future__ import annotations

from pathlib import Path

from ewm_engine.core.actions import Action
from ewm_engine.core.spec import WorldFactory, WorldSpec
from ewm_engine.decision.engine import DecisionEngine
from ewm_engine.evaluation.abstention import CompetenceEstimator, PhysicsIQGuardrail
from ewm_engine.experimentation.params import ParameterDef, ParameterSpace
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario


def _get_warehouse_spec() -> WorldSpec:
    repo_root = Path(__file__).resolve().parent.parent.parent
    path = repo_root / "examples" / "minimal_warehouse" / "spec.yaml"
    return WorldSpec.from_yaml(path)


def test_competence_estimator_within_support() -> None:
    """Trajectories within verified horizon and support retain high competence."""
    spec = _get_warehouse_spec()
    world = WorldFactory().create_world(spec)
    sim = SimulationEngine()

    scen = Scenario(scenario_id="in_support", horizon=5, samples=1, seed=42)
    res = sim.run(world, scen)
    traj = res.trajectories[0]

    estimator = CompetenceEstimator(
        max_verified_horizon=10,
        support_bounds={"warehouse_a": (0.0, 300.0)},
        abstention_threshold=0.35,
    )
    competence = estimator.evaluate_trajectory(traj)

    assert competence.is_competent is True
    assert competence.competence_score >= 0.8
    assert competence.abstention_step is None
    assert competence.abstention_reason is None


def test_competence_estimator_horizon_decay_abstention() -> None:
    """Long-horizon rollouts beyond verified capacity decay exponentially and trigger abstention."""
    spec = _get_warehouse_spec()
    world = WorldFactory().create_world(spec)
    sim = SimulationEngine()

    # Long horizon rollout (step 30 with verified horizon of 5)
    scen = Scenario(scenario_id="long_horizon", horizon=30, samples=1, seed=42)
    res = sim.run(world, scen)
    traj = res.trajectories[0]

    estimator = CompetenceEstimator(
        max_verified_horizon=5,
        horizon_decay_rate=0.2,
        abstention_threshold=0.35,
    )
    competence = estimator.evaluate_trajectory(traj)

    assert competence.is_competent is False
    assert competence.competence_score < 0.35
    assert competence.abstention_step is not None
    assert competence.abstention_step > 5
    assert competence.abstention_reason is not None
    assert "exceeds verified envelope" in competence.abstention_reason


def test_competence_estimator_out_of_support_bounds() -> None:
    """Drifting outside historical support bounds penalizes competence."""
    spec = _get_warehouse_spec()
    world = WorldFactory().create_world(spec)
    sim = SimulationEngine()

    scen = Scenario(scenario_id="supp_scen", horizon=5, samples=1, seed=42)
    res = sim.run(world, scen)
    traj = res.trajectories[0]

    # Set support bounds that current warehouse_a exceeds
    estimator = CompetenceEstimator(
        max_verified_horizon=10,
        support_bounds={"warehouse_a": (0.0, 10.0)},  # warehouse_a starts at 100
        abstention_threshold=0.5,
    )
    competence = estimator.evaluate_trajectory(traj)

    assert competence.is_competent is False
    assert "drifted outside support" in str(competence.abstention_reason)


def test_physics_iq_guardrail_clean_and_violated() -> None:
    """Physics-IQ checks physical invariants (non-negativity & conservation) without perceptual proxies."""
    spec = _get_warehouse_spec()
    world = WorldFactory().create_world(spec)
    sim = SimulationEngine()

    scen = Scenario(scenario_id="piq_scen", horizon=4, samples=1, seed=42)
    res = sim.run(world, scen)
    clean_traj = res.trajectories[0]

    guardrail = PhysicsIQGuardrail()

    # 1. Clean trajectory passes
    score, is_sound, viols = guardrail.evaluate_physics_iq(clean_traj)
    assert score >= 0.999
    assert is_sound is True
    assert len(viols) == 0

    # 2. Corrupted trajectory with negative physical inventory
    corrupted_steps = []
    for step in clean_traj.steps:
        next_s = step.transition_result.next_state
        corrupted_res = dict(next_s.resources)
        corrupted_res["warehouse_a"] = corrupted_res["warehouse_a"].model_copy(
            update={"current": -25.0}  # Negative inventory violation!
        )
        corrupted_trans = step.transition_result.model_copy(
            update={"next_state": next_s.model_copy(update={"resources": corrupted_res})}
        )
        corrupted_steps.append(step.model_copy(update={"transition_result": corrupted_trans}))

    corrupted_traj = clean_traj.model_copy(update={"steps": tuple(corrupted_steps)})
    bad_score, bad_sound, bad_viols = guardrail.evaluate_physics_iq(corrupted_traj)

    assert bad_sound is False
    assert bad_score < 1.0
    assert any("Negative physical resource" in v for v in bad_viols)


def test_decision_engine_competence_weighting_integration() -> None:
    """DecisionEngine weights or filters ungrounded rollouts based on epistemic competence."""
    spec = _get_warehouse_spec()
    world = WorldFactory().create_world(spec)

    space = ParameterSpace(
        name="ParamSpace",
        parameters=(
            ParameterDef(
                name="w_a",
                type="continuous",
                bounds=(50.0, 150.0),
                default=100.0,
                target="resource",
                target_id="warehouse_a",
            ),
        ),
    )

    candidates = {
        "status_quo": None,
        "policy_transfer": Action(
            id="transfer_act",
            type="transfer_resource",
            parameters={"source_resource": "warehouse_a", "target_resource": "warehouse_b", "quantity": 20.0},
        ),
    }

    engine = DecisionEngine()

    # Run with competence weighting enabled
    report_weighted = engine.decide(
        world=world,
        uncertainty_space=space,
        candidates=candidates,
        objective_metric="unserved_demand",
        satisficing_threshold=15.0,
        n_samples=5,
        horizon=6,
        seed=42,
        weight_by_competence=True,
    )

    assert report_weighted.candidates == ("status_quo", "policy_transfer")
    assert report_weighted.robustness is not None

    # Run with abstention filtering enabled
    report_filtered = engine.decide(
        world=world,
        uncertainty_space=space,
        candidates=candidates,
        objective_metric="unserved_demand",
        satisficing_threshold=15.0,
        n_samples=5,
        horizon=6,
        seed=42,
        filter_abstained=True,
    )

    assert report_filtered.robustness is not None
