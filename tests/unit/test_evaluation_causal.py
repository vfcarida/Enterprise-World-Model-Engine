"""Unit tests for causal-correctness and intervention-selectivity gates."""

from __future__ import annotations

from pathlib import Path

from ewm_engine.core.actions import Intervention
from ewm_engine.core.spec import WorldFactory, WorldSpec
from ewm_engine.evaluation.causal import (
    CausalCorrectnessGate,
    MonotonicDirection,
    evaluate_intervention_selectivity,
)
from ewm_engine.simulation.counterfactual import CounterfactualEngine
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario


def _get_warehouse_spec() -> WorldSpec:
    repo_root = Path(__file__).resolve().parent.parent.parent
    path = repo_root / "examples" / "minimal_warehouse" / "spec.yaml"
    return WorldSpec.from_yaml(path)


def test_intervention_selectivity_evaluation_clean_world() -> None:
    """Intervention on warehouse_a only affects warehouse_a and downstream descendants."""
    spec = _get_warehouse_spec()
    world = WorldFactory().create_world(spec)

    sim = SimulationEngine()
    cf = CounterfactualEngine()

    horizon = 5
    seed = 42
    scen = Scenario(scenario_id="fact", horizon=horizon, samples=1, seed=seed)
    res = sim.run(world, scen)
    f_traj = res.trajectories[0]

    # Intervene on warehouse_a
    act = Intervention(
        id="do_a",
        description="do(warehouse_a = 150)",
        parameters={"warehouse_a": 150.0},
    )
    cf_res = cf.run(world=world, observed=f_traj, intervention=act, horizon=horizon, seed=seed)

    scm = world.as_scm(horizon=horizon)
    is_correct, violations, checks = evaluate_intervention_selectivity(
        scm=scm,
        factual_trajectory=f_traj,
        counterfactual_trajectory=cf_res.counterfactual_trajectory,
        intervened_variable="resource:warehouse_a",
    )

    assert is_correct is True
    assert len(violations) == 0
    assert checks > 0


def test_intervention_selectivity_detects_non_descendant_divergence() -> None:
    """Spurious perturbation of an independent non-descendant variable triggers a violation."""
    spec = _get_warehouse_spec()
    world = WorldFactory().create_world(spec)

    sim = SimulationEngine()
    cf = CounterfactualEngine()

    horizon = 4
    seed = 42
    scen = Scenario(scenario_id="fact", horizon=horizon, samples=1, seed=seed)
    res = sim.run(world, scen)
    f_traj = res.trajectories[0]

    # Intervene on warehouse_a
    act = Intervention(
        id="do_a",
        description="do(warehouse_a = 150)",
        parameters={"warehouse_a": 150.0},
    )
    cf_res = cf.run(world=world, observed=f_traj, intervention=act, horizon=horizon, seed=seed)
    cf_traj = cf_res.counterfactual_trajectory

    # Artificially perturb a variable that is NOT a descendant in the counterfactual trajectory
    from ewm_engine.simulation.trajectory import StepRecord

    corrupted_steps: list[StepRecord] = []
    for step in cf_traj.steps:
        next_state = step.transition_result.next_state
        # Inject spurious divergence in a non-descendant variable (or create dummy non-descendant)
        updated_res = dict(next_state.resources)
        if "warehouse_b" in updated_res:
            updated_res["warehouse_b"] = updated_res["warehouse_b"].model_copy(
                update={"current": float(updated_res["warehouse_b"].current + 50.0)}
            )
        corrupted_trans = step.transition_result.model_copy(
            update={"next_state": next_state.model_copy(update={"resources": updated_res})}
        )
        corrupted_steps.append(step.model_copy(update={"transition_result": corrupted_trans}))

    corrupted_cf = cf_traj.model_copy(update={"steps": tuple(corrupted_steps)})

    scm = world.as_scm(horizon=horizon)
    # Check if warehouse_b is descendant; if not descendant, must be flagged
    if "resource:warehouse_b" not in scm.causal_graph.descendants("resource:warehouse_a"):
        is_correct, violations, _ = evaluate_intervention_selectivity(
            scm=scm,
            factual_trajectory=f_traj,
            counterfactual_trajectory=corrupted_cf,
            intervened_variable="resource:warehouse_a",
        )
        assert is_correct is False
        assert any(v.violation_type == "non_descendant_divergence" for v in violations)


def test_monotonicity_sign_checking() -> None:
    """Intervention checking validates monotonic edge consistency and flags sign inversions."""
    spec = _get_warehouse_spec()
    world = WorldFactory().create_world(spec)

    sim = SimulationEngine()
    cf = CounterfactualEngine()

    horizon = 3
    seed = 42
    scen = Scenario(scenario_id="fact", horizon=horizon, samples=1, seed=seed)
    res = sim.run(world, scen)
    f_traj = res.trajectories[0]

    # Increasing warehouse_a by +50.0
    act = Intervention(
        id="do_a",
        description="do(warehouse_a = 150)",
        parameters={"warehouse_a": 150.0},
    )
    cf_res = cf.run(world=world, observed=f_traj, intervention=act, horizon=horizon, seed=seed)
    cf_traj = cf_res.counterfactual_trajectory

    scm = world.as_scm(horizon=horizon)
    descendants = scm.causal_graph.descendants("resource:warehouse_a")

    if descendants:
        first_desc = next(iter(descendants))
        # Declare a negative monotonic edge when effect is actually positive/neutral
        edge_signs = {("resource:warehouse_a", first_desc): MonotonicDirection.NEGATIVE}

        # If descendant increases, negative declared sign must trigger a monotonicity violation
        raw_target = first_desc.split(":", 1)[-1]
        f_val = f_traj.final_state.resources[raw_target].current
        cf_val = cf_traj.final_state.resources[raw_target].current

        if cf_val > f_val:
            _, violations, _ = evaluate_intervention_selectivity(
                scm=scm,
                factual_trajectory=f_traj,
                counterfactual_trajectory=cf_traj,
                intervened_variable="resource:warehouse_a",
                monotonic_edge_signs=edge_signs,
            )
            assert any(v.violation_type == "monotonicity_sign_inversion" for v in violations)


def test_causal_correctness_gate_summary_and_disclaimer() -> None:
    """CausalCorrectnessGate runs on a world and produces audited report with disclaimer."""
    spec = _get_warehouse_spec()
    world = WorldFactory().create_world(spec)

    gate = CausalCorrectnessGate()
    interventions = [
        {"target_variable": "resource:warehouse_a", "value": 120.0},
    ]
    report = gate.evaluate_world(world=world, interventions=interventions, horizon=3, seed=123)

    assert report.is_causally_correct is True
    assert report.pass_rate >= 0.999
    assert report.violations_count == 0
    assert "resource:warehouse_a" in report.evaluated_interventions

    summary = report.summary_table()
    assert "[PASS]" in summary
    assert "Vafa et al." in summary
    assert "Passing causal-correctness gates certifies consistency" in report.evidence_disclaimer
