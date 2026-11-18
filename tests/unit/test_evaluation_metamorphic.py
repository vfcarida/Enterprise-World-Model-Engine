"""Unit tests for metamorphic relation cards and mutation adequacy evaluation."""

from __future__ import annotations

from pathlib import Path

from ewm_engine.core.spec import WorldSpec
from ewm_engine.evaluation.metamorphic import (
    MR_DETERMINISM,
    MR_ENTITY_PERMUTATION,
    MR_MONOTONICITY,
    MR_SCALING,
    STANDARD_MR_CARDS,
    MetamorphicSuite,
    MutationAdequacyEvaluator,
    MutationType,
)
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario


def _get_warehouse_spec() -> WorldSpec:
    repo_root = Path(__file__).resolve().parent.parent.parent
    path = repo_root / "examples" / "minimal_warehouse" / "spec.yaml"
    return WorldSpec.from_yaml(path)


def test_mr_determinism_card() -> None:
    """MR-1 Determinism: Same spec and seed yields identical state hashes."""
    spec = _get_warehouse_spec()
    scen = Scenario(scenario_id="det_scen", horizon=5, samples=1, seed=42)
    engine = SimulationEngine()

    verdict = MR_DETERMINISM.evaluate(spec, scen, engine)
    assert verdict.passed is True
    assert verdict.discrepancy == 0.0
    assert "Identical state hashes" in verdict.details


def test_mr_scaling_card() -> None:
    """MR-3 Scaling: Scaling resource levels propagates proportionally."""
    spec = _get_warehouse_spec()
    scen = Scenario(scenario_id="scale_scen", horizon=4, samples=1, seed=42)
    engine = SimulationEngine()

    verdict = MR_SCALING.evaluate(spec, scen, engine)
    assert verdict.passed is True
    assert verdict.discrepancy <= 0.25


def test_mr_monotonicity_card() -> None:
    """MR-4 Monotonicity: Increasing initial stock does not decrease final availability."""
    spec = _get_warehouse_spec()
    scen = Scenario(scenario_id="mono_scen", horizon=4, samples=1, seed=42)
    engine = SimulationEngine()

    verdict = MR_MONOTONICITY.evaluate(spec, scen, engine)
    assert verdict.passed is True
    assert verdict.discrepancy == 0.0


def test_mr_entity_permutation_card() -> None:
    """MR-5 Permutation: Permuting entity IDs leaves aggregate resource dynamics invariant."""
    spec = _get_warehouse_spec()
    scen = Scenario(scenario_id="perm_scen", horizon=4, samples=1, seed=42)
    engine = SimulationEngine()

    verdict = MR_ENTITY_PERMUTATION.evaluate(spec, scen, engine)
    assert verdict.passed is True
    assert verdict.discrepancy <= 1e-5


def test_metamorphic_suite_execution() -> None:
    """MetamorphicSuite executes all standard cards and produces summary table."""
    spec = _get_warehouse_spec()
    suite = MetamorphicSuite(cards=STANDARD_MR_CARDS)
    report = suite.run_suite(spec)

    assert report.total_cards == len(STANDARD_MR_CARDS)
    assert report.passed_cards >= 4
    table = report.summary_table()
    assert "Metamorphic Self-Knowledge Suite" in table
    assert "MR-1-DETERMINISM" in table


def test_mutation_adequacy_evaluation() -> None:
    """MutationAdequacyEvaluator generates synthetic mutants and confirms they are killed."""
    spec = _get_warehouse_spec()
    suite = MetamorphicSuite(cards=STANDARD_MR_CARDS)
    evaluator = MutationAdequacyEvaluator(suite=suite)

    mutants = evaluator.generate_mutants(spec)
    assert len(mutants) >= 2
    types = {m.mutation_type for m in mutants}
    assert MutationType.LEAKAGE in types

    score, killed, total, killed_names = evaluator.evaluate_mutation_adequacy(spec)
    assert total > 0
    assert killed > 0
    assert score >= 0.5  # Suite effectively kills injected mutants
    assert len(killed_names) == killed


def test_dummy_metamorphic_structurally_identical_inputs_yield_identical_outputs(
    tmp_path: Path,
) -> None:
    """Dummy metamorphic test confirming that structurally identical inputs yield identical outputs."""
    import json

    import numpy as np

    from ewm_engine.core.actions import Action
    from ewm_engine.core.entities import Entity, Relationship
    from ewm_engine.core.resources import Resource
    from ewm_engine.core.state import WorldState
    from ewm_engine.core.world import World
    from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
    from ewm_engine.simulation.scenario import ScheduledAction

    # Input A: constructed with entities/resources in forward order
    state_a = WorldState(
        entities=[
            Entity(id="node_alpha", type="facility", attributes={"tier": 1, "active": True}),
            Entity(id="node_beta", type="facility", attributes={"tier": 2, "active": True}),
        ],
        relationships=[
            Relationship(
                source="node_alpha",
                target="node_beta",
                type="transfers_to",
                attributes={"capacity": 100.0},
            )
        ],
        resources=[
            Resource(
                id="res_alpha",
                entity_id="node_alpha",
                current=120.0,
                min_value=0.0,
                max_value=300.0,
            ),
            Resource(
                id="res_beta", entity_id="node_beta", current=40.0, min_value=0.0, max_value=300.0
            ),
        ],
        memory={"batch_id": "b_01"},
        active_rules={"enforce_cap": True},
        context={"regime": "baseline"},
        step=0,
        timestamp=0.0,
    )

    # Input B: structurally identical input constructed with permuted insertion order and round-tripped via isolated tmp_path
    state_b_raw = WorldState(
        entities=[
            Entity(id="node_beta", type="facility", attributes={"active": True, "tier": 2}),
            Entity(id="node_alpha", type="facility", attributes={"active": True, "tier": 1}),
        ],
        relationships=[
            Relationship(
                source="node_alpha",
                target="node_beta",
                type="transfers_to",
                attributes={"capacity": 100.0},
            )
        ],
        resources=[
            Resource(
                id="res_beta", entity_id="node_beta", current=40.0, min_value=0.0, max_value=300.0
            ),
            Resource(
                id="res_alpha",
                entity_id="node_alpha",
                current=120.0,
                min_value=0.0,
                max_value=300.0,
            ),
        ],
        memory={"batch_id": "b_01"},
        active_rules={"enforce_cap": True},
        context={"regime": "baseline"},
        step=0,
        timestamp=0.0,
    )
    snapshot_file = tmp_path / "structural_state_b.json"
    snapshot_file.write_text(json.dumps(state_b_raw.to_dict(), sort_keys=True), encoding="utf-8")
    state_b = WorldState.from_dict(json.loads(snapshot_file.read_text(encoding="utf-8")))

    # Metamorphic relation 1: Identical canonical fingerprints on initial state
    assert state_a.state_hash == state_b.state_hash
    assert state_a.fingerprint == state_b.fingerprint

    # Metamorphic relation 2: Identical single-step dynamics transition output under fixed seed
    dynamics = DeterministicTransferDynamics()
    transfer_action = Action(
        id="act_transfer_25",
        type="transfer_resource",
        parameters={
            "source_resource": "res_alpha",
            "target_resource": "res_beta",
            "quantity": 25.0,
        },
    )
    out_a = dynamics.transition(state_a, [transfer_action], [], np.random.default_rng(42))
    out_b = dynamics.transition(state_b, [transfer_action], [], np.random.default_rng(42))
    assert out_a.next_state.state_hash == out_b.next_state.state_hash
    assert out_a.applied_changes == out_b.applied_changes

    # Metamorphic relation 3: Identical multi-step simulation rollout trajectories
    world_a = World(state=state_a, dynamics=dynamics)
    world_b = World(state=state_b, dynamics=dynamics)
    scen = Scenario(
        scenario_id="metamorphic_equivalence",
        horizon=3,
        samples=1,
        seed=42,
        scheduled_actions=(ScheduledAction(step=0, action=transfer_action),),
    )
    engine = SimulationEngine()
    sim_a = engine.run(world_a, scen)
    sim_b = engine.run(world_b, scen)

    assert (
        sim_a.trajectories[0].final_state.fingerprint
        == sim_b.trajectories[0].final_state.fingerprint
    )
    assert [s.state_hash for s in sim_a.trajectories[0].steps] == [
        s.state_hash for s in sim_b.trajectories[0].steps
    ]
