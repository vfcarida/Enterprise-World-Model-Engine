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
