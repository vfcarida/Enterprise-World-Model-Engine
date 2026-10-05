"""Unit and property tests for FEAT-001 (Evaluation Upgrade: Bootstrap CIs, Significance, and Pareto Frontiers)."""

from __future__ import annotations

import numpy as np
import pytest

from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.evaluation.comparison import ScenarioComparison, compare_scenarios
from ewm_engine.evaluation.pareto import (
    ObjectiveDirection,
    ObjectiveSpec,
    compute_pareto_frontier,
)
from ewm_engine.evaluation.uncertainty import (
    CalibrationDiagnostic,
    compute_bootstrap_ci,
    compute_bootstrap_delta,
    compute_crps,
    compute_interval_coverage,
    compute_tail_metrics,
)
from ewm_engine.provenance.metadata import SimulationMetadata
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import SimulationResult, StepRecord, Trajectory


class TestBootstrapConfidenceIntervals:
    """Tests for non-parametric bootstrap confidence intervals."""

    def test_bootstrap_ci_empty_and_single_element(self) -> None:
        """Verify empty and single-element edge cases handle gracefully."""
        ci_empty = compute_bootstrap_ci([])
        assert ci_empty.mean == 0.0
        assert ci_empty.ci_lower == 0.0
        assert ci_empty.ci_upper == 0.0

        ci_single = compute_bootstrap_ci([42.0])
        assert ci_single.mean == 42.0
        assert ci_single.ci_lower == 42.0
        assert ci_single.ci_upper == 42.0
        assert ci_single.standard_error == 0.0

    def test_bootstrap_ci_coverage_and_ordering(self) -> None:
        """Verify bootstrap interval bounds enclose the mean."""
        values = [10.0, 12.0, 9.0, 11.0, 15.0, 8.0, 13.0, 10.5]
        ci = compute_bootstrap_ci(values, confidence_level=0.95, n_resamples=500, seed=42)

        assert ci.ci_lower <= ci.mean <= ci.ci_upper
        assert ci.standard_error > 0.0
        assert ci.confidence_level == 0.95

    def test_bootstrap_ci_confidence_level_scaling(self) -> None:
        """Higher confidence level must produce wider interval."""
        values = list(range(20, 80))
        ci_90 = compute_bootstrap_ci(values, confidence_level=0.90, n_resamples=1000, seed=123)
        ci_99 = compute_bootstrap_ci(values, confidence_level=0.99, n_resamples=1000, seed=123)

        width_90 = ci_90.ci_upper - ci_90.ci_lower
        width_99 = ci_99.ci_upper - ci_99.ci_lower
        assert width_99 > width_90

    def test_bootstrap_ci_seed_determinism(self) -> None:
        """Same seed must produce bitwise identical bootstrap intervals."""
        values = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        ci_a = compute_bootstrap_ci(values, seed=999)
        ci_b = compute_bootstrap_ci(values, seed=999)

        assert ci_a.ci_lower == ci_b.ci_lower
        assert ci_a.ci_upper == ci_b.ci_upper
        assert ci_a.standard_error == ci_b.standard_error


class TestBootstrapDeltaAndSignificance:
    """Tests for counterfactual delta bootstrap uncertainty and hypothesis testing."""

    def test_bootstrap_delta_empty_inputs(self) -> None:
        """Verify empty inputs return neutral non-significant delta."""
        delta = compute_bootstrap_delta([], [1.0, 2.0])
        assert delta.absolute_delta == 0.0
        assert not delta.is_significant
        assert delta.p_value == 1.0

    def test_statistically_significant_reduction(self) -> None:
        """Candidate with clearly lower values must show negative delta and significance."""
        base_vals = [100.0, 105.0, 98.0, 102.0, 101.0, 99.0]
        cand_vals = [20.0, 22.0, 19.0, 21.0, 20.5, 18.5]

        delta = compute_bootstrap_delta(base_vals, cand_vals, seed=42)

        assert delta.absolute_delta < -70.0
        assert delta.percent_delta < -70.0
        assert delta.ci_upper < 0.0  # Entire CI is negative
        assert delta.is_significant
        assert delta.p_value < 0.01

    def test_non_significant_difference_encloses_zero(self) -> None:
        """Overlapping distributions must produce a CI containing zero and is_significant=False."""
        rng = np.random.default_rng(42)
        base_vals = list(rng.normal(50.0, 5.0, size=20))
        cand_vals = list(rng.normal(50.2, 5.0, size=20))

        delta = compute_bootstrap_delta(base_vals, cand_vals, seed=42)

        # 95% CI should span across 0.0
        assert delta.ci_lower <= 0.0 <= delta.ci_upper
        assert not delta.is_significant
        assert delta.p_value > 0.05

    def test_bootstrap_delta_determinism(self) -> None:
        """Identical seeds must produce bitwise identical delta metrics."""
        base = [10.0, 20.0, 30.0]
        cand = [15.0, 25.0, 35.0]

        delta_1 = compute_bootstrap_delta(base, cand, seed=777)
        delta_2 = compute_bootstrap_delta(base, cand, seed=777)

        assert delta_1.ci_lower == delta_2.ci_lower
        assert delta_1.ci_upper == delta_2.ci_upper
        assert delta_1.p_value == delta_2.p_value


class TestParetoFrontierAnalysis:
    """Tests for multi-objective optimization and Pareto dominance."""

    def test_pareto_dominance_and_frontier(self) -> None:
        """Test Pareto partitioning across conflicting objectives."""
        # 3 Scenarios:
        # Policy A: Cost=10 (low), Violations=50 (high) -> Trade-off (Pareto)
        # Policy B: Cost=20 (medium), Violations=10 (low) -> Trade-off (Pareto)
        # Policy C: Cost=30 (high), Violations=60 (worst) -> Dominated by both A and B
        scores = {
            "PolicyA": {"cost": 10.0, "violations": 50.0},
            "PolicyB": {"cost": 20.0, "violations": 10.0},
            "PolicyC": {"cost": 30.0, "violations": 60.0},
        }
        objectives = [
            ObjectiveSpec(metric_name="cost", direction=ObjectiveDirection.MINIMIZE),
            ObjectiveSpec(metric_name="violations", direction=ObjectiveDirection.MINIMIZE),
        ]

        frontier_res = compute_pareto_frontier(scores, objectives)

        assert set(frontier_res.frontier) == {"PolicyA", "PolicyB"}
        assert "PolicyC" not in frontier_res.frontier
        assert "PolicyA" in frontier_res.dominated_by["PolicyC"]
        assert "PolicyB" in frontier_res.dominated_by["PolicyC"]
        assert "PolicyC" in frontier_res.dominates["PolicyA"]
        assert "PolicyC" in frontier_res.dominates["PolicyB"]

    def test_weighted_utility_scoring(self) -> None:
        """Test normalized weighted utility scoring when weights are provided."""
        scores = {
            "PolicyA": {"cost": 10.0, "service_level": 90.0},
            "PolicyB": {"cost": 20.0, "service_level": 95.0},
        }
        objectives = [
            ObjectiveSpec(metric_name="cost", direction=ObjectiveDirection.MINIMIZE, weight=0.5),
            ObjectiveSpec(
                metric_name="service_level", direction=ObjectiveDirection.MAXIMIZE, weight=0.5
            ),
        ]

        frontier = compute_pareto_frontier(scores, objectives)
        assert frontier.weighted_scores is not None
        assert "PolicyA" in frontier.weighted_scores
        assert "PolicyB" in frontier.weighted_scores
        # Scores must be bounded within [0, 1]
        assert 0.0 <= frontier.weighted_scores["PolicyA"] <= 1.0
        assert 0.0 <= frontier.weighted_scores["PolicyB"] <= 1.0

    def test_epistemic_warning_present(self) -> None:
        """Pareto analysis must include an epistemic humility warning against unweighted auto-winners."""
        scores = {"Base": {"m": 1.0}, "Cand": {"m": 2.0}}
        objectives = [ObjectiveSpec(metric_name="m", direction=ObjectiveDirection.MINIMIZE)]
        frontier = compute_pareto_frontier(scores, objectives)
        assert "No single winner is declared" in frontier.epistemic_warning


@pytest.fixture
def comparison_simulation_results(
    sample_world_state: WorldState,
) -> tuple[SimulationResult, SimulationResult, SimulationResult]:
    """Create baseline and two candidate simulation results."""
    meta = SimulationMetadata(
        scenario_id="eval_test",
        world_hash=sample_world_state.state_hash,
        dynamics_name="DeterministicTransfer",
        random_seed=42,
        horizon=2,
        samples=4,
    )

    def make_res(name: str, unmet_vals: list[float], cost_vals: list[float]) -> SimulationResult:
        trajs = []
        for idx, (unmet, cost) in enumerate(zip(unmet_vals, cost_vals, strict=True)):
            t = Trajectory(sample_id=idx, seed=42 + idx, initial_state=sample_world_state)
            t.steps.append(
                StepRecord(
                    step=0,
                    timestamp=0.0,
                    state_hash="",
                    transition_result=TransitionResult(next_state=sample_world_state),
                    step_metrics={"unmet_demand": unmet, "holding_cost": cost},
                )
            )
            trajs.append(t)
        return SimulationResult(
            scenario=Scenario(name=name, horizon=2, samples=len(trajs), seed=42),
            metadata=meta,
            trajectories=trajs,
        )

    base_res = make_res("Baseline", [50.0, 52.0, 48.0, 50.0], [100.0, 102.0, 98.0, 100.0])
    cand_a_res = make_res("PolicyA", [10.0, 12.0, 8.0, 10.0], [150.0, 152.0, 148.0, 150.0])
    cand_b_res = make_res("PolicyB", [70.0, 75.0, 68.0, 72.0], [180.0, 185.0, 175.0, 180.0])

    return base_res, cand_a_res, cand_b_res


class TestScenarioComparisonUpgrade:
    """Integration tests for compare_scenarios with bootstrap CIs and Pareto analysis."""

    def test_compare_scenarios_bootstrap_and_pareto(
        self,
        comparison_simulation_results: tuple[SimulationResult, SimulationResult, SimulationResult],
    ) -> None:
        """Verify full compare_scenarios run with bootstrap CIs and Pareto objectives."""
        base, cand_a, cand_b = comparison_simulation_results

        comparison = compare_scenarios(
            baseline=base,
            candidates=[cand_a, cand_b],
            confidence_level=0.95,
            n_bootstrap=500,
            seed=42,
            objectives={
                "unmet_demand": ObjectiveDirection.MINIMIZE,
                "holding_cost": ObjectiveDirection.MINIMIZE,
            },
        )

        assert isinstance(comparison, ScenarioComparison)
        assert "PolicyA" in comparison.bootstrap_deltas
        assert "PolicyB" in comparison.bootstrap_deltas

        # PolicyA unmet_demand delta vs Baseline is ~ -40 (statistically significant reduction)
        unmet_delta_a = comparison.bootstrap_deltas["PolicyA"]["unmet_demand"]
        assert unmet_delta_a.absolute_delta < -35.0
        assert unmet_delta_a.is_significant
        assert unmet_delta_a.ci_upper < 0.0

        # Multi-objective Pareto checks:
        assert comparison.pareto_frontier is not None
        # PolicyA (unmet: 10, cost: 150), Baseline (unmet: 50, cost: 100) -> Trade-off (Pareto)
        # PolicyB (unmet: ~71, cost: ~180) -> Dominated by Baseline (and PolicyA)
        assert set(comparison.pareto_frontier.frontier) == {"Baseline", "PolicyA"}
        assert "PolicyB" in comparison.pareto_frontier.dominated_by
        assert "Baseline" in comparison.pareto_frontier.dominated_by["PolicyB"]

        # Serialization to dictionary
        comp_dict = comparison.to_dict()
        assert "bootstrap_deltas" in comp_dict
        assert "pareto_frontier" in comp_dict
        assert "Baseline" in comp_dict["scenarios"]
        assert "deltas_vs_baseline" in comp_dict

        # Formatted summary table output
        table_str = comparison.summary_table()
        assert "Multi-Objective Pareto Analysis" in table_str
        assert "Non-Dominated Pareto Frontier" in table_str
        assert "Baseline" in table_str
        assert "PolicyA" in table_str
        assert "*" in table_str  # Significant delta indicator


class TestCalibrationDiagnosticsAndTailRisk:
    """Tests for interval coverage, continuous ranked probability score (CRPS), and tail risk."""

    def test_interval_coverage_computation(self) -> None:
        """Verify empirical coverage rate calculation."""
        actuals = [10.0, 20.0, 30.0, 40.0]
        lower_bounds = [5.0, 15.0, 25.0, 45.0]
        upper_bounds = [15.0, 25.0, 35.0, 50.0]

        # Targets 10.0, 20.0, 30.0 are covered; 40.0 is not in [45.0, 50.0] -> 3/4 = 0.75
        coverage = compute_interval_coverage(actuals, lower_bounds, upper_bounds)
        assert coverage == 0.75

        # Empty inputs
        assert compute_interval_coverage([], [], []) == 0.0

    def test_crps_exact_and_dispersed(self) -> None:
        """Verify sample CRPS computation."""
        # Exact match: all samples equal to actual -> CRPS = 0.0
        samples_perfect = [25.0, 25.0, 25.0, 25.0]
        assert compute_crps(samples_perfect, 25.0) == 0.0

        # Non-zero error
        samples_shifted = [20.0, 20.0, 20.0]
        crps_shifted = compute_crps(samples_shifted, 25.0)
        assert crps_shifted == 5.0

        # Dispersed samples around target
        samples_spread = [10.0, 20.0, 30.0]
        crps_spread = compute_crps(samples_spread, 20.0)
        assert crps_spread >= 0.0

    def test_calibration_diagnostic_model(self) -> None:
        """Verify CalibrationDiagnostic model attributes and epistemic guardrail."""
        diag = CalibrationDiagnostic(
            nominal_coverage=0.90,
            empirical_coverage=0.88,
            coverage_error=-0.02,
            mean_crps=1.45,
        )
        assert diag.nominal_coverage == 0.90
        assert diag.empirical_coverage == 0.88
        assert diag.coverage_error == -0.02
        assert "Rollout calibration measures internal consistency" in diag.epistemic_note

    def test_compute_tail_metrics(self) -> None:
        """Verify CVaR and worst-k extraction."""
        values = [10.0, 50.0, 2.0, 100.0, 5.0, 30.0]
        # Sorted: [2.0, 5.0, 10.0, 30.0, 50.0, 100.0]
        tail = compute_tail_metrics(values, alpha=0.33, worst_k=2)
        assert tail["worst_k_values"] == [2.0, 5.0]
        assert tail["cvar"] == pytest.approx((2.0 + 5.0) / 2.0)
        assert tail["tail_threshold"] == 5.0

        # Empty handling
        empty_tail = compute_tail_metrics([])
        assert empty_tail["cvar"] == 0.0
        assert empty_tail["worst_k_values"] == []


class TestUserObjectiveRanking:
    """Tests verifying the 'No objective => no ranking' invariant and explicit ranking."""

    def test_no_objective_means_no_ranking(
        self,
        comparison_simulation_results: tuple[SimulationResult, SimulationResult, SimulationResult],
    ) -> None:
        """When objective is omitted, rankings MUST be None (no automated winner)."""
        base, cand_a, cand_b = comparison_simulation_results

        comp_no_obj = compare_scenarios(baseline=base, candidates=[cand_a, cand_b])
        assert comp_no_obj.rankings is None
        assert comp_no_obj.to_dict()["rankings"] is None
        assert "User Objective Ranking" not in comp_no_obj.summary_table()

    def test_user_objective_ranks_candidates(
        self,
        comparison_simulation_results: tuple[SimulationResult, SimulationResult, SimulationResult],
    ) -> None:
        """When a user objective callable is supplied, candidates are ranked deterministically."""
        base, cand_a, cand_b = comparison_simulation_results

        # User objective: heavily penalize unmet_demand, lightly penalize holding_cost
        def custom_utility(m: dict[str, float]) -> float:
            return -10.0 * m["unmet_demand"] - 0.1 * m["holding_cost"]

        comp_with_obj = compare_scenarios(
            baseline=base,
            candidates=[cand_a, cand_b],
            objective=custom_utility,
        )

        assert comp_with_obj.rankings is not None
        assert len(comp_with_obj.rankings) == 3
        # PolicyA has low unmet_demand (~10.0) -> best score
        # Baseline has medium unmet_demand (~50.0) -> second
        # PolicyB has high unmet_demand (~71.0) -> lowest score
        ranked_names = [name for name, _score in comp_with_obj.rankings]
        assert ranked_names == ["PolicyA", "Baseline", "PolicyB"]

        # Serialization
        comp_dict = comp_with_obj.to_dict()
        assert comp_dict["rankings"] is not None
        assert comp_dict["rankings"][0]["scenario"] == "PolicyA"

        # Summary table renders the ranking section
        table_str = comp_with_obj.summary_table()
        assert "=== User Objective Ranking ===" in table_str
        assert "1. PolicyA" in table_str
        assert "2. Baseline" in table_str
        assert "3. PolicyB" in table_str


class TestEvaluationProperties:
    """Property assertions on uncertainty bounds and Pareto non-dominance."""

    def test_delta_ci_brackets_point_estimate(self) -> None:
        """Confidence interval must enclose the point delta estimate."""
        rng = np.random.default_rng(101)
        base = list(rng.normal(100.0, 15.0, size=25))
        cand = list(rng.normal(85.0, 10.0, size=25))

        delta = compute_bootstrap_delta(base, cand, confidence_level=0.95, seed=42)
        assert delta.ci_lower <= delta.absolute_delta <= delta.ci_upper

    def test_pareto_frontier_is_non_dominated(self) -> None:
        """No scenario on the Pareto frontier can be dominated by any scenario in the set."""
        scores = {f"Scenario_{i}": {"m1": float(i), "m2": float(10 - i)} for i in range(11)}
        objectives = [
            ObjectiveSpec(metric_name="m1", direction=ObjectiveDirection.MINIMIZE),
            ObjectiveSpec(metric_name="m2", direction=ObjectiveDirection.MINIMIZE),
        ]

        frontier_res = compute_pareto_frontier(scores, objectives)

        # For all pairs in the frontier, neither can dominate the other
        for scn_name in frontier_res.frontier:
            assert len(frontier_res.dominated_by[scn_name]) == 0
