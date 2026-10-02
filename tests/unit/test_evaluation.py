"""Unit tests for evaluation metrics, uncertainty quantification, and scenario comparisons."""

from __future__ import annotations

from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.evaluation.comparison import compare_scenarios
from ewm_engine.evaluation.metrics import (
    DistributionalEquityMetric,
)
from ewm_engine.evaluation.uncertainty import summarize_distribution
from ewm_engine.provenance.metadata import SimulationMetadata
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import SimulationResult, StepRecord, Trajectory


def test_summarize_distribution() -> None:
    """Test distribution percentiles and mean."""
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    dist = summarize_distribution(values)
    assert dist.mean == 30.0
    assert dist.median == 30.0
    assert dist.min_val == 10.0
    assert dist.max_val == 50.0
    assert dist.iqr == 40.0 - 20.0


def test_distributional_equity_metric(sample_world_state: WorldState) -> None:
    """Test Gini-based equity metric on equal vs unequal states."""
    metric = DistributionalEquityMetric(resource_ids=["stock_wh1", "stock_wh2"])

    # Equal allocation: wh1=100, wh2=100
    equal_state = sample_world_state.update_resource("stock_wh2", new_value=100.0)
    traj_equal = Trajectory(sample_id=0, seed=42, initial_state=equal_state)
    traj_equal._final_state = equal_state

    score_equal = metric.compute(traj_equal)
    assert score_equal == 1.0

    # Highly unequal allocation: wh1=190, wh2=10
    unequal_state = sample_world_state.update_resource(
        "stock_wh1", new_value=190.0
    ).update_resource("stock_wh2", new_value=10.0)
    traj_unequal = Trajectory(sample_id=1, seed=42, initial_state=unequal_state)
    traj_unequal._final_state = unequal_state

    score_unequal = metric.compute(traj_unequal)
    assert score_unequal < 1.0


def test_scenario_comparison_summary(sample_world_state: WorldState) -> None:
    """Test comparing baseline and candidate policy results."""
    scn_base = Scenario(name="Baseline", horizon=2, samples=2, seed=42)
    meta_base = SimulationMetadata(
        scenario_id="s_base",
        world_hash=sample_world_state.state_hash,
        dynamics_name="Test",
        random_seed=42,
        horizon=2,
        samples=2,
    )
    t1 = Trajectory(0, 42, sample_world_state)
    t1.steps.append(
        StepRecord(
            step=0,
            timestamp=0.0,
            state_hash="",
            transition_result=TransitionResult(next_state=sample_world_state),
            step_metrics={"unmet_demand": 50.0},
        )
    )
    t2 = Trajectory(1, 43, sample_world_state)
    t2.steps.append(
        StepRecord(
            step=0,
            timestamp=0.0,
            state_hash="",
            transition_result=TransitionResult(next_state=sample_world_state),
            step_metrics={"unmet_demand": 40.0},
        )
    )
    res_base = SimulationResult(scenario=scn_base, metadata=meta_base, trajectories=[t1, t2])

    scn_cand = Scenario(name="PolicyA", horizon=2, samples=2, seed=42)
    t3 = Trajectory(0, 42, sample_world_state)
    t3.steps.append(
        StepRecord(
            step=0,
            timestamp=0.0,
            state_hash="",
            transition_result=TransitionResult(next_state=sample_world_state),
            step_metrics={"unmet_demand": 10.0},
        )
    )
    t4 = Trajectory(1, 43, sample_world_state)
    t4.steps.append(
        StepRecord(
            step=0,
            timestamp=0.0,
            state_hash="",
            transition_result=TransitionResult(next_state=sample_world_state),
            step_metrics={"unmet_demand": 20.0},
        )
    )
    res_cand = SimulationResult(scenario=scn_cand, metadata=meta_base, trajectories=[t3, t4])

    comparison = compare_scenarios(
        baseline=res_base, candidates=[res_cand], metrics=["unmet_demand"]
    )
    comp_dict = comparison.to_dict()
    assert "Baseline" in comp_dict["scenarios"]
    assert "PolicyA" in comp_dict["scenarios"]

    # Baseline mean = 45.0, PolicyA mean = 15.0 -> Delta = -30.0
    delta_data = comp_dict["deltas_vs_baseline"]["PolicyA"]["unmet_demand"]
    assert delta_data["absolute_delta"] == -30.0

    table_text = comparison.summary_table()
    assert "Scenario Comparison" in table_text
    assert "PolicyA" in table_text
    assert "-30.00" in table_text
