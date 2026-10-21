"""High-level Decision Engine orchestrating scenario discovery and multi-robustness evaluation.

Scientific Grounding:
    - Elmachtoub, A. N., & Grigas, P. (2022). "Smart 'Predict-then-Optimize'."
      Management Science, 68(1), 9-26. (Decision-focused loss over prediction accuracy).
    - Lempert, R. J., et al. (2006). "A general, analytic method for generating robust
      strategies and narrative scenarios." Management Science, 52(4), 514-528.
    - McPhail et al. (2018). "Robustness metrics: How choice of metric changes which policy wins."
    - Friston (2010), Da Costa et al. (2024). Active Inference EFE action scoring.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.core.actions import Action, Intervention
from ewm_engine.core.world import World
from ewm_engine.decision.discovery import (
    ScenarioDiscoveryResult,
    run_cart,
    run_prim,
)
from ewm_engine.decision.efe import ActiveInferenceScorer, ExpectedFreeEnergyScore
from ewm_engine.decision.report import DecisionReport
from ewm_engine.decision.robustness import (
    DecisionComparisonResult,
    evaluate_decision_robustness,
)
from ewm_engine.experimentation.doe import generate_lhs
from ewm_engine.experimentation.params import ParameterSpace
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario, ScheduledAction
from ewm_engine.simulation.trajectory import Trajectory


class DecisionEngine:
    """Orchestrates scenario discovery and multi-robustness evaluation under deep uncertainty."""

    def __init__(
        self,
        simulation_engine: SimulationEngine | None = None,
        discovery_algorithm: Literal["prim", "cart"] = "prim",
    ) -> None:
        self.sim_engine = simulation_engine or SimulationEngine()
        self.discovery_algorithm = discovery_algorithm

    def decide(
        self,
        world: World,
        uncertainty_space: ParameterSpace,
        candidates: Mapping[str, Action | Intervention | Sequence[Action] | None],
        objective_metric: str | Callable[[Trajectory], float],
        satisficing_threshold: float,
        higher_is_better: bool = False,
        horizon: int = 5,
        n_samples: int = 40,
        seed: int = 42,
        include_efe: bool = False,
        efe_weight: float = 0.3,
        weight_by_competence: bool = False,
        filter_abstained: bool = False,
        report_title: str = "Decision-Focused Analysis",
    ) -> DecisionReport:
        """Execute decision analysis: ensemble rollouts, scenario discovery, and multi-robustness ranking.

        Args:
            world: Baseline World model.
            uncertainty_space: Specification of uncertain parameters and ranges.
            candidates: Dict mapping candidate strategy ID to action/intervention or None (baseline).
            objective_metric: Metric name or callable extracting scalar performance from a trajectory.
            satisficing_threshold: Threshold defining acceptable performance under Starr's criterion.
            higher_is_better: True if metric represents utility/reward; False if cost/loss.
            horizon: Simulation time horizon.
            n_samples: Number of parameter uncertainty samples (LHS).
            seed: Master random seed for reproducible sampling and rollouts.
            include_efe: Whether to compute Active Inference Expected Free Energy scores.
            efe_weight: Epistemic weight coefficient in EFE scoring (default: 0.3).
            report_title: Title for the emitted DecisionReport.

        Returns:
            Audited, reproducible DecisionReport.
        """
        if not candidates:
            raise ValueError("Candidates dictionary cannot be empty.")

        # 1. Sample uncertainty space using deterministic Latin Hypercube Sampling (LHS)
        parameter_points = generate_lhs(uncertainty_space, n_samples=n_samples, seed=seed)

        # 2. Simulate all candidates across all parameter scenarios
        # We use a dedicated child seed sequence per scenario for strict reproducibility
        outcomes_by_candidate: dict[str, list[float]] = {k: [] for k in candidates}
        trajectories_by_candidate: dict[str, list[Trajectory]] = {k: [] for k in candidates}

        for scenario_idx, pt in enumerate(parameter_points):
            scenario_seed = (seed * 10007 + scenario_idx * 37) % (2**31 - 1)
            # Create base scenario template
            base_scenario = Scenario(
                scenario_id=f"scenario_{scenario_idx}",
                name=f"Scenario-{scenario_idx}",
                horizon=horizon,
                seed=scenario_seed,
            )

            # Parameterize world and scenario
            param_world, param_scenario = uncertainty_space.apply_to_world_and_scenario(
                pt, world, base_scenario
            )

            for cand_id, cand_action in candidates.items():
                cand_scenario = param_scenario
                if cand_action is not None:
                    if isinstance(cand_action, Intervention):
                        cand_scenario = cand_scenario.model_copy(
                            update={"intervention": cand_action}
                        )
                    elif isinstance(cand_action, Action):
                        cand_scenario = cand_scenario.model_copy(
                            update={
                                "scheduled_actions": (ScheduledAction(step=0, action=cand_action),)
                            }
                        )
                    elif isinstance(cand_action, Sequence):
                        actions = tuple(
                            ScheduledAction(step=0, action=a) if isinstance(a, Action) else a
                            for a in cand_action
                        )
                        cand_scenario = cand_scenario.model_copy(
                            update={"scheduled_actions": actions}
                        )

                sim_res = self.sim_engine.run(param_world, cand_scenario)
                traj = sim_res.trajectories[0]
                trajectories_by_candidate[cand_id].append(traj)

                # Extract scalar performance
                if callable(objective_metric):
                    val = float(objective_metric(traj))
                else:
                    val = float(traj.final_metric(objective_metric))

                # Apply epistemic competence weighting or abstention filter if enabled
                if weight_by_competence or filter_abstained:
                    from ewm_engine.evaluation.abstention import CompetenceEstimator

                    comp_est = CompetenceEstimator(max_verified_horizon=horizon)
                    comp = comp_est.evaluate_trajectory(traj)

                    if filter_abstained and not comp.is_competent:
                        val = (
                            satisficing_threshold * 2.0
                            if not higher_is_better
                            else satisficing_threshold * 0.5
                        )
                    elif weight_by_competence:
                        if higher_is_better:
                            val = val * comp.competence_score
                        else:
                            val = val / max(comp.competence_score, 0.1)

                outcomes_by_candidate[cand_id].append(val)

        # 3. Multi-Robustness Evaluation & Epistemic Refusal to Rank
        robustness_result: DecisionComparisonResult = evaluate_decision_robustness(
            outcomes=outcomes_by_candidate,
            satisficing_threshold=satisficing_threshold,
            higher_is_better=higher_is_better,
        )

        # 4. Scenario Discovery: Where does each strategy break?
        vulnerabilities: dict[str, ScenarioDiscoveryResult] = {}
        for cand_id in candidates:
            cand_vals = outcomes_by_candidate[cand_id]
            # Failure definition: performance does NOT meet satisficing threshold
            # (i.e. cost > threshold or utility < threshold)
            if not higher_is_better:
                failures = [1 if v > satisficing_threshold else 0 for v in cand_vals]
            else:
                failures = [1 if v < satisficing_threshold else 0 for v in cand_vals]

            if self.discovery_algorithm == "cart":
                disc_res = run_cart(
                    points=parameter_points,
                    targets=failures,
                    parameter_space=uncertainty_space,
                    threshold=0.5,
                    strategy_id=cand_id,
                )
            else:
                disc_res = run_prim(
                    points=parameter_points,
                    targets=failures,
                    parameter_space=uncertainty_space,
                    threshold=0.5,
                    strategy_id=cand_id,
                )
            vulnerabilities[cand_id] = disc_res

        # 5. Optional Active Inference EFE Scoring
        efe_scores: dict[str, ExpectedFreeEnergyScore] | None = None
        if include_efe:
            scorer = ActiveInferenceScorer(weight_epistemic=efe_weight)
            efe_scores = scorer.score(
                candidate_outcomes=outcomes_by_candidate,
                parameter_points=parameter_points,
                higher_is_better=higher_is_better,
            )

        # 6. Report Fingerprint & Synthesis
        report_meta = {
            "world": getattr(world, "name", "EnterpriseWorld"),
            "uncertainty_space": uncertainty_space.space_hash,
            "sample_count": n_samples,
            "seed": seed,
            "robustness": robustness_result.fingerprint,
            "vulnerabilities": {k: v.discovery_fingerprint for k, v in vulnerabilities.items()},
        }
        report_fp = canonical_sha256(report_meta)

        return DecisionReport(
            title=report_title,
            world_name=getattr(world, "name", "EnterpriseWorld"),
            uncertainty_space_name=uncertainty_space.name,
            sample_count=n_samples,
            candidates=tuple(candidates.keys()),
            robustness=robustness_result,
            vulnerabilities=vulnerabilities,
            efe_scores=efe_scores,
            fingerprint=report_fp,
        )
