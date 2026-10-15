"""Unified World Model Reliability Evaluation Suite.

Aggregates causal correctness, metamorphic relation satisfaction, mutation adequacy,
epistemic abstention, and Physics-IQ physical reasoning into a single auditable artifact.

Scientific Anchors:
    - Vafa et al. (NeurIPS 2024, arXiv:2406.03689): "Evaluating the World Model Implicit
      in a Generative Model" (Intervention-selectivity & internal causal maps).
    - Vafa et al. (ICML 2025, arXiv:2507.06952): "What Has a Foundation Model Found?
      (Inductive-bias probes beyond surface rollout error)".
    - Kang et al. (ICML 2025, arXiv:2411.02385): "How Far is Video Generation from World Model".
    - Motamed et al. (2025, arXiv:2501.09038): "Physics-IQ: Evaluating Physical Reasoning
      in World Models beyond Perceptual Realism".
    - RWM-U (arXiv:2504.16680): Uncertainty-aware world modeling & epistemic abstention.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.spec import WorldFactory, WorldSpec
from ewm_engine.evaluation.abstention import CompetenceEstimator, PhysicsIQGuardrail
from ewm_engine.evaluation.causal import CausalCorrectnessGate, CausalCorrectnessReport
from ewm_engine.evaluation.metamorphic import (
    MetamorphicSuite,
    MetamorphicSuiteReport,
    MutationAdequacyEvaluator,
)
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario

if TYPE_CHECKING:
    pass


class WorldModelReliabilityReport(BaseModel):
    """Auditable reliability report certifying world model properties beyond rollout error."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_reliable: bool = Field(
        description="True if all deterministic gates (causal, metamorphic, physics-iq) passed."
    )
    causal_report: CausalCorrectnessReport = Field(
        description="Results of interventional selectivity and monotonicity probes."
    )
    metamorphic_report: MetamorphicSuiteReport = Field(
        description="Results of metamorphic relation cards evaluated against the engine."
    )
    mutation_adequacy_score: float = Field(
        description="Fraction of synthetic defect mutants killed by the metamorphic suite."
    )
    epistemic_abstention_rate: float = Field(
        description="Fraction of long-horizon or out-of-support trajectories triggering abstention."
    )
    mean_competence_score: float = Field(
        description="Average epistemic competence score across evaluated trajectories."
    )
    physics_iq_score: float = Field(
        description="Physics-IQ adherence score: conservation and non-negativity compliance."
    )
    evidence_disclaimer: str = Field(
        default=(
            "Passing reliability gates certifies model internal consistency, adherence to declared "
            "physical/accounting constraints, and conformity with the assumed structural causal graph. "
            "It does NOT assert observational point forecasts are infallible, nor does it automatically "
            "upgrade EvidenceLevel without causal identification."
        ),
        description="Anti-overclaiming disclaimer attached to all reliability reports.",
    )

    def summary_table(self) -> str:
        """Render concise plain-text summary table."""
        lines: list[str] = []
        lines.append("World Model Reliability Report (Beyond Rollout Error)")
        lines.append("=" * 76)
        status = "[PASS]" if self.is_reliable else "[FAIL]"
        lines.append(f"Overall Gate Verdict:        {status} {'CERTIFIED RELIABLE' if self.is_reliable else 'UNRELIABLE / VIOLATIONS'}")
        lines.append(f"Causal Selectivity Pass Rate: {self.causal_report.pass_rate * 100:.1f}% ({'PASS' if self.causal_report.is_causally_correct else 'FAIL'})")
        lines.append(f"Metamorphic Pass Rate:        {self.metamorphic_report.pass_rate * 100:.1f}% ({self.metamorphic_report.passed_cards}/{self.metamorphic_report.total_cards})")
        lines.append(f"Mutation Adequacy Score:      {self.mutation_adequacy_score * 100:.1f}%")
        lines.append(f"Mean Epistemic Competence:    {self.mean_competence_score * 100:.1f}%")
        lines.append(f"Epistemic Abstention Rate:    {self.epistemic_abstention_rate * 100:.1f}%")
        lines.append(f"Physics-IQ Score:             {self.physics_iq_score * 100:.1f}%")
        lines.append("-" * 76)
        lines.append("Physics-IQ Guardrail: Plausibility/realism is NEVER used as a correctness proxy.")
        lines.append(f"Audit Note: {self.evidence_disclaimer}")
        lines.append("=" * 76)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Render rich GitHub-flavored Markdown report."""
        lines: list[str] = [
            "# World Model Reliability Report",
            "",
            "> Evaluates world model faithfulness, causal consistency, and epistemic honesty beyond rollout error.",
            "> Anchored in Vafa et al. (NeurIPS 2024 / ICML 2025), Physics-IQ (Motamed et al. 2025), and RWM-U.",
            "",
            "## Executive Summary",
            "",
            f"- **Overall Reliability Gate**: **{'PASSED' if self.is_reliable else 'FAILED'}**",
            f"- **Causal Correctness**: {self.causal_report.pass_rate * 100:.1f}% pass rate ({self.causal_report.violations_count} violations)",
            f"- **Metamorphic Invariants**: {self.metamorphic_report.pass_rate * 100:.1f}% ({self.metamorphic_report.passed_cards}/{self.metamorphic_report.total_cards} MR-cards satisfied)",
            f"- **Mutation Adequacy**: {self.mutation_adequacy_score * 100:.1f}% of non-equivalent mutants killed",
            f"- **Physics-IQ (Physical Law Adherence)**: {self.physics_iq_score * 100:.1f}%",
            f"- **Mean Epistemic Competence**: {self.mean_competence_score * 100:.1f}% (Abstention rate: {self.epistemic_abstention_rate * 100:.1f}%)",
            "",
            "## Detailed Reliability Metrics",
            "",
            "| Reliability Pillar | Metric | Observed Value | Gate Criterion | Status |",
            "|---|---|---|---|---|",
            f"| Causal Selectivity | Descendant-only divergence | {self.causal_report.pass_rate * 100:.1f}% | 100.0% | {'PASS' if self.causal_report.is_causally_correct else 'FAIL'} |",
            f"| Metamorphic Oracle | Invariant card satisfaction | {self.metamorphic_report.pass_rate * 100:.1f}% | 100.0% | {'PASS' if self.metamorphic_report.all_passed else 'FAIL'} |",
            f"| Mutation Testing | Injected mutant kill rate | {self.mutation_adequacy_score * 100:.1f}% | >= 80.0% | {'PASS' if self.mutation_adequacy_score >= 0.8 else 'FAIL'} |",
            f"| Physics-IQ Guardrail | Non-negativity & Conservation | {self.physics_iq_score * 100:.1f}% | 100.0% | {'PASS' if self.physics_iq_score >= 0.999 else 'FAIL'} |",
            f"| Epistemic Abstention | Long-horizon refusal rate | {self.epistemic_abstention_rate * 100:.1f}% | Audited signal | INFO |",
            "",
            "## Metamorphic Relation Cards Breakdown",
            "",
            "| MR ID | Relation Name | Verdict | Discrepancy | Details |",
            "|---|---|---|---|---|",
        ]
        for v in self.metamorphic_report.verdicts:
            status = "PASS" if v.passed else "FAIL"
            lines.append(f"| {v.mr_id} | {v.name} | {status} | {v.discrepancy:.2e} | {v.details} |")

        if self.causal_report.violations:
            lines.extend([
                "",
                "## Detected Causal Violations",
                "",
                "| Intervened Variable | Affected Variable | Step | Violation Type | Delta |",
                "|---|---|---|---|---|",
            ])
            for viol in self.causal_report.violations:
                lines.append(
                    f"| {viol.intervened_variable} | {viol.affected_variable} | {viol.step} | {viol.violation_type} | {viol.observed_delta:+.4f} |"
                )

        lines.extend([
            "",
            "## Scientific Doctrine: Physics-IQ & Epistemic Honesty",
            "",
            "> [!IMPORTANT]",
            "> **Anti-Plausibility Guardrail**: Under no circumstances is visual realism or high generative",
            "> likelihood treated as evidence of physical correctness. Models can generate visually plausible",
            "> rollouts while catastrophically violating conservation laws and directional causality.",
            "> Correctness is certified solely against the symbolic rule kernel and structural invariants.",
            "",
            "> [!NOTE]",
            f"> **Evidence Disclaimer**: {self.evidence_disclaimer}",
        ])
        return "\n".join(lines)

    def to_json(self, indent: int = 2) -> str:
        """Export serialized JSON representation of report."""
        return json.dumps(self.model_dump(), indent=indent)


def evaluate_world_model_reliability(
    spec: WorldSpec,
    interventions: Sequence[dict[str, Any]] | None = None,
    horizon: int = 6,
    seed: int = 42,
    conserved_groups: Sequence[Sequence[str]] | None = None,
) -> WorldModelReliabilityReport:
    """Execute end-to-end reliability evaluation on a WorldSpec.

    Evaluates:
        1. Causal intervention selectivity and monotonicity gates.
        2. Metamorphic relation cards across input transformations.
        3. Mutation adequacy score against synthetic defects.
        4. Epistemic competence and out-of-competence abstention rate.
        5. Physics-IQ conservation and non-negativity compliance.

    Args:
        spec: Target WorldSpec defining the system.
        interventions: Specific causal interventions do(X = x) to test. If None, default
                       interventions are auto-derived from available resources.
        horizon: Simulation horizon for test rollouts.
        seed: Random seed for deterministic simulation.
        conserved_groups: Declared resource subsets that must satisfy strict conservation.

    Returns:
        Comprehensive WorldModelReliabilityReport.
    """
    factory = WorldFactory()
    world = factory.create_world(spec)
    sim_engine = SimulationEngine()

    # 1. Derive default interventions if not provided
    test_interventions: list[dict[str, Any]] = []
    if interventions is not None:
        test_interventions = list(interventions)
    elif spec.resources:
        for r in spec.resources[:2]:
            test_interventions.append({
                "target_variable": f"resource:{r.id}",
                "value": float(r.current * 1.5 + 5.0),
            })

    # 2. Causal Correctness Gate
    causal_gate = CausalCorrectnessGate()
    causal_report = causal_gate.evaluate_world(
        world=world,
        interventions=test_interventions,
        horizon=horizon,
        seed=seed,
    )

    # 3. Metamorphic Relations Suite & Mutation Adequacy
    mr_suite = MetamorphicSuite(sim_engine=sim_engine)
    base_scen = Scenario(scenario_id="rel_eval_scen", horizon=horizon, samples=1, seed=seed)
    mr_report = mr_suite.run_suite(spec=spec, scenario=base_scen)

    mut_eval = MutationAdequacyEvaluator(suite=mr_suite)
    mut_score, _killed, _total, _names = mut_eval.evaluate_mutation_adequacy(
        clean_spec=spec, scenario=base_scen
    )

    # Attach mutation score to metamorphic report copy
    mr_report = mr_report.model_copy(update={"mutation_adequacy_score": mut_score})

    # 4. Epistemic Competence & Abstention Evaluation
    # We evaluate standard horizon rollout and extended out-of-support rollout
    comp_est = CompetenceEstimator(max_verified_horizon=horizon)
    standard_scen = Scenario(scenario_id="std_scen", horizon=horizon, samples=1, seed=seed)
    std_res = sim_engine.run(world, standard_scen)
    std_traj = std_res.trajectories[0]

    extended_scen = Scenario(scenario_id="ext_scen", horizon=horizon * 3, samples=1, seed=seed)
    ext_res = sim_engine.run(world, extended_scen)
    ext_traj = ext_res.trajectories[0]

    std_comp = comp_est.evaluate_trajectory(std_traj)
    ext_comp = comp_est.evaluate_trajectory(ext_traj)

    eval_trajs = [std_traj, ext_traj]
    eval_comps = [std_comp, ext_comp]

    abstained_count = sum(1 for c in eval_comps if not c.is_competent)
    abstention_rate = float(abstained_count / len(eval_comps))
    mean_competence = float(sum(c.competence_score for c in eval_comps) / len(eval_comps))

    # 5. Physics-IQ Guardrail
    piq_guard = PhysicsIQGuardrail()
    physics_scores: list[float] = []
    for t in eval_trajs:
        score_t, _sound, _viols = piq_guard.evaluate_physics_iq(
            trajectory=t,
            conserved_groups=conserved_groups,
        )
        physics_scores.append(score_t)
    mean_physics_iq = float(sum(physics_scores) / len(physics_scores)) if physics_scores else 1.0

    # Determine overall reliability pass
    is_reliable = (
        causal_report.is_causally_correct
        and mr_report.all_passed
        and mut_score >= 0.8
        and mean_physics_iq >= 0.999
    )

    return WorldModelReliabilityReport(
        is_reliable=is_reliable,
        causal_report=causal_report,
        metamorphic_report=mr_report,
        mutation_adequacy_score=mut_score,
        epistemic_abstention_rate=abstention_rate,
        mean_competence_score=mean_competence,
        physics_iq_score=mean_physics_iq,
    )
