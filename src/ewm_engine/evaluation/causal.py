"""Causal-correctness and intervention-selectivity evaluation gates.

Scientific Anchors:
    - Vafa et al. (NeurIPS 2024, arXiv:2406.03689): "Evaluating the World Model Implicit
      in a Generative Model" (Intervention-selectivity and causal structure probes).
    - Vafa et al. (ICML 2025, arXiv:2507.06952): "What Has a Foundation Model Found?
      (Inductive-bias and directional causal probes)".
    - Kang et al. (ICML 2025, arXiv:2411.02385): "How Far is Video Generation from World Model".
    - Pearl, J. (2009): "Causality: Models, Reasoning, and Inference", Cambridge University Press.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ewm_engine.core.scm import StructuralCausalModel
    from ewm_engine.core.world import World
    from ewm_engine.simulation.trajectory import Trajectory


class MonotonicDirection(StrEnum):
    """Declared monotonic direction of a direct causal edge."""

    POSITIVE = "positive"  # Increasing X increases Y
    NEGATIVE = "negative"  # Increasing X decreases Y
    UNSPECIFIED = "unspecified"


class InterventionSelectivityViolation(BaseModel):
    """Diagnostic record of an intervention-selectivity or monotonicity failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intervened_variable: str = Field(description="Name of the intervened variable do(X = x).")
    affected_variable: str = Field(description="Name of the variable exhibiting invalid divergence.")
    step: int = Field(description="Simulation time step where violation was observed.")
    violation_type: Literal[
        "non_descendant_divergence",
        "monotonicity_sign_inversion",
        "spurious_cross_talk",
    ] = Field(description="Classification of causal failure.")
    observed_delta: float = Field(description="Observed counterfactual divergence (Y_cf - Y_factual).")
    expected_behavior: str = Field(description="Human-readable specification of the violated causal invariant.")
    message: str = Field(description="Detailed diagnostic message.")


class CausalCorrectnessReport(BaseModel):
    """Audit report evaluating whether model response to do(x) conforms to the causal graph."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_causally_correct: bool = Field(
        description="True if all intervention-selectivity and monotonicity checks passed."
    )
    pass_rate: float = Field(description="Fraction of evaluated variable-step pairs conforming to causal invariants.")
    total_checks: int = Field(description="Total number of evaluated variable-step counterfactual pairs.")
    violations_count: int = Field(description="Count of identified causal violations.")
    violations: tuple[InterventionSelectivityViolation, ...] = Field(
        default_factory=tuple, description="Immutable sequence of detected causal failures."
    )
    evaluated_interventions: tuple[str, ...] = Field(
        default_factory=tuple, description="Names of intervened variables tested."
    )
    evidence_disclaimer: str = Field(
        default=(
            "Passing causal-correctness gates certifies consistency with the assumed structural causal graph. "
            "It does NOT prove real-world causality or upgrade observational evidence without causal identification."
        ),
        description="Anti-overclaiming disclaimer attached to all causal evaluation results.",
    )

    def summary_table(self) -> str:
        """Render formatted console summary table of the causal correctness evaluation."""
        lines: list[str] = []
        lines.append("Causal-Correctness Evaluation Gate (Vafa et al., NeurIPS 2024)")
        lines.append("=" * 72)
        status_marker = "[PASS]" if self.is_causally_correct else "[FAIL]"
        lines.append(f"Gate Verdict:           {status_marker} {'CONSISTENT' if self.is_causally_correct else 'VIOLATIONS DETECTED'}")
        lines.append(f"Causal Pass Rate:       {self.pass_rate * 100:.2f}%")
        lines.append(f"Total Checks:           {self.total_checks}")
        lines.append(f"Violations Identified:  {self.violations_count}")
        lines.append(f"Interventions Tested:   {', '.join(self.evaluated_interventions) if self.evaluated_interventions else 'None'}")

        if self.violations:
            lines.append("\nTop Causal Violations:")
            lines.append("-" * 72)
            lines.append(f"{'Intervention':<20} {'Affected Var':<20} {'Step':<6} {'Type':<24}")
            lines.append("-" * 72)
            for v in self.violations[:5]:
                lines.append(f"{v.intervened_variable:<20} {v.affected_variable:<20} {v.step:<6} {v.violation_type:<24}")
            if len(self.violations) > 5:
                lines.append(f"... and {len(self.violations) - 5} more violations.")

        lines.append("-" * 72)
        lines.append(f"Honesty Note: {self.evidence_disclaimer}")
        return "\n".join(lines)


def evaluate_intervention_selectivity(
    scm: StructuralCausalModel,
    factual_trajectory: Trajectory,
    counterfactual_trajectory: Trajectory,
    intervened_variable: str,
    monotonic_edge_signs: Mapping[tuple[str, str], str | MonotonicDirection] | None = None,
    tolerance: float = 1e-5,
) -> tuple[bool, list[InterventionSelectivityViolation], int]:
    """Evaluate descendant-selectivity and monotonicity sign consistency between factual and CF rollouts.

    Invariants:
        1. Descendant Selectivity (Vafa et al., 2024):
           For any variable Y not in descendants(X) and Y != X:
           |Y_cf(t) - Y_factual(t)| <= tolerance for all steps t.
           Non-descendants MUST NOT be perturbed by do(X = x).
        2. Monotonicity Direction (Vafa et al., 2025):
           For any descendant D with declared edge sign (X, D):
           sign(D_cf(t) - D_factual(t)) must align with declared monotonic direction.
    """
    violations: list[InterventionSelectivityViolation] = []
    checks = 0

    # Normalize variable key: if resource ID passed, map to 'resource:<id>'
    var_key = intervened_variable
    if not var_key.startswith("resource:") and var_key in [r.split(":", 1)[-1] for r in scm.endogenous_variables]:
        var_key = f"resource:{intervened_variable}"

    # Extract descendants from causal graph
    descendants = scm.causal_graph.descendants(var_key)

    # Collect all step states
    f_states = [factual_trajectory.initial_state] + [
        step.transition_result.next_state for step in factual_trajectory.steps
    ]
    cf_states = [counterfactual_trajectory.initial_state] + [
        step.transition_result.next_state for step in counterfactual_trajectory.steps
    ]

    min_steps = min(len(f_states), len(cf_states))
    edge_signs = dict(monotonic_edge_signs or {})

    # Determine initial intervention delta
    raw_var = var_key.split(":", 1)[-1] if ":" in var_key else var_key
    f_val0 = f_states[0].resources[raw_var].current if raw_var in f_states[0].resources else 0.0
    cf_val0 = cf_states[0].resources[raw_var].current if raw_var in cf_states[0].resources else 0.0
    intervention_delta = float(cf_val0 - f_val0)

    # Evaluate all resources across all steps
    all_resource_ids = sorted(
        set(f_states[0].resources.keys()).union(cf_states[0].resources.keys())
    )

    for step_idx in range(min_steps):
        s_fact = f_states[step_idx]
        s_cf = cf_states[step_idx]

        for rid in all_resource_ids:
            checks += 1
            node_name = f"resource:{rid}"

            v_fact = float(s_fact.resources[rid].current) if rid in s_fact.resources else 0.0
            v_cf = float(s_cf.resources[rid].current) if rid in s_cf.resources else 0.0
            delta_v = v_cf - v_fact

            is_descendant = (node_name in descendants)
            is_target = (node_name == var_key or rid == raw_var)

            # Invariant 1: Non-descendants must remain invariant
            if not is_descendant and not is_target:
                if abs(delta_v) > tolerance:
                    violations.append(
                        InterventionSelectivityViolation(
                            intervened_variable=var_key,
                            affected_variable=node_name,
                            step=step_idx,
                            violation_type="non_descendant_divergence",
                            observed_delta=delta_v,
                            expected_behavior="Non-descendant variables must remain invariant under do(X = x).",
                            message=(
                                f"Variable '{node_name}' is not a descendant of '{var_key}', but diverged "
                                f"by delta={delta_v:+.4f} (factual={v_fact:.4f}, cf={v_cf:.4f}) at step {step_idx}."
                            ),
                        )
                    )

            # Invariant 2: Monotonic direction check on declared edges
            if is_descendant:
                edge_key = (var_key, node_name)
                declared_sign = edge_signs.get(edge_key)
                if declared_sign is not None and abs(intervention_delta) > tolerance and abs(delta_v) > tolerance:
                    sign_str = str(declared_sign).lower()
                    if sign_str == "positive" and (intervention_delta * delta_v) < 0:
                        violations.append(
                            InterventionSelectivityViolation(
                                intervened_variable=var_key,
                                affected_variable=node_name,
                                step=step_idx,
                                violation_type="monotonicity_sign_inversion",
                                observed_delta=delta_v,
                                expected_behavior=f"Positive monotonic edge ({var_key} -> {node_name}) requires delta >= 0.",
                                message=(
                                    f"Monotonic sign inversion: intervention delta was {intervention_delta:+.4f}, "
                                    f"but affected descendant '{node_name}' diverged in the opposite direction "
                                    f"(delta={delta_v:+.4f}) at step {step_idx}."
                                ),
                            )
                        )
                    elif sign_str == "negative" and (intervention_delta * delta_v) > 0:
                        violations.append(
                            InterventionSelectivityViolation(
                                intervened_variable=var_key,
                                affected_variable=node_name,
                                step=step_idx,
                                violation_type="monotonicity_sign_inversion",
                                observed_delta=delta_v,
                                expected_behavior=f"Negative monotonic edge ({var_key} -> {node_name}) requires delta <= 0.",
                                message=(
                                    f"Monotonic sign inversion: intervention delta was {intervention_delta:+.4f}, "
                                    f"but affected descendant '{node_name}' increased (delta={delta_v:+.4f}) at step {step_idx}."
                                ),
                            )
                        )

    is_correct = len(violations) == 0
    return is_correct, violations, checks


class CausalCorrectnessGate:
    """Executes a suite of causal intervention-selectivity and Twin-Rollout consistency gates."""

    def __init__(
        self,
        tolerance: float = 1e-5,
        monotonic_edge_signs: Mapping[tuple[str, str], str | MonotonicDirection] | None = None,
    ) -> None:
        self.tolerance = tolerance
        self.monotonic_edge_signs = dict(monotonic_edge_signs or {})

    def evaluate_world(
        self,
        world: World,
        interventions: Sequence[dict[str, Any]],
        horizon: int = 5,
        seed: int = 42,
    ) -> CausalCorrectnessReport:
        """Run factual rollouts and coupled counterfactual interventions, evaluating causal correctness."""
        from ewm_engine.core.actions import Intervention
        from ewm_engine.simulation.counterfactual import CounterfactualEngine
        from ewm_engine.simulation.engine import SimulationEngine
        from ewm_engine.simulation.scenario import Scenario

        sim_engine = SimulationEngine()
        cf_engine = CounterfactualEngine(locality_tolerance=self.tolerance)

        # 1. Run factual trajectory
        scen = Scenario(scenario_id="causal_eval_base", horizon=horizon, samples=1, seed=seed)
        res = sim_engine.run(world, scen)
        factual_traj = res.trajectories[0]

        scm = world.as_scm(horizon=horizon)

        all_violations: list[InterventionSelectivityViolation] = []
        total_checks = 0
        tested_interventions: list[str] = []

        for item in interventions:
            var_name = str(item.get("target_variable", ""))
            new_val = float(item.get("value", 0.0))
            if not var_name:
                continue

            tested_interventions.append(var_name)
            raw_var = var_name.split(":", 1)[-1] if ":" in var_name else var_name

            cf_action = Intervention(
                id=f"gate_do_{raw_var}",
                description=f"Evaluate do({raw_var} = {new_val})",
                parameters={raw_var: new_val},
            )

            cf_res = cf_engine.run(
                world=world,
                observed=factual_traj,
                intervention=cf_action,
                horizon=horizon,
                seed=seed,
            )

            _, violations, n_checks = evaluate_intervention_selectivity(
                scm=scm,
                factual_trajectory=factual_traj,
                counterfactual_trajectory=cf_res.counterfactual_trajectory,
                intervened_variable=var_name,
                monotonic_edge_signs=self.monotonic_edge_signs,
                tolerance=self.tolerance,
            )

            all_violations.extend(violations)
            total_checks += n_checks

        is_passed = len(all_violations) == 0
        pass_rate = float((total_checks - len(all_violations)) / total_checks) if total_checks > 0 else 1.0

        return CausalCorrectnessReport(
            is_causally_correct=is_passed,
            pass_rate=pass_rate,
            total_checks=total_checks,
            violations_count=len(all_violations),
            violations=tuple(all_violations),
            evaluated_interventions=tuple(tested_interventions),
        )
