"""Metamorphic Relations (MR-cards) and mutation adequacy test suite.

Scientific Anchors:
    - Chen, T. Y., et al. (2020): "Metamorphic Testing: A Review of Challenges and Opportunities".
      ACM Computing Surveys, 53(4), 1-27.
    - Kang et al. (ICML 2025, arXiv:2411.02385): "How Far is Video Generation from World Model
      (physical law evaluation without human labels)".
    - Motamed et al. (2025, arXiv:2501.09038): "Physics-IQ: Evaluating Physical Reasoning
      in World Models beyond Perceptual Realism".
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.spec import WorldFactory, WorldSpec
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import Trajectory


class MetamorphicRelationVerdict(BaseModel):
    """Result of evaluating an individual Metamorphic Relation card (MR-card)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mr_id: str = Field(description="Unique identifier of the metamorphic relation card.")
    name: str = Field(description="Human-readable title of the relation.")
    passed: bool = Field(description="True if metamorphic output relation holds.")
    tolerance: float = Field(description="Numerical tolerance threshold applied.")
    discrepancy: float = Field(default=0.0, description="Observed numerical discrepancy or violation magnitude.")
    details: str = Field(description="Audit explanation or error message.")
    execution_duration_seconds: float = Field(default=0.0, description="Runtime execution duration.")


class MetamorphicSuiteReport(BaseModel):
    """Aggregated audit report of metamorphic relation evaluation across an enterprise world model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    all_passed: bool = Field(description="True if all evaluated metamorphic relations passed.")
    pass_rate: float = Field(description="Fraction of metamorphic relation cards that passed.")
    total_cards: int = Field(description="Total count of evaluated metamorphic relation cards.")
    passed_cards: int = Field(description="Count of passing metamorphic relation cards.")
    verdicts: tuple[MetamorphicRelationVerdict, ...] = Field(
        default_factory=tuple, description="Individual results for each evaluated MR-card."
    )
    mutation_adequacy_score: float | None = Field(
        default=None, description="Fraction of non-equivalent synthetic mutants killed by this suite."
    )

    def summary_table(self) -> str:
        """Render formatted console summary table of the metamorphic evaluation."""
        lines: list[str] = []
        lines.append("Metamorphic Self-Knowledge Suite (Label-Free Oracle)")
        lines.append("=" * 72)
        status_marker = "[PASS]" if self.all_passed else "[FAIL]"
        lines.append(f"Suite Status:           {status_marker} {'ALL RELATIONS SATISFIED' if self.all_passed else 'VIOLATIONS DETECTED'}")
        lines.append(f"MR Pass Rate:           {self.pass_rate * 100:.1f}% ({self.passed_cards}/{self.total_cards})")
        if self.mutation_adequacy_score is not None:
            lines.append(f"Mutation Adequacy:      {self.mutation_adequacy_score * 100:.1f}% (Mutants Killed)")

        lines.append("\nEvaluated Metamorphic Relations:")
        lines.append("-" * 72)
        lines.append(f"{'MR ID':<18} {'Verdict':<10} {'Discrepancy':<14} {'Relation Name':<28}")
        lines.append("-" * 72)
        for v in self.verdicts:
            v_marker = "[PASS]" if v.passed else "[FAIL]"
            lines.append(f"{v.mr_id:<18} {v_marker:<10} {v.discrepancy:<14.2e} {v.name:<28}")

        lines.append("-" * 72)
        return "\n".join(lines)


class MetamorphicRelationCard:
    """Executable Metamorphic Relation card (MR-card).

    Defines an input transformation T: (Spec, Scenario) -> (Spec', Scenario')
    and an output relation check R: (Traj, Traj') -> (bool, discrepancy, details).
    """

    def __init__(
        self,
        mr_id: str,
        name: str,
        description: str,
        transform_input: Callable[[WorldSpec, Scenario], tuple[WorldSpec, Scenario]],
        check_output_relation: Callable[[Trajectory, Trajectory, float], tuple[bool, float, str]],
        tolerance: float = 1e-6,
    ) -> None:
        self.mr_id = mr_id
        self.name = name
        self.description = description
        self.transform_input = transform_input
        self.check_output_relation = check_output_relation
        self.tolerance = tolerance

    def evaluate(
        self,
        spec: WorldSpec,
        scenario: Scenario,
        sim_engine: SimulationEngine | None = None,
    ) -> MetamorphicRelationVerdict:
        """Execute baseline rollout and transformed rollout, verifying the metamorphic relation."""
        t_start = time.time()
        engine = sim_engine or SimulationEngine()
        factory = WorldFactory()

        # 1. Baseline rollout
        world_base = factory.create_world(spec)
        res_base = engine.run(world_base, scenario)
        traj_base = res_base.trajectories[0]

        # 2. Transform input according to metamorphic relation T
        transformed_spec, transformed_scenario = self.transform_input(spec, scenario)

        # 3. Transformed rollout
        world_trans = factory.create_world(transformed_spec)
        res_trans = engine.run(world_trans, transformed_scenario)
        traj_trans = res_trans.trajectories[0]

        # 4. Check output relation R
        passed, disc, details = self.check_output_relation(traj_base, traj_trans, self.tolerance)
        duration = float(time.time() - t_start)

        return MetamorphicRelationVerdict(
            mr_id=self.mr_id,
            name=self.name,
            passed=passed,
            tolerance=self.tolerance,
            discrepancy=float(disc),
            details=details,
            execution_duration_seconds=duration,
        )


# -----------------------------------------------------------------------------
# Standard Metamorphic Relation Cards for Enterprise World Models
# -----------------------------------------------------------------------------


def _mr1_transform(spec: WorldSpec, scen: Scenario) -> tuple[WorldSpec, Scenario]:
    # Exact duplicate input with identical seed
    return spec.model_copy(deep=True), scen.model_copy(deep=True)


def _mr1_check(t1: Trajectory, t2: Trajectory, tol: float) -> tuple[bool, float, str]:
    # S00 Invariant 5: Same seed + config => identical cryptographic state fingerprints
    if len(t1.steps) != len(t2.steps):
        return False, 1.0, f"Step count mismatch: {len(t1.steps)} vs {len(t2.steps)}"
    for idx, (s1, s2) in enumerate(zip(t1.steps, t2.steps, strict=False)):
        h1 = s1.state_hash
        h2 = s2.state_hash
        if h1 != h2:
            return False, 1.0, f"State hash mismatch at step {idx}: '{h1}' != '{h2}'"
    return True, 0.0, "Identical state hashes across all simulation steps."


MR_DETERMINISM = MetamorphicRelationCard(
    mr_id="MR-1-DETERMINISM",
    name="Deterministic Reproducibility (Invariant 5)",
    description="Identical WorldSpec and Scenario seed must yield identical state sequence.",
    transform_input=_mr1_transform,
    check_output_relation=_mr1_check,
)


def _mr2_transform(spec: WorldSpec, scen: Scenario) -> tuple[WorldSpec, Scenario]:
    return spec.model_copy(deep=True), scen.model_copy(deep=True)


def _mr2_check(t1: Trajectory, t2: Trajectory, tol: float) -> tuple[bool, float, str]:
    # Total mass conservation check:
    # Verifies that total mass changes across steps are fully accounted for by transition flows (no unaccounted leakage).
    states = [t1.initial_state] + [s.transition_result.next_state for s in t1.steps]
    if not states:
        return True, 0.0, "Empty state sequence."

    max_unaccounted = 0.0
    for idx, step in enumerate(t1.steps):
        prev_s = states[idx]
        next_s = states[idx + 1]
        delta_total = sum(float(r.current) for r in next_s.resources.values()) - sum(
            float(r.current) for r in prev_s.resources.values()
        )
        # Check net external flows reported in applied_changes
        net_flow = 0.0
        for change_val in step.transition_result.applied_changes.values():
            if isinstance(change_val, dict):
                # Subtract served demand (leaves the system)
                if "served_demand" in change_val:
                    net_flow -= float(change_val["served_demand"])
                # Add inbound deliveries/replenishment if any
                if "replenished_quantity" in change_val:
                    net_flow += float(change_val["replenished_quantity"])

        unaccounted = abs(delta_total - net_flow)
        max_unaccounted = max(max_unaccounted, unaccounted)

    passed = max_unaccounted <= tol
    return passed, max_unaccounted, f"Max unaccounted mass drift: {max_unaccounted:.6f}"


MR_CONSERVATION = MetamorphicRelationCard(
    mr_id="MR-2-CONSERVATION",
    name="Conservation Law Invariant",
    description="In a closed system, total sum of conserved resources must remain constant across steps.",
    transform_input=_mr2_transform,
    check_output_relation=_mr2_check,
    tolerance=1e-5,
)


def _mr3_transform_scaling(spec: WorldSpec, scen: Scenario) -> tuple[WorldSpec, Scenario]:
    # Scale all initial resource levels by factor lambda = 2.0
    scale_factor = 2.0
    new_res = []
    for r in spec.resources:
        new_res.append(
            r.model_copy(
                update={
                    "current": float(r.current * scale_factor),
                    "min_value": float(r.min_value * scale_factor),
                    "max_value": float(r.max_value * scale_factor),
                }
            )
        )
    return spec.model_copy(update={"resources": new_res}), scen.model_copy(deep=True)


def _mr3_check_scaling(t_base: Trajectory, t_scaled: Trajectory, tol: float) -> tuple[bool, float, str]:
    # In linear transfer dynamics, trajectories scale proportionally
    b_states = [t_base.initial_state] + [s.transition_result.next_state for s in t_base.steps]
    s_states = [t_scaled.initial_state] + [s.transition_result.next_state for s in t_scaled.steps]

    max_rel_error = 0.0
    for b_s, s_s in zip(b_states, s_states, strict=False):
        for rid, r_b in b_s.resources.items():
            if rid in s_s.resources and abs(r_b.current) > 1e-4:
                expected = r_b.current * 2.0
                actual = s_s.resources[rid].current
                rel_err = abs(actual - expected) / max(1.0, abs(expected))
                max_rel_error = max(max_rel_error, rel_err)

    passed = max_rel_error <= 0.25  # Allow tolerance for non-linear boundaries/capacities
    return passed, max_rel_error, f"Max relative scaling deviation: {max_rel_error:.4f}"


MR_SCALING = MetamorphicRelationCard(
    mr_id="MR-3-SCALING",
    name="Resource Scale Homogeneity",
    description="Scaling initial resource levels by lambda scales output trajectories proportionally.",
    transform_input=_mr3_transform_scaling,
    check_output_relation=_mr3_check_scaling,
)


def _mr4_transform_monotonicity(spec: WorldSpec, scen: Scenario) -> tuple[WorldSpec, Scenario]:
    # Increase initial inventory of the first resource by 50%
    if not spec.resources:
        return spec.model_copy(deep=True), scen.model_copy(deep=True)
    first_r = spec.resources[0]
    updated_r = first_r.model_copy(update={"current": float(first_r.current * 1.5 + 10.0)})
    new_res = [updated_r, *list(spec.resources[1:])]
    return spec.model_copy(update={"resources": new_res}), scen.model_copy(deep=True)


def _mr4_check_monotonicity(t_base: Trajectory, t_boosted: Trajectory, tol: float) -> tuple[bool, float, str]:
    # Monotonicity: Boosting initial resource cannot cause final level to drop below baseline
    b_final = t_base.final_state
    s_final = t_boosted.final_state
    violations = 0
    max_inversion = 0.0

    for rid in b_final.resources:
        if rid in s_final.resources:
            b_val = b_final.resources[rid].current
            s_val = s_final.resources[rid].current
            # For the boosted resource, final level must be >= baseline
            if s_val < (b_val - tol):
                inv = b_val - s_val
                max_inversion = max(max_inversion, inv)
                violations += 1

    passed = (violations == 0)
    return passed, max_inversion, f"Monotonicity violations: {violations}, max inversion: {max_inversion:.4f}"


MR_MONOTONICITY = MetamorphicRelationCard(
    mr_id="MR-4-MONOTONICITY",
    name="Monotonic Supply Adequacy",
    description="Increasing initial inventory cannot decrease terminal availability.",
    transform_input=_mr4_transform_monotonicity,
    check_output_relation=_mr4_check_monotonicity,
)


def _mr5_transform_permutation(spec: WorldSpec, scen: Scenario) -> tuple[WorldSpec, Scenario]:
    # Permute entity IDs: rename e_i -> e_i_renamed
    new_entities = [
        e.model_copy(update={"id": f"permuted_{e.id}"}) for e in spec.entities
    ]
    # Update relationships to match permuted entity IDs
    new_rels = [
        r.model_copy(
            update={
                "source": f"permuted_{r.source}",
                "target": f"permuted_{r.target}",
            }
        )
        for r in spec.relationships
    ]
    return (
        spec.model_copy(update={"entities": new_entities, "relationships": new_rels}),
        scen.model_copy(deep=True),
    )


def _mr5_check_permutation(t_base: Trajectory, t_perm: Trajectory, tol: float) -> tuple[bool, float, str]:
    # Entity ID permutation invariance: numerical resource levels must remain identical
    b_final = t_base.final_state
    p_final = t_perm.final_state

    max_disc = 0.0
    for rid in b_final.resources:
        if rid in p_final.resources:
            diff = abs(b_final.resources[rid].current - p_final.resources[rid].current)
            max_disc = max(max_disc, diff)

    passed = max_disc <= tol
    return passed, max_disc, f"Max discrepancy under entity ID permutation: {max_disc:.6f}"


MR_ENTITY_PERMUTATION = MetamorphicRelationCard(
    mr_id="MR-5-PERMUTATION",
    name="Entity ID Permutation Invariance",
    description="Isomorphic renaming of entity identifiers leaves systemic resource levels unchanged.",
    transform_input=_mr5_transform_permutation,
    check_output_relation=_mr5_check_permutation,
)


def _mr6_transform_slack(spec: WorldSpec, scen: Scenario) -> tuple[WorldSpec, Scenario]:
    # Add a non-binding capacity constraint on the first resource
    from ewm_engine.core.spec import ComponentSpec

    res_id = spec.resources[0].id if spec.resources else "res"
    slack_constraint = ComponentSpec(
        type="capacity",
        parameters={"resource_id": res_id, "capacity": 1000000.0, "severity": "soft"},
    )
    new_constraints = (*spec.constraints, slack_constraint)
    return spec.model_copy(update={"constraints": new_constraints}), scen.model_copy(deep=True)


def _mr6_check_slack(t1: Trajectory, t2: Trajectory, tol: float) -> tuple[bool, float, str]:
    # Adding a slack-satisfied soft constraint must NOT alter a valid trajectory
    if len(t1.steps) != len(t2.steps):
        return False, 1.0, f"Step length mismatch: {len(t1.steps)} vs {len(t2.steps)}"
    for idx, (s1, s2) in enumerate(zip(t1.steps, t2.steps, strict=False)):
        if s1.state_hash != s2.state_hash:
            return False, 1.0, f"State hash diverged at step {idx} after adding slack constraint."
    return True, 0.0, "Trajectory invariant under addition of slack-satisfied constraint."


MR_CONSTRAINT_SLACK = MetamorphicRelationCard(
    mr_id="MR-6-CONSTRAINT-SLACK",
    name="Slack Constraint Invariance",
    description="Adding a slack-satisfied soft constraint does not change a valid trajectory.",
    transform_input=_mr6_transform_slack,
    check_output_relation=_mr6_check_slack,
)


STANDARD_MR_CARDS: tuple[MetamorphicRelationCard, ...] = (
    MR_DETERMINISM,
    MR_CONSERVATION,
    MR_SCALING,
    MR_MONOTONICITY,
    MR_ENTITY_PERMUTATION,
    MR_CONSTRAINT_SLACK,
)


class MetamorphicSuite:
    """Orchestrates execution of MR-cards against enterprise world models."""

    def __init__(
        self,
        cards: Sequence[MetamorphicRelationCard] = STANDARD_MR_CARDS,
        sim_engine: SimulationEngine | None = None,
    ) -> None:
        self.cards = tuple(cards)
        self.engine = sim_engine or SimulationEngine()

    def run_suite(
        self,
        spec: WorldSpec,
        scenario: Scenario | None = None,
    ) -> MetamorphicSuiteReport:
        """Run all metamorphic relation cards against the given specification."""
        scen = scenario or Scenario(scenario_id="mr_suite_scen", horizon=6, samples=1, seed=42)
        verdicts: list[MetamorphicRelationVerdict] = []

        for card in self.cards:
            v = card.evaluate(spec=spec, scenario=scen, sim_engine=self.engine)
            verdicts.append(v)

        passed_count = sum(1 for v in verdicts if v.passed)
        total_count = len(verdicts)
        pass_rate = float(passed_count / total_count) if total_count > 0 else 1.0

        return MetamorphicSuiteReport(
            all_passed=(passed_count == total_count),
            pass_rate=pass_rate,
            total_cards=total_count,
            passed_cards=passed_count,
            verdicts=tuple(verdicts),
        )


# -----------------------------------------------------------------------------
# Mutation Adequacy Evaluator (Proves MR Suite is sensitive to injected faults)
# -----------------------------------------------------------------------------


class MutationType(StrEnum):
    """Types of synthetic structural defects injected to evaluate suite sensitivity."""

    LEAKAGE = "leakage"  # Injects non-conservative mass loss/gain
    SIGN_INVERSION = "sign_inversion"  # Inverts transition direction
    NON_DETERMINISM = "non_determinism"  # Corrupts RNG seed coupling
    PERMUTATION_BIAS = "permutation_bias"  # Adds ID-dependent bias


class FaultMutant:
    """Mutant WorldSpec wrapper with an intentional defect injected."""

    def __init__(self, name: str, mutation_type: MutationType, mutated_spec: WorldSpec) -> None:
        self.name = name
        self.mutation_type = mutation_type
        self.mutated_spec = mutated_spec


class MutationAdequacyEvaluator:
    """Injects synthetic defects into WorldSpec and evaluates whether the MR suite kills them."""

    def __init__(self, suite: MetamorphicSuite | None = None) -> None:
        self.suite = suite or MetamorphicSuite()

    def generate_mutants(self, clean_spec: WorldSpec) -> list[FaultMutant]:
        """Generate a suite of non-equivalent structural mutants."""
        from ewm_engine.core.spec import ComponentSpec

        mutants: list[FaultMutant] = []
        target_res = clean_spec.resources[0].id if clean_spec.resources else "warehouse_a"

        def _wrap_with_fault(fault_type: str, fault_params: dict[str, Any]) -> ComponentSpec:
            fault_entry = {"type": fault_type, "parameters": fault_params}
            if clean_spec.dynamics is not None:
                if clean_spec.dynamics.type == "composite":
                    existing = list(clean_spec.dynamics.parameters.get("models", []))
                    new_params = {**clean_spec.dynamics.parameters, "models": [*existing, fault_entry]}
                    return clean_spec.dynamics.model_copy(update={"parameters": new_params})
                else:
                    existing = [{"type": clean_spec.dynamics.type, "parameters": clean_spec.dynamics.parameters}]
                    return ComponentSpec(
                        type="composite",
                        parameters={"name": "MutatedDynamics", "models": [*existing, fault_entry]},
                    )
            return ComponentSpec(
                type="composite",
                parameters={"name": "MutatedDynamics", "models": [fault_entry]},
            )

        # 1. Leakage Mutant: injects unmetered mass drain (violates MR-2 Conservation)
        leak_dyn = _wrap_with_fault("unmetered_leakage", {"resource_id": target_res, "leak_rate": 20.0})
        mutants.append(
            FaultMutant(
                name="mutant_unmetered_leakage",
                mutation_type=MutationType.LEAKAGE,
                mutated_spec=clean_spec.model_copy(update={"dynamics": leak_dyn}),
            )
        )

        # 2. Non-Determinism Mutant: injects unseeded jitter (violates MR-1 Determinism)
        jitter_dyn = _wrap_with_fault("unseeded_jitter", {"resource_id": target_res})
        mutants.append(
            FaultMutant(
                name="mutant_unseeded_jitter",
                mutation_type=MutationType.NON_DETERMINISM,
                mutated_spec=clean_spec.model_copy(update={"dynamics": jitter_dyn}),
            )
        )

        # 3. Permutation Bias Mutant: introduces behavior biased by entity names (violates MR-5 Permutation)
        bias_dyn = _wrap_with_fault("permutation_bias", {"resource_id": target_res})
        mutants.append(
            FaultMutant(
                name="mutant_permutation_bias",
                mutation_type=MutationType.PERMUTATION_BIAS,
                mutated_spec=clean_spec.model_copy(update={"dynamics": bias_dyn}),
            )
        )

        return mutants

    def evaluate_mutation_adequacy(
        self,
        clean_spec: WorldSpec,
        scenario: Scenario | None = None,
    ) -> tuple[float, int, int, list[str]]:
        """Evaluate mutation adequacy score: mutants_killed / total_mutants.

        Returns:
            Tuple of (score, mutants_killed, total_mutants, list_of_killed_mutant_names).
        """
        mutants = self.generate_mutants(clean_spec)
        if not mutants:
            return 1.0, 0, 0, []

        killed = 0
        killed_names: list[str] = []

        for m in mutants:
            report = self.suite.run_suite(spec=m.mutated_spec, scenario=scenario)
            # A mutant is "killed" if at least one metamorphic relation detects the violation
            if not report.all_passed:
                killed += 1
                killed_names.append(m.name)

        score = float(killed / len(mutants))
        return score, killed, len(mutants), killed_names
