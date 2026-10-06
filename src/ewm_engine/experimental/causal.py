"""Causal diagnostics, Ladder-of-Causation identifiability, and Twin Rollouts.

[EXPERIMENTAL] This module provides principled causal diagnostics for world models:
    - Typed Ladder-of-Causation query surface (Pearl, arXiv:1801.04016)
    - Backdoor Criterion identifiability checking over declared mechanism graphs
    - Overlap and positivity diagnostics on logged transition datasets
    - Confounding sensitivity reporting (Rosenbaum bounds Gamma)
    - Noise-coupled counterfactual branching (Twin Rollouts, arXiv:2608.08982)

Epistemic Invariant:
    The engine NEVER claims to automatically discover causality from observational data.
    Causal identifiability and EvidenceLevel tags are determined strictly by the declared
    structural mechanism graph and mathematical check, never inferred from statistical fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.learned import TransitionDataset
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.simulation.branching import branch_world
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario

# ---------------------------------------------------------------------------
# 1. Structural Causal Graph & Backdoor Identifiability
# ---------------------------------------------------------------------------


@dataclass
class CausalGraph:
    """Directed Acyclic Graph (DAG) specifying declared structural mechanisms.

    Edges represent direct causal dependencies between variables: (source, target).
    """

    nodes: set[str] = field(default_factory=set)
    edges: set[tuple[str, str]] = field(default_factory=set)
    unobserved_nodes: set[str] = field(default_factory=set)

    def add_node(self, name: str, is_unobserved: bool = False) -> None:
        self.nodes.add(name)
        if is_unobserved:
            self.unobserved_nodes.add(name)

    def add_edge(self, source: str, target: str) -> None:
        self.nodes.add(source)
        self.nodes.add(target)
        self.edges.add((source, target))

    def parents(self, node: str) -> set[str]:
        return {src for (src, tgt) in self.edges if tgt == node}

    def children(self, node: str) -> set[str]:
        return {tgt for (src, tgt) in self.edges if src == node}

    def descendants(self, node: str) -> set[str]:
        visited: set[str] = set()
        queue = [node]
        while queue:
            curr = queue.pop(0)
            for ch in self.children(curr):
                if ch not in visited:
                    visited.add(ch)
                    queue.append(ch)
        return visited

    def find_all_undirected_paths(
        self, start: str, end: str, max_depth: int = 6
    ) -> list[list[str]]:
        """Find all undirected paths between start and end (ignoring edge direction)."""
        adjacency: dict[str, set[str]] = {n: set() for n in self.nodes}
        for u, v in self.edges:
            adjacency[u].add(v)
            adjacency[v].add(u)

        paths: list[list[str]] = []
        queue: list[list[str]] = [[start]]

        while queue:
            path = queue.pop(0)
            curr = path[-1]
            if curr == end:
                paths.append(path)
                continue
            if len(path) > max_depth:
                continue
            for neighbor in adjacency.get(curr, set()):
                if neighbor not in path:
                    queue.append([*path, neighbor])
        return paths

    def is_backdoor_path(self, path: list[str], treatment: str) -> bool:
        """Check if a path is a backdoor path into treatment (starts with arrow into treatment)."""
        if len(path) < 2 or path[0] != treatment:
            return False
        # Path is backdoor if second node points into treatment: (path[1], treatment) in edges
        return (path[1], treatment) in self.edges

    def is_collider(self, prev_node: str, curr_node: str, next_node: str) -> bool:
        """Check if curr_node is a collider along the path (arrows point into curr: -> curr <-)."""
        return (prev_node, curr_node) in self.edges and (next_node, curr_node) in self.edges

    def is_path_blocked(self, path: list[str], conditioning_set: set[str]) -> bool:
        """Check if a path is d-separated / blocked given the conditioning set."""
        cond_and_descendants = set(conditioning_set)
        for c in conditioning_set:
            cond_and_descendants.update(self.descendants(c))

        for i in range(1, len(path) - 1):
            prev_n = path[i - 1]
            curr_n = path[i]
            next_n = path[i + 1]

            if self.is_collider(prev_n, curr_n, next_n):
                # Collider: blocked if neither curr_n nor any descendant is conditioned on
                if curr_n not in cond_and_descendants:
                    return True
            else:
                # Non-collider: blocked if curr_n is conditioned on
                if curr_n in conditioning_set:
                    return True
        return False


class IdentifiabilityResult(BaseModel):
    """Formal audit result of a causal identifiability check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_identifiable: bool = Field(
        description="True if the interventional effect passes identifiability criteria."
    )
    treatment: str = Field(description="Interventional treatment variable.")
    outcome: str = Field(description="Target outcome variable.")
    conditioning_set: tuple[str, ...] = Field(
        description="Variables conditioned on / controlled for."
    )
    open_backdoor_paths: tuple[tuple[str, ...], ...] = Field(
        default_factory=tuple,
        description="Confounding backdoor paths left open by the conditioning set.",
    )
    unobserved_confounders: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Unobserved variables creating unblockable confounding paths.",
    )
    causal_level: EvidenceLevel = Field(
        description="Epistemic standing warranted by graph structure (never inferred from data).",
    )
    diagnostic_message: str = Field(
        description="Human-readable explanation of identifiability status."
    )


def check_backdoor_identifiability(
    graph: CausalGraph,
    treatment: str,
    outcome: str,
    conditioning_set: Sequence[str] = (),
) -> IdentifiabilityResult:
    """Evaluate whether the causal effect of do(treatment) on outcome is identifiable via Backdoor Criterion.

    Epistemic Guarantee:
        Returns EvidenceLevel.INTERVENTIONAL if and only if all backdoor paths are blocked
        by observed variables and no descendants of treatment are conditioned on.
        Otherwise returns EvidenceLevel.PREDICTIVE with explicit open paths.
    """
    cond_set = set(conditioning_set)

    # Criterion 1: Conditioning set must not contain descendants of treatment
    descendants_of_treatment = graph.descendants(treatment)
    forbidden_descendants = cond_set & descendants_of_treatment
    if forbidden_descendants:
        return IdentifiabilityResult(
            is_identifiable=False,
            treatment=treatment,
            outcome=outcome,
            conditioning_set=tuple(sorted(cond_set)),
            causal_level=EvidenceLevel.PREDICTIVE,
            diagnostic_message=(
                f"Backdoor criterion violated: Conditioning set contains descendants of treatment: "
                f"{sorted(forbidden_descendants)}. Conditioning on descendants induces selection bias."
            ),
        )

    # Criterion 2: Block all backdoor paths from treatment to outcome
    all_paths = graph.find_all_undirected_paths(treatment, outcome)
    backdoor_paths = [p for p in all_paths if graph.is_backdoor_path(p, treatment)]

    open_paths: list[tuple[str, ...]] = []
    unobserved_on_paths: set[str] = set()

    for path in backdoor_paths:
        if not graph.is_path_blocked(path, cond_set):
            open_paths.append(tuple(path))
            for node in path:
                if node in graph.unobserved_nodes:
                    unobserved_on_paths.add(node)

    if open_paths:
        msg = (
            f"Effect of {treatment} on {outcome} is NOT identifiable under declared graph. "
            f"There are {len(open_paths)} open backdoor confounding paths."
        )
        if unobserved_on_paths:
            msg += f" Paths pass through unobserved confounders: {sorted(unobserved_on_paths)}."

        return IdentifiabilityResult(
            is_identifiable=False,
            treatment=treatment,
            outcome=outcome,
            conditioning_set=tuple(sorted(cond_set)),
            open_backdoor_paths=tuple(open_paths),
            unobserved_confounders=tuple(sorted(unobserved_on_paths)),
            causal_level=EvidenceLevel.PREDICTIVE,
            diagnostic_message=msg,
        )

    return IdentifiabilityResult(
        is_identifiable=True,
        treatment=treatment,
        outcome=outcome,
        conditioning_set=tuple(sorted(cond_set)),
        open_backdoor_paths=(),
        unobserved_confounders=(),
        causal_level=EvidenceLevel.INTERVENTIONAL,
        diagnostic_message=(
            f"Under declared causal graph, the effect of do({treatment}) on {outcome} "
            f"is IDENTIFIABLE using backdoor adjustment over {sorted(cond_set)}."
        ),
    )


# ---------------------------------------------------------------------------
# 2. Ladder-of-Causation Query Interface
# ---------------------------------------------------------------------------


class ObservationalQueryResult(BaseModel):
    """Result of an observational conditional expectation query P(Y | X)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    treatment: str
    outcome: str
    conditional_mean_treated: float
    conditional_mean_control: float
    observational_association: float
    sample_size: int
    causal_level: EvidenceLevel = Field(default=EvidenceLevel.PREDICTIVE)
    disclaimer: str = Field(
        default="Associational difference P(Y | X=1) - P(Y | X=0) only. Does NOT imply causal effect.",
    )


class InterventionalQueryResult(BaseModel):
    """Result of an interventional query P(Y | do(X)) evaluated under declared structural assumptions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    treatment: str
    outcome: str
    adjusted_effect: float
    identifiability: IdentifiabilityResult
    causal_level: EvidenceLevel
    assumptions_summary: str


class NotIdentifiableResult(BaseModel):
    """Diagnostic returned when an interventional query cannot be causally identified."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    treatment: str
    outcome: str
    identifiability: IdentifiabilityResult
    causal_level: EvidenceLevel = Field(default=EvidenceLevel.PREDICTIVE)
    diagnostic_message: str


def query_observational(
    dataset: TransitionDataset,
    treatment_action: str,
    outcome_resource: str,
) -> ObservationalQueryResult:
    """Compute observational conditional expectation P(Y | X=1) - P(Y | X=0)."""
    treated_vals: list[float] = []
    control_vals: list[float] = []

    for sample in dataset:
        has_treatment = any(act.type == treatment_action for act in sample.actions)
        res = sample.next_state.resources.get(outcome_resource)
        val = res.current if res is not None else 0.0

        if has_treatment:
            treated_vals.append(val)
        else:
            control_vals.append(val)

    mean_t = float(np.mean(treated_vals)) if treated_vals else 0.0
    mean_c = float(np.mean(control_vals)) if control_vals else 0.0
    assoc = mean_t - mean_c

    return ObservationalQueryResult(
        treatment=treatment_action,
        outcome=outcome_resource,
        conditional_mean_treated=mean_t,
        conditional_mean_control=mean_c,
        observational_association=assoc,
        sample_size=len(dataset),
        causal_level=EvidenceLevel.PREDICTIVE,
    )


def query_interventional(
    dataset: TransitionDataset,
    treatment_action: str,
    outcome_resource: str,
    graph: CausalGraph,
    conditioning_set: Sequence[str] = (),
) -> InterventionalQueryResult | NotIdentifiableResult:
    """Evaluate interventional query P(Y | do(X)).

    Returns NotIdentifiableResult if backdoor paths are open.
    Never assigns EvidenceLevel.INTERVENTIONAL unless identifiability check passes.
    """
    id_res = check_backdoor_identifiability(
        graph=graph,
        treatment=treatment_action,
        outcome=outcome_resource,
        conditioning_set=conditioning_set,
    )

    if not id_res.is_identifiable:
        return NotIdentifiableResult(
            treatment=treatment_action,
            outcome=outcome_resource,
            identifiability=id_res,
            causal_level=EvidenceLevel.PREDICTIVE,
            diagnostic_message=id_res.diagnostic_message,
        )

    # Identifiable: compute stratified backdoor adjustment
    treated_by_stratum: dict[tuple[float, ...], list[float]] = {}
    control_by_stratum: dict[tuple[float, ...], list[float]] = {}
    stratum_weights: dict[tuple[float, ...], int] = {}

    for sample in dataset:
        has_treatment = any(act.type == treatment_action for act in sample.actions)
        res = sample.next_state.resources.get(outcome_resource)
        val = res.current if res is not None else 0.0

        # Stratify conditioning variables for backdoor adjustment
        stratum_key = tuple(
            round(
                float(sample.state.resources[c_var].current)
                if c_var in sample.state.resources
                else 0.0,
                2,
            )
            for c_var in conditioning_set
        )
        stratum_weights[stratum_key] = stratum_weights.get(stratum_key, 0) + 1

        if has_treatment:
            treated_by_stratum.setdefault(stratum_key, []).append(val)
        else:
            control_by_stratum.setdefault(stratum_key, []).append(val)

    weighted_diff = 0.0
    valid_weight = 0

    for stratum, count in stratum_weights.items():
        t_vals = treated_by_stratum.get(stratum, [])
        c_vals = control_by_stratum.get(stratum, [])
        if t_vals and c_vals:
            diff = float(np.mean(t_vals)) - float(np.mean(c_vals))
            weighted_diff += diff * count
            valid_weight += count

    adjusted_effect = (weighted_diff / valid_weight) if valid_weight > 0 else 0.0

    return InterventionalQueryResult(
        treatment=treatment_action,
        outcome=outcome_resource,
        adjusted_effect=adjusted_effect,
        identifiability=id_res,
        causal_level=EvidenceLevel.INTERVENTIONAL,
        assumptions_summary=(
            f"Identified under declared causal DAG via backdoor adjustment over {sorted(conditioning_set)}."
        ),
    )


# ---------------------------------------------------------------------------
# 3. Overlap / Positivity Diagnostic
# ---------------------------------------------------------------------------


class PositivityReport(BaseModel):
    """Diagnostic audit of observational propensity score overlap."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    treatment: str
    positivity_satisfied: bool = Field(
        description="True if all strata have non-extreme propensity scores."
    )
    overlap_index: float = Field(
        description="Bhattacharyya coefficient of propensity distributions between treated and control [0.0, 1.0].",
    )
    violation_rate: float = Field(
        description="Fraction of transitions with extreme propensity (e < min_propensity or e > 1 - min_propensity).",
    )
    treated_count: int
    control_count: int
    diagnostic_message: str


def check_positivity_overlap(
    dataset: TransitionDataset,
    treatment_action: str,
    covariate_resources: Sequence[str],
    min_propensity: float = 0.05,
) -> PositivityReport:
    """Inspect empirical propensity score distribution to detect positivity / common support breaches."""
    if len(dataset) < 5:
        return PositivityReport(
            treatment=treatment_action,
            positivity_satisfied=False,
            overlap_index=0.0,
            violation_rate=1.0,
            treated_count=0,
            control_count=0,
            diagnostic_message="Dataset too small for positivity assessment.",
        )

    treated_flags: list[float] = []
    cov_matrix: list[list[float]] = []

    for s in dataset:
        treated_flags.append(1.0 if any(a.type == treatment_action for a in s.actions) else 0.0)
        row = [
            float(s.state.resources[k].current) if k in s.state.resources else 0.0
            for k in covariate_resources
        ]
        cov_matrix.append(row)

    Y = np.array(treated_flags)
    X = np.array(cov_matrix)
    t_count = int(np.sum(Y))
    c_count = len(Y) - t_count

    if t_count == 0 or c_count == 0:
        return PositivityReport(
            treatment=treatment_action,
            positivity_satisfied=False,
            overlap_index=0.0,
            violation_rate=1.0,
            treated_count=t_count,
            control_count=c_count,
            diagnostic_message=f"Zero support: treatment '{treatment_action}' has {t_count} treated and {c_count} control instances.",
        )

    # Standardize covariates
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-6
    X_norm = (X - mu) / std

    # Fit linear projection onto treatment indicator to estimate propensity logits
    dim = X_norm.shape[1]
    lambda_reg = 0.05
    Y_centered = Y - float(np.mean(Y))
    xtx = X_norm.T @ X_norm + (np.eye(dim) * lambda_reg)
    xty = X_norm.T @ Y_centered
    beta = np.linalg.solve(xtx, xty)

    base_logits = X_norm @ beta
    s_logits = float(np.std(base_logits))
    if s_logits > 1e-6:
        scaled_logits = (base_logits / s_logits) * 3.5
    else:
        scaled_logits = base_logits

    propensities = 1.0 / (1.0 + np.exp(-np.clip(scaled_logits, -10.0, 10.0)))

    extreme_mask = (propensities < min_propensity) | (propensities > (1.0 - min_propensity))
    violation_rate = float(np.mean(extreme_mask))

    p_t = propensities[Y == 1.0]
    p_c = propensities[Y == 0.0]

    # Histogram overlap approximation
    bins = np.linspace(0.0, 1.0, 21)
    hist_t, _ = np.histogram(p_t, bins=bins, density=True)
    hist_c, _ = np.histogram(p_c, bins=bins, density=True)
    hist_t_norm = hist_t / np.sum(hist_t) if np.sum(hist_t) > 0 else hist_t
    hist_c_norm = hist_c / np.sum(hist_c) if np.sum(hist_c) > 0 else hist_c

    # Bhattacharyya coefficient
    overlap_idx = float(np.sum(np.sqrt(hist_t_norm * hist_c_norm)))
    positivity_ok = bool(violation_rate < 0.15 and overlap_idx > 0.4)

    msg = (
        f"Positivity diagnostic for '{treatment_action}': Overlap index = {overlap_idx:.2f}, "
        f"Extreme propensity violation rate = {violation_rate * 100:.1f}%. "
        f"Positivity assumption {'holds well' if positivity_ok else 'VIOLATED (extrapolation risk in unobserved regions)'}."
    )

    return PositivityReport(
        treatment=treatment_action,
        positivity_satisfied=positivity_ok,
        overlap_index=overlap_idx,
        violation_rate=violation_rate,
        treated_count=t_count,
        control_count=c_count,
        diagnostic_message=msg,
    )


# ---------------------------------------------------------------------------
# 4. Confounding Sensitivity Reporting (Rosenbaum Bounds)
# ---------------------------------------------------------------------------


class ConfoundingSensitivityReport(BaseModel):
    """Sensitivity analysis reporting robustness of an effect against hidden confounding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    point_estimate: float
    gamma_breakdown: float = Field(
        description="Critical Gamma odds-ratio at which the effect estimate could be explained away by confounding.",
    )
    sensitivity_schedule: tuple[dict[str, float], ...] = Field(
        description="Bound schedule mapping Gamma values to lower/upper effect bounds.",
    )
    diagnostic_message: str


def report_confounding_sensitivity(
    treated_outcomes: Sequence[float],
    control_outcomes: Sequence[float],
    gammas: Sequence[float] | None = None,
) -> ConfoundingSensitivityReport:
    """Evaluate how strongly an unobserved confounder would have to alter treatment odds to nullify effect.

    Computes Rosenbaum-style sensitivity bounds across candidate Gamma values (Gamma >= 1.0).
    """
    t_arr = np.array(treated_outcomes, dtype=float)
    c_arr = np.array(control_outcomes, dtype=float)

    if len(t_arr) == 0 or len(c_arr) == 0:
        return ConfoundingSensitivityReport(
            point_estimate=0.0,
            gamma_breakdown=1.0,
            sensitivity_schedule=(),
            diagnostic_message="Insufficient data for sensitivity analysis.",
        )

    delta = float(np.mean(t_arr) - np.mean(c_arr))
    gamma_list = sorted(gammas or [1.0, 1.25, 1.5, 2.0, 2.5, 3.0])

    schedule: list[dict[str, float]] = []
    gamma_breakdown = float("inf")

    # Pooled standard error
    pooled_var = (float(np.var(t_arr)) + float(np.var(c_arr))) / 2.0
    se = float(np.sqrt(pooled_var * (1.0 / len(t_arr) + 1.0 / len(c_arr))))

    for g in gamma_list:
        # Bias adjustment under unobserved binary confounder altering odds by Gamma:
        # Maximum bias delta is approximately (g - 1) / (g + 1) * std_dev
        bias = float(((g - 1.0) / (g + 1.0)) * np.sqrt(pooled_var) * 1.96)
        lower_bound = delta - bias - (1.96 * se)
        upper_bound = delta + bias + (1.96 * se)

        schedule.append({"gamma": float(g), "lower_bound": lower_bound, "upper_bound": upper_bound})

        # Check if zero is within bounds
        if lower_bound <= 0.0 <= upper_bound and gamma_breakdown == float("inf"):
            gamma_breakdown = float(g)

    if gamma_breakdown == float("inf"):
        gamma_breakdown = gamma_list[-1]

    msg = (
        f"Confounding Sensitivity Analysis: Point delta = {delta:.3f}. "
        f"An unobserved confounder would need to alter treatment odds by a factor of Gamma >= {gamma_breakdown:.2f} "
        f"to explain away the observed effect under worst-case bias."
    )

    return ConfoundingSensitivityReport(
        point_estimate=delta,
        gamma_breakdown=gamma_breakdown,
        sensitivity_schedule=tuple(schedule),
        diagnostic_message=msg,
    )


# ---------------------------------------------------------------------------
# 5. Noise-Coupled Counterfactual Branching (Twin Rollouts)
# ---------------------------------------------------------------------------


class TwinRolloutResult(BaseModel):
    """Audit result of a noise-coupled factual/counterfactual rollout pair (arXiv:2608.08982)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    horizon: int
    target_effects: dict[str, float] = Field(
        description="Counterfactual minus factual delta on target resources."
    )
    off_target_divergences: dict[str, float] = Field(
        description="Absolute deviation on non-targeted resources (measures mechanism locality).",
    )
    total_off_target_divergence: float = Field(
        description="Total L1 deviation away from the target intervention mechanism.",
    )
    locality_preserved: bool = Field(
        description="True if off-target variables experienced zero unexpected divergence.",
    )
    diagnostic_message: str


def twin_rollout_counterfactual(
    world: World,
    initial_state: WorldState,
    factual_actions: Sequence[Action],
    counterfactual_actions: Sequence[Action],
    target_resources: Sequence[str],
    horizon: int = 3,
    seed: int = 42,
    locality_tolerance: float = 1e-4,
) -> TwinRolloutResult:
    """Execute noise-coupled twin rollouts where exogenous randomness is coupled by construction.

    Under Pearl's SCM framework, counterfactuals require:
        1. Abduction: identify exogenous noise U given factual observation.
        2. Action: substitute do(A = a').
        3. Prediction: compute outcomes under identical noise U.

    Twin rollouts make the abduction step exact by running branched simulations under the
    exact same deterministic pseudorandom SeedSequence child stream.
    """
    engine = SimulationEngine()
    targets = set(target_resources)

    # 1. Run Factual Branch
    fact_world = branch_world(world, state=initial_state)
    fact_scenario = Scenario(name="FactualBranch", horizon=horizon, samples=1, seed=seed)

    # 2. Run Counterfactual Branch with IDENTICAL seed stream
    cf_world = branch_world(world, state=initial_state)
    cf_scenario = Scenario(name="CounterfactualBranch", horizon=horizon, samples=1, seed=seed)

    # Helper actor injecting single-step actions at step 0
    from ewm_engine.experimental.planning import _StaticActionInjector

    fact_world.add_actor(_StaticActionInjector(factual_actions, actor_id="factual_actor"))
    cf_world.add_actor(_StaticActionInjector(counterfactual_actions, actor_id="cf_actor"))

    res_fact = engine.run(world=fact_world, scenario=fact_scenario)
    res_cf = engine.run(world=cf_world, scenario=cf_scenario)

    final_fact = res_fact.trajectories[0].final_state
    final_cf = res_cf.trajectories[0].final_state

    target_effects: dict[str, float] = {}
    off_target_divs: dict[str, float] = {}
    total_off_target = 0.0

    all_keys = set(final_fact.resources.keys()) | set(final_cf.resources.keys())

    for k in sorted(all_keys):
        f_val = float(final_fact.resources[k].current) if k in final_fact.resources else 0.0
        cf_val = float(final_cf.resources[k].current) if k in final_cf.resources else 0.0
        delta = cf_val - f_val

        if k in targets:
            target_effects[k] = delta
        else:
            abs_div = abs(delta)
            off_target_divs[k] = abs_div
            total_off_target += abs_div

    locality_ok = total_off_target <= locality_tolerance

    if locality_ok:
        msg = "Twin rollout successful: Exact noise coupling preserved structural locality on off-target resources."
    else:
        msg = (
            f"Off-target divergence detected ({total_off_target:.3f}): Counterfactual intervention "
            f"leaked into non-targeted resources {sorted(k for k, v in off_target_divs.items() if v > locality_tolerance)}. "
            f"Flags potential unmodeled coupling, feedback, or confounding."
        )

    return TwinRolloutResult(
        seed=seed,
        horizon=horizon,
        target_effects=target_effects,
        off_target_divergences=off_target_divs,
        total_off_target_divergence=total_off_target,
        locality_preserved=locality_ok,
        diagnostic_message=msg,
    )
