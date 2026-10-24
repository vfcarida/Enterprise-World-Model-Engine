"""Causal and epistemic foundations for Enterprise World Model simulations."""

from __future__ import annotations

from enum import StrEnum


class DependencyEvidenceNature(StrEnum):
    """Classification of a dependency or mechanism by the nature of its supporting evidence.

    Why this taxonomy exists alongside ``EvidenceLevel``:
    - ``DependencyEvidenceNature`` classifies **why we believe a dependency exists**
      (its source mechanism or empirical justification).
    - ``EvidenceLevel`` classifies the **effective query-capability rung** of a component,
      trace edge, or composite trajectory after fail-closed propagation.

    Categories:
    - ``KNOWN_FUNCTIONAL``: Deterministic accounting identities, conservation laws,
      physical capacities, or engineered system formulas where output is a known
      function of inputs.
    - ``BUSINESS_RULE_MECHANISM``: Enforced institutional, contractual, workflow, or
      software-encoded policy mechanisms that hold while the governing policy context
      $K_t$ remains in force.
    - ``EXPERIMENTALLY_SUPPORTED``: Dependencies whose directional effect has been
      empirically identified via randomized controlled experiments ($do$-interventions,
      A/B tests) or quasi-experimental designs that cleared refutation and overlap checks.
    - ``CAUSAL_HYPOTHESIS``: Domain-posited or structure-learned directed causal
      hypotheses that have **not** yet cleared empirical identification and refutation
      gates. Cannot be used for interventional claims without explicit identification
      evidence.
    - ``STATISTICAL_ASSOCIATION``: Observational correlations, predictive regressions,
      or learned conditional distributions $P(Y \\mid X)$ without causal identification.
      First-class for observational prediction and state filtering, **strictly forbidden**
      from answering interventional $do(\\cdot)$ or counterfactual queries.
    """

    KNOWN_FUNCTIONAL = "known_functional"
    STATISTICAL_ASSOCIATION = "statistical_association"
    CAUSAL_HYPOTHESIS = "causal_hypothesis"
    EXPERIMENTALLY_SUPPORTED = "experimentally_supported"
    BUSINESS_RULE_MECHANISM = "business_rule_mechanism"


EvidenceNature = DependencyEvidenceNature


class DependencyQueryKind(StrEnum):
    """Class of query executed against a dependency or world model component."""

    OBSERVATIONAL_PREDICTION = "observational_prediction"
    STATE_EVOLUTION = "state_evolution"
    INTERVENTIONAL_DO = "interventional_do"
    COUNTERFACTUAL_TWIN = "counterfactual_twin"


class EvidenceLevel(StrEnum):
    """Epistemic classification of dynamics, relationships, and simulation traces.

    EWM Engine distinguishes observational correlations from interventional mechanics:
    - COUNTERFACTUAL: Rung-3 counterfactual under a fully specified simulated SCM (noise coupling / Twin Rollout).
      Evaluated strictly within the model's structural mechanics; carries an explicit model-counterfactual disclaimer.
    - STRUCTURAL: Grounded in deterministic physical laws, accounting conservation, or hard rules.
    - INTERVENTIONAL: Empirically validated under controlled intervention / do-calculus.
    - QUASI_CAUSAL: Derived from econometric/quasi-experimental designs (e.g., diff-in-diff, IV).
    - PREDICTIVE: Observational statistical or machine-learned association P(Y | X).
    - ASSUMED: Working hypothesis or heuristic assumption not yet validated.
    """

    COUNTERFACTUAL = "counterfactual"
    STRUCTURAL = "structural"
    INTERVENTIONAL = "interventional"
    QUASI_CAUSAL = "quasi_causal"
    PREDICTIVE = "predictive"
    ASSUMED = "assumed"
    # World-model induction and causal equivalence class standings
    ASSERTED = "asserted"
    PROVISIONAL = "provisional"
    DIRECTIONAL_UNCERTAIN = "directional_uncertain"
    UNIDENTIFIED = "unidentified"
    CONFOUNDED = "confounded"
    CALIBRATED = "calibrated"
    HYPOTHETICAL = "hypothetical"


EVIDENCE_NATURE_TO_DEFAULT_LEVEL: dict[DependencyEvidenceNature, EvidenceLevel] = {
    DependencyEvidenceNature.KNOWN_FUNCTIONAL: EvidenceLevel.STRUCTURAL,
    DependencyEvidenceNature.BUSINESS_RULE_MECHANISM: EvidenceLevel.ASSERTED,
    DependencyEvidenceNature.EXPERIMENTALLY_SUPPORTED: EvidenceLevel.INTERVENTIONAL,
    DependencyEvidenceNature.CAUSAL_HYPOTHESIS: EvidenceLevel.ASSUMED,
    DependencyEvidenceNature.STATISTICAL_ASSOCIATION: EvidenceLevel.PREDICTIVE,
}

EVIDENCE_LEVEL_TO_NATURE: dict[EvidenceLevel, DependencyEvidenceNature] = {
    EvidenceLevel.COUNTERFACTUAL: DependencyEvidenceNature.KNOWN_FUNCTIONAL,
    EvidenceLevel.STRUCTURAL: DependencyEvidenceNature.KNOWN_FUNCTIONAL,
    EvidenceLevel.ASSERTED: DependencyEvidenceNature.BUSINESS_RULE_MECHANISM,
    EvidenceLevel.INTERVENTIONAL: DependencyEvidenceNature.EXPERIMENTALLY_SUPPORTED,
    EvidenceLevel.QUASI_CAUSAL: DependencyEvidenceNature.EXPERIMENTALLY_SUPPORTED,
    EvidenceLevel.CALIBRATED: DependencyEvidenceNature.STATISTICAL_ASSOCIATION,
    EvidenceLevel.PREDICTIVE: DependencyEvidenceNature.STATISTICAL_ASSOCIATION,
    EvidenceLevel.PROVISIONAL: DependencyEvidenceNature.CAUSAL_HYPOTHESIS,
    EvidenceLevel.DIRECTIONAL_UNCERTAIN: DependencyEvidenceNature.STATISTICAL_ASSOCIATION,
    EvidenceLevel.ASSUMED: DependencyEvidenceNature.CAUSAL_HYPOTHESIS,
    EvidenceLevel.HYPOTHETICAL: DependencyEvidenceNature.CAUSAL_HYPOTHESIS,
    EvidenceLevel.UNIDENTIFIED: DependencyEvidenceNature.CAUSAL_HYPOTHESIS,
    EvidenceLevel.CONFOUNDED: DependencyEvidenceNature.STATISTICAL_ASSOCIATION,
}

PERMISSIBLE_QUERIES_BY_NATURE: dict[DependencyEvidenceNature, frozenset[DependencyQueryKind]] = {
    DependencyEvidenceNature.STATISTICAL_ASSOCIATION: frozenset(
        {
            DependencyQueryKind.OBSERVATIONAL_PREDICTION,
            DependencyQueryKind.STATE_EVOLUTION,
        }
    ),
    DependencyEvidenceNature.CAUSAL_HYPOTHESIS: frozenset(
        {
            DependencyQueryKind.OBSERVATIONAL_PREDICTION,
            DependencyQueryKind.STATE_EVOLUTION,
        }
    ),
    DependencyEvidenceNature.EXPERIMENTALLY_SUPPORTED: frozenset(
        {
            DependencyQueryKind.OBSERVATIONAL_PREDICTION,
            DependencyQueryKind.STATE_EVOLUTION,
            DependencyQueryKind.INTERVENTIONAL_DO,
        }
    ),
    DependencyEvidenceNature.BUSINESS_RULE_MECHANISM: frozenset(
        {
            DependencyQueryKind.OBSERVATIONAL_PREDICTION,
            DependencyQueryKind.STATE_EVOLUTION,
            DependencyQueryKind.INTERVENTIONAL_DO,
            DependencyQueryKind.COUNTERFACTUAL_TWIN,
        }
    ),
    DependencyEvidenceNature.KNOWN_FUNCTIONAL: frozenset(
        {
            DependencyQueryKind.OBSERVATIONAL_PREDICTION,
            DependencyQueryKind.STATE_EVOLUTION,
            DependencyQueryKind.INTERVENTIONAL_DO,
            DependencyQueryKind.COUNTERFACTUAL_TWIN,
        }
    ),
}


def evidence_nature_to_level(
    nature: DependencyEvidenceNature,
    *,
    has_identification_evidence: bool = False,
) -> EvidenceLevel:
    """Map a ``DependencyEvidenceNature`` to its effective ``EvidenceLevel``.

    - ``STATISTICAL_ASSOCIATION`` always maps to ``EvidenceLevel.PREDICTIVE`` and is
      never silently promoted to a causal level, even if ``has_identification_evidence``
      is passed as True.
    - ``CAUSAL_HYPOTHESIS`` maps to ``EvidenceLevel.ASSUMED`` by default, or upgrades
      to ``EvidenceLevel.QUASI_CAUSAL`` only when explicit identification evidence
      (backdoor adjustment, IV, or passed refutation certificate) is supplied.
    """
    if nature == DependencyEvidenceNature.STATISTICAL_ASSOCIATION:
        return EvidenceLevel.PREDICTIVE
    if nature == DependencyEvidenceNature.CAUSAL_HYPOTHESIS:
        return EvidenceLevel.QUASI_CAUSAL if has_identification_evidence else EvidenceLevel.ASSUMED
    return EVIDENCE_NATURE_TO_DEFAULT_LEVEL[nature]


def evidence_level_to_nature(level: EvidenceLevel) -> DependencyEvidenceNature:
    """Map an ``EvidenceLevel`` to its canonical ``DependencyEvidenceNature``."""
    return EVIDENCE_LEVEL_TO_NATURE[level]
