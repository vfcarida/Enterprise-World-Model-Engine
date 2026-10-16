"""Domain-specific exceptions for Enterprise World Model Engine (EWM Engine).

All exceptions inherit from EWMError to provide a consistent error hierarchy
across simulation, dynamics, constraints, and state management.
"""

from __future__ import annotations


class EWMError(Exception):
    """Base class for all Enterprise World Model Engine exceptions."""


class InvalidWorldStateError(EWMError):
    """Raised when a world state violates internal schema, structure, or invariant requirements."""


class ConstraintViolationError(EWMError):
    """Raised when a hard constraint is violated during pre-action or post-state evaluation."""

    def __init__(
        self,
        message: str,
        constraint_id: str,
        step: int | None = None,
        violating_entities: list[str] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        EWMError.__init__(self, message)
        self.constraint_id = constraint_id
        self.step = step
        self.violating_entities = violating_entities or []
        self.metadata = metadata or {}


class InvalidActionError(EWMError):
    """Raised when an actor proposes an action that is malformed or invalid."""


class ResourceBoundsError(EWMError):
    """Raised when a resource update breaches min/max bounds under hard limits."""


class SimulationConfigurationError(EWMError):
    """Raised when simulation parameters (horizon, samples, seeds) are invalid."""


class DynamicsError(EWMError):
    """Raised when a dynamics model fails to compute a state transition."""


class ScenarioError(EWMError):
    """Raised when scenario definition, initialization, or execution fails."""


class BranchingError(EWMError):
    """Raised when scenario branching from a state snapshot fails."""


class ProvenanceError(EWMError):
    """Raised when provenance tracking or causal evidence metadata is inconsistent."""


class SerializationError(SimulationConfigurationError):
    """Raised when serialization or deserialization fails."""


class SerializationSecurityError(SerializationError):
    """Raised when an unsafe serialization or deserialization operation is detected."""


# ==============================================================================
# Teaching Errors (Honesty Guardrails as Developer Experience)
# ==============================================================================


class TeachingError(EWMError):
    """Base exception for errors that teach the developer how to resolve structural refusals.

    Pattern inspired by Pydantic's structured validation errors:
    State clearly why the engine refused, how to resolve or re-formulate the query,
    and link directly to documentation.
    """

    def __init__(
        self,
        reason: str,
        remediation: str,
        doc_url: str | None = None,
        title: str | None = None,
    ) -> None:
        self.reason = reason
        self.remediation = remediation
        self.doc_url = doc_url or "https://vfcarida.github.io/Enterprise-World-Model-Engine/"
        self.title = title or self.__class__.__name__
        formatted = (
            f"\n[EWM Refusal] {self.title}\n"
            f"Why I refused: {self.reason}\n"
            f"How to resolve: {self.remediation}\n"
            f"Documentation: {self.doc_url}"
        )
        EWMError.__init__(self, formatted)


class NonIdentifiableCausalQueryError(TeachingError):
    """Raised when an interventional or counterfactual query do(X = x) is structurally non-identifiable."""

    def __init__(
        self,
        treatment: str,
        outcome: str,
        open_backdoor_paths: list[list[str]] | tuple[tuple[str, ...], ...] | None = None,
        doc_url: str | None = None,
    ) -> None:
        path_info = (
            f" via open backdoor path(s): {list(open_backdoor_paths)}"
            if open_backdoor_paths
            else ""
        )
        super().__init__(
            title="Non-Identifiable Causal Query Refusal",
            reason=(
                f"The causal effect of do({treatment}) on '{outcome}' is not identifiable{path_info} "
                "due to unobserved confounding or missing adjustment variables."
            ),
            remediation=(
                "Condition on an admissible adjustment set blocking all backdoor paths, "
                "or specify an explicit Structural Causal Model (SCM) with known functional mechanisms."
            ),
            doc_url=doc_url
            or "https://vfcarida.github.io/Enterprise-World-Model-Engine/concepts/causality/",
        )
        self.treatment = treatment
        self.outcome = outcome
        self.open_backdoor_paths = open_backdoor_paths or ()


class UncertifiedCalibrationError(TeachingError):
    """Raised when simulation-based calibration or posterior coverage fails certified coverage guarantees."""

    def __init__(
        self,
        empirical_coverage: float,
        target_coverage: float,
        refusal_reason: str | None = None,
        doc_url: str | None = None,
    ) -> None:
        super().__init__(
            title="Uncertified Calibration Refusal",
            reason=(
                refusal_reason
                or f"Empirical coverage ({empirical_coverage:.1%}) fell below required target "
                f"coverage ({target_coverage:.1%}) under certified calibration gates."
            ),
            remediation=(
                "Increase calibration sample size, expand prior dispersion over structural parameters, "
                "or activate conformal prediction intervals (ConformalCalibrator)."
            ),
            doc_url=doc_url
            or "https://vfcarida.github.io/Enterprise-World-Model-Engine/concepts/uncertainty/",
        )
        self.empirical_coverage = empirical_coverage
        self.target_coverage = target_coverage


class ConstraintViolationTeachingError(ConstraintViolationError, TeachingError):
    """Raised when a hard operational constraint is violated, returning an actionable teaching explanation."""

    def __init__(
        self,
        constraint_id: str,
        step: int | None = None,
        violating_entities: list[str] | None = None,
        reason: str | None = None,
        remediation: str | None = None,
        doc_url: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        step_str = f" at step {step}" if step is not None else ""
        ent_str = f" for entities {violating_entities}" if violating_entities else ""
        r_msg = reason or f"Hard invariant '{constraint_id}' was violated{step_str}{ent_str}."
        h_msg = remediation or (
            "Adjust proposed action parameters (e.g., quantities, capacity limits) to respect physical and "
            "accounting conservation laws, or configure this invariant as a SOFT constraint with a penalty."
        )
        d_url = (
            doc_url
            or "https://vfcarida.github.io/Enterprise-World-Model-Engine/concepts/constraints/"
        )
        formatted = (
            f"\n[EWM Refusal] Hard Invariant Violation [{constraint_id}]\n"
            f"Why I refused: {r_msg}\n"
            f"How to resolve: {h_msg}\n"
            f"Documentation: {d_url}"
        )
        ConstraintViolationError.__init__(
            self,
            message=formatted,
            constraint_id=constraint_id,
            step=step,
            violating_entities=violating_entities,
            metadata=metadata,
        )
        self.reason = r_msg
        self.remediation = h_msg
        self.doc_url = d_url
        self.title = f"Hard Invariant Violation [{constraint_id}]"


class RefusalToRankError(TeachingError):
    """Raised when decision criteria or robustness metrics disagree under deep uncertainty."""

    def __init__(
        self,
        disagreements: dict[str, str] | None = None,
        doc_url: str | None = None,
    ) -> None:
        dis_info = f" Conflicting criteria: {disagreements}." if disagreements else ""
        super().__init__(
            title="Epistemic Refusal to Rank Decisions",
            reason=(
                f"Pareto robustness metrics disagree across candidate policies under deep uncertainty.{dis_info} "
                "EWM strictly refuses to declare an automated single winner when criteria conflict."
            ),
            remediation=(
                "Supply an explicit multi-attribute utility function with trade-off weights, "
                "or inspect the full non-dominated frontier in `ScenarioComparison.pareto_frontier`."
            ),
            doc_url=doc_url
            or "https://vfcarida.github.io/Enterprise-World-Model-Engine/concepts/decision/",
        )
        self.disagreements = disagreements or {}
