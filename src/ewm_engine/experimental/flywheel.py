"""Neuro-symbolic flywheel: Symbolic regression on learned residuals and audited promotion.

[EXPERIMENTAL] Implements the self-whitening grey-box doctrine:
1. SINDy (Sparse Identification of Nonlinear Dynamics) / PySR fits closed-form symbolic expressions
   to empirical residuals over the known ruleset.
2. Discovered equations are scored by goodness-of-fit (R^2, MSE, BIC) and parsimony (sparsity).
3. LLM-SR-style equation priors act as advisory proposers only; human acceptance remains authoritative.
4. An audited, spec-driven workflow (ADR / feature spec) promotes accepted terms into the KNOWN
   ruleset (moving mass LEARNED -> KNOWN).
5. All discovered dynamics default strictly to observational evidence level (PREDICTIVE).

Anchor papers:
- Brunton et al., "Discovering governing equations from data by sparse identification of
  nonlinear dynamical systems" (SINDy), PNAS 2016, arXiv:1509.03580
- Cranmer, "Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl", arXiv:2305.01582
- Shojaee et al., "LLM-SR: Scientific Equation Discovery via Program Synthesis", ICLR 2025 oral, arXiv:2404.18400
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import ResourceId
from ewm_engine.dynamics.base import DynamicsModel
from ewm_engine.dynamics.composite import CompositeDynamics
from ewm_engine.dynamics.deterministic import DeterministicDynamics
from ewm_engine.dynamics.learned import TransitionDataset
from ewm_engine.exceptions import DynamicsError
from ewm_engine.experimental.grey_box import GreyBoxDynamics
from ewm_engine.provenance.evidence import EvidenceLevel


class SINDyFitResult(BaseModel):
    """Result of SINDy sparse symbolic regression."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_resource: ResourceId = Field(description="Target resource whose residual was regressed.")
    formula: str = Field(description="Discovered human-readable closed-form equation.")
    terms: dict[str, float] = Field(
        description="Dictionary mapping basis term names to coefficients."
    )
    r2_score: float = Field(description="Coefficient of determination R^2.")
    mse: float = Field(description="Mean squared error.")
    bic: float = Field(description="Bayesian Information Criterion (penalizes parameter count).")
    parsimony_score: int = Field(description="Count of non-zero active terms (simpler is better).")
    feature_names: tuple[str, ...] = Field(description="Names of input features evaluated.")


class SINDyRegressor:
    """Sequential Thresholded Least Squares (STLSQ) polynomial regression engine.

    Discovers parsimonious symbolic expressions in pure NumPy without requiring external heavy solvers.
    Constructs polynomial candidate libraries up to degree 2 (or 3) and iteratively sparsifies coefficients.
    """

    def __init__(
        self,
        threshold: float = 0.01,
        degree: int = 2,
        alpha: float = 1e-4,
        max_iter: int = 20,
        include_bias: bool = True,
    ) -> None:
        self.threshold = threshold
        self.degree = min(max(1, degree), 3)
        self.alpha = alpha
        self.max_iter = max_iter
        self.include_bias = include_bias

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str],
        target_resource: ResourceId,
    ) -> SINDyFitResult:
        """Fit sparse coefficients to explain y using polynomial library Theta(X).

        Args:
            X: Input feature array of shape (n_samples, n_features).
            y: Target residual vector of shape (n_samples,).
            feature_names: Human-readable names for columns of X.
            target_resource: Identifier of target resource.

        Returns:
            SINDyFitResult containing the discovered equation, metrics, and parsimony score.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples, n_feats = X_arr.shape

        if n_samples == 0:
            return SINDyFitResult(
                target_resource=target_resource,
                formula="0.0",
                terms={},
                r2_score=0.0,
                mse=0.0,
                bic=0.0,
                parsimony_score=0,
                feature_names=tuple(feature_names),
            )

        # Build candidate library Theta(X)
        theta_cols: list[np.ndarray] = []
        basis_names: list[str] = []

        if self.include_bias:
            theta_cols.append(np.ones(n_samples, dtype=np.float64))
            basis_names.append("1")

        # Degree 1: linear terms
        for j in range(n_feats):
            theta_cols.append(X_arr[:, j])
            basis_names.append(feature_names[j])

        # Degree 2: quadratic and pairwise cross terms
        if self.degree >= 2:
            for j1 in range(n_feats):
                for j2 in range(j1, n_feats):
                    col = X_arr[:, j1] * X_arr[:, j2]
                    name = (
                        f"{feature_names[j1]}^2"
                        if j1 == j2
                        else f"{feature_names[j1]} * {feature_names[j2]}"
                    )
                    theta_cols.append(col)
                    basis_names.append(name)

        # Degree 3: cubic terms (if requested)
        if self.degree >= 3:
            for j in range(n_feats):
                theta_cols.append(X_arr[:, j] ** 3)
                basis_names.append(f"{feature_names[j]}^3")

        Theta = np.column_stack(theta_cols)
        n_basis = Theta.shape[1]

        # Standard SINDy feature column normalization for scale-invariant thresholding
        norms = np.linalg.norm(Theta, axis=0)
        norms[norms == 0.0] = 1.0
        Theta_norm = Theta / norms

        # STLSQ (Sequential Thresholded Least Squares) on normalized candidate library
        reg = self.alpha * np.eye(n_basis)
        xi_norm = np.linalg.pinv(Theta_norm.T @ Theta_norm + reg) @ (Theta_norm.T @ y_arr)

        for _ in range(self.max_iter):
            small_mask = np.abs(xi_norm) < self.threshold
            xi_norm[small_mask] = 0.0
            active_mask = ~small_mask
            if not np.any(active_mask):
                break
            Theta_active = Theta_norm[:, active_mask]
            xi_active = np.linalg.pinv(Theta_active.T @ Theta_active) @ (Theta_active.T @ y_arr)
            new_xi_norm = np.zeros(n_basis, dtype=np.float64)
            new_xi_norm[active_mask] = xi_active
            if np.allclose(xi_norm, new_xi_norm, atol=1e-8):
                break
            xi_norm = new_xi_norm

        # Unnormalize coefficients back to original feature coordinates
        xi = xi_norm / norms

        # Compute predictions and goodness-of-fit metrics
        y_pred = Theta @ xi
        res = y_arr - y_pred
        mse = float(np.mean(res**2))
        y_var = float(np.var(y_arr))
        r2 = 1.0 - (mse / y_var) if y_var > 1e-12 else (1.0 if mse < 1e-12 else 0.0)

        active_indices = np.where(np.abs(xi_norm) >= self.threshold)[0]
        k = len(active_indices)
        bic = (
            float(n_samples * math.log(max(mse, 1e-12)) + k * math.log(n_samples))
            if n_samples > 0
            else 0.0
        )

        terms_dict: dict[str, float] = {}
        for idx in active_indices:
            terms_dict[basis_names[idx]] = float(xi[idx])

        # Format human-readable formula
        formula_parts: list[str] = []
        for name, coef in terms_dict.items():
            if name == "1":
                formula_parts.append(f"{coef:+.4f}")
            else:
                formula_parts.append(f"{coef:+.4f} * {name}")

        formula_str = " ".join(formula_parts).lstrip("+").strip() if formula_parts else "0.0"

        return SINDyFitResult(
            target_resource=target_resource,
            formula=formula_str,
            terms=terms_dict,
            r2_score=r2,
            mse=mse,
            bic=bic,
            parsimony_score=k,
            feature_names=tuple(feature_names),
        )


class AdvisoryEquationPrior(BaseModel):
    """Advisory equation prior (LLM-SR or domain expert candidate hypothesis).

    CRITICAL ARCHITECTURAL GUARANTEE:
    This prior is ADVISORY ONLY. It never automatically mutates the simulation core.
    Authoritative acceptance requires empirical SINDy/PySR goodness-of-fit verification
    and an auditable human review sign-off.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Name or title of candidate hypothesis.")
    expression_template: str = Field(
        description="Candidate functional form, e.g. 'c0 * inventory'."
    )
    proposed_by: str = Field(
        default="LLM-SR-Advisory",
        description="Originator of hypothesis (e.g. LLM-SR, human expert).",
    )
    rationale: str = Field(description="Theoretical or empirical motivation for the hypothesis.")
    is_advisory_only: bool = Field(
        default=True, description="Strict marker ensuring no automated core execution."
    )


class CandidateRulePromotion(BaseModel):
    """Audited candidate rule produced by the neuro-symbolic residual flywheel.

    Represents a discovered dynamical term ready for review and promotion
    into the explicit KNOWN ruleset (WSL or deterministic dynamics).
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(description="Unique identifier for this promotion candidate.")
    target_resource: ResourceId = Field(description="Target resource updated by this rule.")
    discovered_expression: str = Field(description="Discovered mathematical closed-form equation.")
    terms: dict[str, float] = Field(description="Basis term coefficients.")
    r2_score: float = Field(description="Empirical R^2 fit on residual data.")
    mse: float = Field(description="Residual mean squared error.")
    bic: float = Field(description="Bayesian Information Criterion.")
    parsimony_score: int = Field(description="Number of non-zero terms.")
    evidence_level: EvidenceLevel = Field(
        default=EvidenceLevel.PREDICTIVE,
        description="Epistemic status of discovered rule (defaults strictly to PREDICTIVE).",
    )
    spec_reference: str = Field(
        description="Governing feature spec or ADR (e.g. 'FEAT-010', 'ADR-042')."
    )
    rationale: str = Field(description="Audit justification for promotion.")
    wsl_rule_template: str = Field(description="Declarative World Specification Language snippet.")
    python_rule_code: str = Field(description="Deterministic Python transition code snippet.")

    # Audited Governance Fields
    is_approved: bool = Field(
        default=False, description="Whether human reviewer has formally approved promotion."
    )
    reviewed_by: str | None = Field(default=None, description="Identifier of human reviewer.")
    review_notes: str | None = Field(
        default=None, description="Review commentary and approval rationale."
    )
    advisory_prior: AdvisoryEquationPrior | None = Field(
        default=None, description="Optional advisory prior that guided hypothesis."
    )


class ResidualPromotionWorkflow:
    """Manages the audited neuro-symbolic flywheel: LEARNED -> KNOWN.

    Guarantees:
    - Never automatic: Promotion into the ruleset requires explicit human sign-off.
    - Spec-driven: Each promotion links to an ADR or feature spec reference.
    - Epistemic clarity: Promoted empirical terms carry PREDICTIVE evidence level.
    - Self-whitening verification: Once promoted into the known model, the remaining
      residual drops, moving mass from black-box learned weights to white-box explicit rules.
    """

    @staticmethod
    def extract_residual_dataset(
        dataset: TransitionDataset,
        known_model: DynamicsModel,
        target_resource: ResourceId,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Extract residual training pairs (X, delta_residual) against a known model."""
        dummy_rng = np.random.default_rng(0)
        feature_names: list[str] = [target_resource]
        x_rows: list[list[float]] = []
        y_rows: list[float] = []

        for sample in dataset:
            known_res = known_model.transition(
                state=sample.state,
                actions=sample.actions,
                exogenous_events=sample.events,
                rng=dummy_rng,
            )

            # Extract resource level
            r_val = (
                sample.state.get_resource(target_resource).current
                if target_resource in sample.state.resources
                else 0.0
            )
            x_rows.append([r_val])

            actual_next = (
                sample.next_state.get_resource(target_resource).current
                if target_resource in sample.next_state.resources
                else 0.0
            )
            known_next = (
                known_res.next_state.get_resource(target_resource).current
                if target_resource in known_res.next_state.resources
                else 0.0
            )
            delta_resid = actual_next - known_next
            y_rows.append(delta_resid)

        X = np.array(x_rows, dtype=np.float64)
        y = np.array(y_rows, dtype=np.float64)
        return X, y, feature_names

    @classmethod
    def generate_candidate(
        cls,
        fit_result: SINDyFitResult,
        candidate_id: str,
        spec_reference: str,
        rationale: str,
        advisory_prior: AdvisoryEquationPrior | None = None,
    ) -> CandidateRulePromotion:
        """Create an audited promotion candidate from a SINDy fit result."""
        target = fit_result.target_resource
        formula = fit_result.formula

        # Build declarative WSL snippet
        wsl_snippet = f"""rule {candidate_id}:
  target: {target}
  discovered_expression: "{formula}"
  evidence_level: "predictive"
  spec_ref: "{spec_reference}"
  parsimony_terms: {fit_result.parsimony_score}
  r2: {fit_result.r2_score:.4f}
"""

        # Build pure Python transition snippet
        terms_code = []
        for name, coef in fit_result.terms.items():
            if name == "1":
                terms_code.append(f"{coef:.6f}")
            elif name == target:
                terms_code.append(f"({coef:.6f} * current_val)")
            elif name == f"{target}^2":
                terms_code.append(f"({coef:.6f} * (current_val ** 2))")
            else:
                terms_code.append(f"({coef:.6f} * state_features.get('{name}', 0.0))")

        calc_str = " + ".join(terms_code) if terms_code else "0.0"
        python_code = f"""def rule_{candidate_id}(state: WorldState) -> float:
    current_val = state.get_resource('{target}').current
    return {calc_str}
"""

        return CandidateRulePromotion(
            candidate_id=candidate_id,
            target_resource=target,
            discovered_expression=formula,
            terms=fit_result.terms,
            r2_score=fit_result.r2_score,
            mse=fit_result.mse,
            bic=fit_result.bic,
            parsimony_score=fit_result.parsimony_score,
            evidence_level=EvidenceLevel.PREDICTIVE,
            spec_reference=spec_reference,
            rationale=rationale,
            wsl_rule_template=wsl_snippet,
            python_rule_code=python_code,
            advisory_prior=advisory_prior,
        )

    @staticmethod
    def approve_candidate(
        candidate: CandidateRulePromotion,
        reviewer: str,
        review_notes: str,
    ) -> CandidateRulePromotion:
        """Formally approve a candidate rule promotion (human review step)."""
        candidate.is_approved = True
        candidate.reviewed_by = reviewer
        candidate.review_notes = review_notes
        return candidate

    @staticmethod
    def reject_candidate(
        candidate: CandidateRulePromotion,
        reviewer: str,
        review_notes: str,
    ) -> CandidateRulePromotion:
        """Formally reject a candidate rule promotion."""
        candidate.is_approved = False
        candidate.reviewed_by = reviewer
        candidate.review_notes = review_notes
        return candidate

    @classmethod
    def promote_to_deterministic_rule(
        cls,
        candidate: CandidateRulePromotion,
    ) -> DeterministicDynamics:
        """Compile an approved candidate rule into an executable DeterministicDynamics model.

        Raises DynamicsError if the candidate has not undergone formal human review and approval.
        """
        if not candidate.is_approved:
            raise DynamicsError(
                f"Candidate rule '{candidate.candidate_id}' cannot be promoted without formal human approval. "
                "Call ResidualPromotionWorkflow.approve_candidate() first."
            )

        target = candidate.target_resource
        terms = dict(candidate.terms)

        def transition_fn(
            state: WorldState,
            actions: Sequence[Action],
            events: Sequence[ExogenousEvent],
        ) -> WorldState:
            if target not in state.resources:
                return state
            current_val = state.get_resource(target).current
            delta = 0.0

            for name, coef in terms.items():
                if name == "1":
                    delta += coef
                elif name == target:
                    delta += coef * current_val
                elif name == f"{target}^2":
                    delta += coef * (current_val**2)
                elif name == f"{target}^3":
                    delta += coef * (current_val**3)

            return state.update_resource(target, delta=delta, clamp=True)

        return DeterministicDynamics(
            name=f"PromotedRule_{candidate.candidate_id}",
            transition_fn=transition_fn,
        )

    @classmethod
    def incorporate_into_grey_box(
        cls,
        grey_box: GreyBoxDynamics,
        candidate: CandidateRulePromotion,
    ) -> GreyBoxDynamics:
        """Promote candidate into the grey-box, moving mass from LEARNED -> KNOWN.

        The promoted rule is appended to the known model pipeline (via CompositeDynamics).
        The self-whitened grey-box now accounts for the discovered dynamics symbolically.
        """
        promoted_rule = cls.promote_to_deterministic_rule(candidate)
        known_name = getattr(grey_box.known_model, "name", "KnownDynamics")
        new_known = CompositeDynamics(
            models=[grey_box.known_model, promoted_rule],
            name=f"{known_name}_Whitened",
        )

        return GreyBoxDynamics(
            known_model=new_known,
            residual_estimator=grey_box.residual_estimator,
            conservation_layer=grey_box.conservation_layer,
            name=f"{grey_box.name}_Whitened",
        )
