"""Off-Policy Evaluation (OPE) with mandatory diagnostics and epistemic gating.

[EXPERIMENTAL] Implements standard OPE estimators over logged simulation rollouts and
observational transition logs:
    - Direct Method (DM)
    - Inverse Propensity Scoring (IPS)
    - Self-Normalized Inverse Propensity Weighting (SNIPW / Hajek)
    - Doubly Robust (DR) - DEFAULT

Epistemic Invariant & Diagnostics Gating:
    - Every OPE estimate MUST execute mandatory diagnostics:
        * Effective Sample Size (ESS)
        * Max-weight ratio
        * Weight percentiles (p95, p99) and clip rate
        * Triangulation agreement between DM, IPS, and DR
    - If diagnostics fail (e.g., severe propensity weight explosion or estimator disagreement),
      confidence is downgraded (e.g., from INTERVENTIONAL to PREDICTIVE / ASSUMED).
    - Under NO circumstances is EvidenceLevel auto-upgraded.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.evaluation.uncertainty import (
    BootstrapConfidenceInterval,
    compute_bootstrap_ci,
)
from ewm_engine.provenance.evidence import EvidenceLevel


class OPEDiagnostics(BaseModel):
    """Mandatory diagnostic metrics auditing importance-sampling stability and overlap."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    effective_sample_size: float = Field(
        description="Kish's Effective Sample Size: (sum w_i)^2 / sum(w_i^2)."
    )
    ess_ratio: float = Field(description="Normalized ESS ratio: ESS / N in [0.0, 1.0].")
    max_weight_ratio: float = Field(
        description="Proportion of total weight concentrated in maximum single sample: max(w) / sum(w)."
    )
    weight_p95: float = Field(description="95th percentile of importance weights.")
    weight_p99: float = Field(description="99th percentile of importance weights.")
    clip_rate: float = Field(
        description="Fraction of weights truncated by the weight clipping threshold."
    )
    estimator_disagreement: float = Field(
        description="Relative divergence between Direct Method and Inverse Propensity Scoring."
    )
    diagnostics_passed: bool = Field(
        description="True if all importance sampling diagnostic checks pass nominal thresholds."
    )
    warnings: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Specific diagnostic warnings triggered during evaluation.",
    )


class OPEResult(BaseModel):
    """Gated result of Off-Policy Evaluation with multi-estimator comparison."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_value: float = Field(
        description="Primary recommended off-policy value estimate (Doubly Robust by default)."
    )
    recommended_estimator: str = Field(default="doubly_robust")
    dm_value: float = Field(description="Direct Method (DM) point estimate.")
    ips_value: float = Field(description="Inverse Propensity Scoring (IPS) point estimate.")
    snipw_value: float = Field(
        description="Self-Normalized Inverse Propensity Weighting (SNIPW / Hajek) point estimate."
    )
    dr_value: float = Field(description="Doubly Robust (DR) point estimate.")
    confidence_interval: BootstrapConfidenceInterval = Field(
        description="Bootstrap confidence interval for the primary estimator."
    )
    diagnostics: OPEDiagnostics = Field(
        description="Mandatory diagnostic suite auditing importance weights and overlap."
    )
    evidence_level: EvidenceLevel = Field(
        description="Epistemic evidence level. Downgraded if diagnostics fail; never auto-promoted."
    )
    passed_gate: bool = Field(
        description="True if OPE diagnostics passed and evidence level was not downgraded."
    )
    gate_reason: str = Field(
        description="Epistemic rationale describing gate pass or downgrade condition."
    )
    epistemic_note: str = Field(
        default=(
            "OPE evaluates counterfactual policy returns from logged behavior. "
            "High variance in importance weights indicates unobserved policy support breach, "
            "triggering an automated epistemic confidence downgrade (no auto-upgrade allowed)."
        )
    )


def evaluate_off_policy(
    rewards: Sequence[float],
    behavior_probs: Sequence[float],
    target_probs: Sequence[float],
    direct_estimates: Sequence[float] | None = None,
    weight_clip: float = 20.0,
    min_ess_ratio: float = 0.10,
    max_weight_ratio_threshold: float = 0.20,
    initial_evidence_level: EvidenceLevel = EvidenceLevel.INTERVENTIONAL,
    n_resamples: int = 500,
    seed: int | None = 42,
) -> OPEResult:
    """Evaluate off-policy value of a target policy with mandatory diagnostic gates.

    Estimators:
        - DM: E[ Q_hat(s, a) ]
        - IPS: (1/N) * sum_i w_i * r_i
        - SNIPW: sum_i (w_i * r_i) / sum_i w_i
        - DR: (1/N) * sum_i [ Q_hat(s, a_target) + w_i * (r_i - Q_hat(s, a_logged)) ]

    Args:
        rewards: Observed outcomes / returns r_i for each logged transition.
        behavior_probs: Logging policy probabilities mu(a_i | s_i) > 0.
        target_probs: Target policy probabilities pi(a_i | s_i) >= 0.
        direct_estimates: Optional model outcome predictions Q_hat(s_i, a_i). If None, defaults to mean reward.
        weight_clip: Maximum allowed importance weight before clipping.
        min_ess_ratio: Minimum acceptable ESS / N ratio (default 0.10).
        max_weight_ratio_threshold: Maximum allowable proportion of total weight on a single sample.
        initial_evidence_level: Initial claimed evidence level (e.g., INTERVENTIONAL).
        n_resamples: Number of bootstrap resamples for confidence interval.
        seed: Random seed for bootstrap confidence intervals.

    Returns:
        OPEResult containing point estimates, bootstrap CIs, diagnostics, and gated evidence level.
    """
    n = min(len(rewards), len(behavior_probs), len(target_probs))
    if n == 0:
        empty_diag = OPEDiagnostics(
            effective_sample_size=0.0,
            ess_ratio=0.0,
            max_weight_ratio=1.0,
            weight_p95=0.0,
            weight_p99=0.0,
            clip_rate=0.0,
            estimator_disagreement=0.0,
            diagnostics_passed=False,
            warnings=("Empty evaluation dataset.",),
        )
        return OPEResult(
            policy_value=0.0,
            dm_value=0.0,
            ips_value=0.0,
            snipw_value=0.0,
            dr_value=0.0,
            confidence_interval=compute_bootstrap_ci([]),
            diagnostics=empty_diag,
            evidence_level=EvidenceLevel.ASSUMED,
            passed_gate=False,
            gate_reason="Empty evaluation dataset; gate blocked.",
        )

    r = np.asarray(rewards[:n], dtype=float)
    mu = np.clip(np.asarray(behavior_probs[:n], dtype=float), 1e-6, 1.0)
    pi = np.clip(np.asarray(target_probs[:n], dtype=float), 0.0, 1.0)

    # Importance weights w_i = pi_i / mu_i
    raw_weights = pi / mu
    clipped_weights = np.clip(raw_weights, 0.0, weight_clip)
    clip_rate = float(np.mean(raw_weights > weight_clip))

    # Diagnostics on weights
    sum_w = float(np.sum(clipped_weights))
    sum_w_sq = float(np.sum(clipped_weights**2))
    ess = (sum_w**2) / (sum_w_sq + 1e-9) if sum_w_sq > 0 else 0.0
    ess_ratio = ess / n
    max_w = float(np.max(clipped_weights))
    max_w_ratio = (max_w / sum_w) if sum_w > 0 else 1.0
    w_p95 = float(np.percentile(clipped_weights, 95.0))
    w_p99 = float(np.percentile(clipped_weights, 99.0))

    # Direct model baseline Q_hat
    if direct_estimates is not None and len(direct_estimates) >= n:
        q_hat = np.asarray(direct_estimates[:n], dtype=float)
    else:
        q_hat = np.full(n, np.mean(r), dtype=float)

    # 1. Direct Method (DM)
    dm_val = float(np.mean(q_hat))

    # 2. Inverse Propensity Scoring (IPS)
    ips_terms = clipped_weights * r
    ips_val = float(np.mean(ips_terms))

    # 3. Self-Normalized IPS (SNIPW / Hajek)
    snipw_val = float((sum_w > 0 and (np.sum(clipped_weights * r) / sum_w)) or 0.0)

    # 4. Doubly Robust (DR)
    # DR combines DM baseline with importance-weighted residual correction
    dr_terms = q_hat + clipped_weights * (r - q_hat)
    dr_val = float(np.mean(dr_terms))

    # Diagnostic warnings & Triangulation
    warnings: list[str] = []
    disagreement = float(abs(ips_val - dm_val) / (abs(dr_val) + 1.0))

    if ess_ratio < min_ess_ratio:
        warnings.append(
            f"Low Effective Sample Size: ESS ratio {ess_ratio:.3f} < threshold {min_ess_ratio:.3f}."
        )
    if max_w_ratio > max_weight_ratio_threshold:
        warnings.append(
            f"Weight concentration hazard: Single observation holds {max_w_ratio * 100:.1f}% of total weight."
        )
    if clip_rate > 0.05:
        warnings.append(
            f"Severe weight clipping: {clip_rate * 100:.1f}% of importance weights exceeded clip threshold {weight_clip}."
        )
    if disagreement > 0.50:
        warnings.append(
            f"High estimator disagreement: DM ({dm_val:.3f}) and IPS ({ips_val:.3f}) diverge significantly (delta ratio {disagreement:.2f})."
        )

    diag_passed = len(warnings) == 0

    # Epistemic Gate: NO AUTO-UPGRADE. Downgrade if diagnostics fail.
    evidence_level = initial_evidence_level
    if not diag_passed:
        if evidence_level == EvidenceLevel.INTERVENTIONAL:
            evidence_level = EvidenceLevel.PREDICTIVE
            gate_reason = (
                f"OPE diagnostics failed ({len(warnings)} warnings). "
                f"Evidence level downgraded from INTERVENTIONAL to PREDICTIVE. "
                f"Primary reason: {warnings[0]}"
            )
        else:
            evidence_level = EvidenceLevel.ASSUMED
            gate_reason = (
                f"OPE diagnostics failed on non-interventional baseline. "
                f"Downgraded to ASSUMED. Primary reason: {warnings[0]}"
            )
        passed_gate = False
    else:
        passed_gate = True
        gate_reason = (
            "All OPE diagnostics passed within nominal tolerance. Evidence level maintained."
        )

    diagnostics = OPEDiagnostics(
        effective_sample_size=float(ess),
        ess_ratio=float(ess_ratio),
        max_weight_ratio=float(max_w_ratio),
        weight_p95=w_p95,
        weight_p99=w_p99,
        clip_rate=clip_rate,
        estimator_disagreement=disagreement,
        diagnostics_passed=diag_passed,
        warnings=tuple(warnings),
    )

    # Compute bootstrap CI on the primary DR estimator terms
    ci = compute_bootstrap_ci(
        values=dr_terms,
        confidence_level=0.95,
        n_resamples=n_resamples,
        seed=seed,
    )

    return OPEResult(
        policy_value=dr_val,
        recommended_estimator="doubly_robust",
        dm_value=dm_val,
        ips_value=ips_val,
        snipw_value=snipw_val,
        dr_value=dr_val,
        confidence_interval=ci,
        diagnostics=diagnostics,
        evidence_level=evidence_level,
        passed_gate=passed_gate,
        gate_reason=gate_reason,
    )
