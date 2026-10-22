"""Scientific Benchmark Protocol and typed result models for EWM Engine.

Defines the benchmarking protocol as a research instrument measuring the structural
generalization, constraint handling, causal correctness, calibration, deployment shift
recovery, partial observability, and decision quality of enterprise world models.

DISCLAIMER: Benchmark results are strictly properties of artificial, synthetic
environments. They do not constitute empirical validation of real-world superiority.
"""

from __future__ import annotations

import hashlib
import math
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, Protocol

import numpy as np
import pydantic
import yaml
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine import __version__ as ENGINE_VERSION
from ewm_engine.dynamics.base import DynamicsModel

BENCHMARK_SPEC_VERSION = "2.0.0"
METRIC_IMPLEMENTATION_VERSION = "2.0.0"


class BenchmarkFamily(StrEnum):
    """Canonical benchmark families probing structural and enterprise world-model capabilities."""

    # Original five shift families (ADR-025)
    INTERVENTION_SHIFT = "InterventionShift"
    RULE_SHIFT = "RuleShift"
    CONSTRAINT_STRESS = "ConstraintStress"
    LONG_HORIZON = "LongHorizon"
    MULTI_AGENT_CASCADE = "MultiAgentCascade"

    # Twelve independent Enterprise World-Model Evaluation families (ADR-054, A-L)
    A_DIRECT_DYNAMICS = "A_DirectDynamics"
    B_HIDDEN_WORKFLOW_CASCADES = "B_HiddenWorkflowCascades"
    C_RUNTIME_CONFIGURATION = "C_RuntimeConfiguration"
    D_DEPLOYMENT_SHIFT = "D_DeploymentShift"
    E_PARTIAL_OBSERVABILITY = "E_PartialObservability"
    F_LONG_HORIZON_BUSINESS_OPERATION = "F_LongHorizonBusinessOperation"
    G_MULTI_AGENT_ECONOMY = "G_MultiAgentEconomy"
    H_CAUSAL_COUNTERFACTUAL_CORRECTNESS = "H_CausalCounterfactualCorrectness"
    I_CONSERVATION_CONSTRAINT_STRESS = "I_ConservationConstraintStress"
    J_UNCERTAINTY_CALIBRATION = "J_UncertaintyCalibration"
    K_OOD_REGIME_SHIFT = "K_OODRegimeShift"
    L_DECISION_QUALITY = "L_DecisionQuality"


class EnterpriseBenchmarkFamily(StrEnum):
    """The twelve independent enterprise world-model benchmark families (A through L)."""

    A_DIRECT_DYNAMICS = "A_DirectDynamics"
    B_HIDDEN_WORKFLOW_CASCADES = "B_HiddenWorkflowCascades"
    C_RUNTIME_CONFIGURATION = "C_RuntimeConfiguration"
    D_DEPLOYMENT_SHIFT = "D_DeploymentShift"
    E_PARTIAL_OBSERVABILITY = "E_PartialObservability"
    F_LONG_HORIZON_BUSINESS_OPERATION = "F_LongHorizonBusinessOperation"
    G_MULTI_AGENT_ECONOMY = "G_MultiAgentEconomy"
    H_CAUSAL_COUNTERFACTUAL_CORRECTNESS = "H_CausalCounterfactualCorrectness"
    I_CONSERVATION_CONSTRAINT_STRESS = "I_ConservationConstraintStress"
    J_UNCERTAINTY_CALIBRATION = "J_UncertaintyCalibration"
    K_OOD_REGIME_SHIFT = "K_OODRegimeShift"
    L_DECISION_QUALITY = "L_DecisionQuality"


class BaselineParadigm(StrEnum):
    """Four canonical world-model baseline paradigms testing the hybrid-context hypothesis."""

    RULES_ONLY = "rules-only"
    LEARNED_ONLY = "learned-only"
    STATIC_HYBRID = "static hybrid"
    CONTEXTUAL_HYBRID = "hybrid + runtime context discovery"


class BenchmarkTaskType(StrEnum):
    """Eleven distinct evaluation task types exercised across the benchmark suite."""

    NEXT_STATE_PREDICTION = "next_state_prediction"
    ACTION_EFFECT_PREDICTION = "action_effect_prediction"
    CONSTRAINT_PREDICTION = "constraint_prediction"
    WORKFLOW_SIDE_EFFECT_PREDICTION = "workflow_side_effect_prediction"
    AUDIT_TRACE_RECONSTRUCTION = "audit_trace_reconstruction"
    CONFIGURATION_CONTEXT_INFERENCE = "configuration_context_inference"
    COUNTERFACTUAL_QUERY = "counterfactual_query"
    POLICY_RANKING = "policy_ranking"
    ROBUST_ACTION_SELECTION = "robust_action_selection"
    MODEL_ABSTENTION = "model_abstention"
    CONFIDENCE_CALIBRATION = "confidence_calibration"


class MetricDimension(StrEnum):
    """Thirteen orthogonal evaluation metric dimensions (never collapsed into one scalar)."""

    TRANSITION_FIDELITY = "TRANSITION_FIDELITY"
    CONSTRAINT_PRESERVATION = "CONSTRAINT_PRESERVATION"
    CAUSAL_CORRECTNESS = "CAUSAL_CORRECTNESS"
    CALIBRATION = "CALIBRATION"
    LONG_HORIZON_ERROR_GROWTH = "LONG_HORIZON_ERROR_GROWTH"
    SHIFT_ROBUSTNESS = "SHIFT_ROBUSTNESS"
    TIME_TO_RECOVERY_AFTER_SHIFT = "TIME_TO_RECOVERY_AFTER_SHIFT"
    DECISION_REGRET = "DECISION_REGRET"
    ABSTENTION_QUALITY = "ABSTENTION_QUALITY"
    TRACE_PROVENANCE_COMPLETENESS = "TRACE_PROVENANCE_COMPLETENESS"
    LATENCY = "LATENCY"
    MEMORY = "MEMORY"
    THROUGHPUT = "THROUGHPUT"


class MetricDefinition(BaseModel):
    """Formal mathematical specification of an evaluation metric dimension."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension: MetricDimension = Field(description="Canonical metric dimension identifier.")
    symbol: str = Field(description="Mathematical symbol for the dimension.")
    formula_latex: str = Field(description="Formal mathematical definition in LaTeX.")
    unit: str = Field(description="Measurement unit or bounded range.")
    direction: Literal["higher_is_better", "lower_is_better"] = Field(
        description="Optimization direction for interpretation."
    )
    description: str = Field(description="Scientific meaning and failure mode probed.")


METRIC_DEFINITIONS: dict[MetricDimension, MetricDefinition] = {
    MetricDimension.TRANSITION_FIDELITY: MetricDefinition(
        dimension=MetricDimension.TRANSITION_FIDELITY,
        symbol=r"\mathcal{F}_{\text{trans}}",
        formula_latex=r"\mathcal{F}_{\text{trans}} = \frac{1}{1 + \frac{1}{N}\sum_{i=1}^N \|S_{t+1}^{(i)} - \hat{S}_{t+1}^{(i)}\|_1 / Z}",
        unit="score in (0, 1]",
        direction="higher_is_better",
        description="Normalized state transition fidelity (1.0 = exact oracle match; decreases monotonically with normalized MAE).",
    ),
    MetricDimension.CONSTRAINT_PRESERVATION: MetricDefinition(
        dimension=MetricDimension.CONSTRAINT_PRESERVATION,
        symbol=r"\mathcal{C}_{\text{pres}}",
        formula_latex=r"\mathcal{C}_{\text{pres}} = 1 - \frac{1}{N}\sum_{i=1}^N \mathbb{I}[\hat{S}_{t+1}^{(i)} \notin \mathcal{F}(S_t^{(i)}, K_t)]",
        unit="pass rate in [0, 1]",
        direction="higher_is_better",
        description="Fraction of transitions satisfying all hard capacity, non-negativity, and stoichiometric conservation laws.",
    ),
    MetricDimension.CAUSAL_CORRECTNESS: MetricDefinition(
        dimension=MetricDimension.CAUSAL_CORRECTNESS,
        symbol=r"\mathcal{C}_{\text{causal}}",
        formula_latex=r"\mathcal{C}_{\text{causal}} = \frac{1}{3}\left(\text{Pass}_{\text{desc}} + \frac{1}{1 + \text{MAE}_{\text{CF}}} + \text{Acc}_{\text{refusal}}\right)",
        unit="score in [0, 1]",
        direction="higher_is_better",
        description="Composite of intervention descendant-selectivity, twin-rollout counterfactual accuracy, and fail-closed refusal on confounded queries.",
    ),
    MetricDimension.CALIBRATION: MetricDefinition(
        dimension=MetricDimension.CALIBRATION,
        symbol=r"\mathcal{Q}_{\text{cal}}",
        formula_latex=r"\mathcal{Q}_{\text{cal}} = \max\left(0, 1 - |\hat{\text{Cov}}_{1-\alpha} - (1-\alpha)| - \text{ACE}\right)",
        unit="score in [0, 1]",
        direction="higher_is_better",
        description="Predictive uncertainty calibration quality combining empirical interval coverage accuracy and Adaptive Calibration Error (ACE).",
    ),
    MetricDimension.LONG_HORIZON_ERROR_GROWTH: MetricDefinition(
        dimension=MetricDimension.LONG_HORIZON_ERROR_GROWTH,
        symbol=r"\gamma_H",
        formula_latex=r"\gamma_H = \frac{\text{MAE}(H) - \text{MAE}(1)}{H - 1}",
        unit="error units / step",
        direction="lower_is_better",
        description="Autoregressive error growth slope across rollout horizon H.",
    ),
    MetricDimension.SHIFT_ROBUSTNESS: MetricDefinition(
        dimension=MetricDimension.SHIFT_ROBUSTNESS,
        symbol=r"\mathcal{R}_{\text{shift}}",
        formula_latex=r"\mathcal{R}_{\text{shift}} = \frac{1}{1 + \text{MAE}_{\text{post-shift}} / Z}",
        unit="score in (0, 1]",
        direction="higher_is_better",
        description="Post-shift normalized transition fidelity under unseen configuration, workflow, or regime changes.",
    ),
    MetricDimension.TIME_TO_RECOVERY_AFTER_SHIFT: MetricDefinition(
        dimension=MetricDimension.TIME_TO_RECOVERY_AFTER_SHIFT,
        symbol=r"\Delta t_{\text{rec}}",
        formula_latex=r"\Delta t_{\text{rec}} = \min\{k \ge 0 : e_{t_{\text{shift}}+k} \le \tau_{\text{rec}}\}",
        unit="steps",
        direction="lower_is_better",
        description="Steps elapsed after a deployment shift until per-step transition error recovers below threshold (capped at post-shift horizon if unrecovered).",
    ),
    MetricDimension.DECISION_REGRET: MetricDefinition(
        dimension=MetricDimension.DECISION_REGRET,
        symbol=r"\text{Regret}(\hat{\pi})",
        formula_latex=r"\text{Regret}(\hat{\pi}) = U^*(\pi^*) - U^*(\hat{\pi})",
        unit="utility units",
        direction="lower_is_better",
        description="True oracle utility lost by executing the policy selected by the candidate world model instead of the oracle-optimal policy.",
    ),
    MetricDimension.ABSTENTION_QUALITY: MetricDefinition(
        dimension=MetricDimension.ABSTENTION_QUALITY,
        symbol=r"\mathcal{A}_{F_1}",
        formula_latex=r"\mathcal{A}_{F_1} = \frac{2\,\text{TP}_{\text{abstain}}}{2\,\text{TP}_{\text{abstain}} + \text{FP}_{\text{abstain}} + \text{FN}_{\text{abstain}}}",
        unit="score in [0, 1]",
        direction="higher_is_better",
        description="F1 score of epistemic abstention across in-competence and out-of-competence/OOD/unidentifiable inputs.",
    ),
    MetricDimension.TRACE_PROVENANCE_COMPLETENESS: MetricDefinition(
        dimension=MetricDimension.TRACE_PROVENANCE_COMPLETENESS,
        symbol=r"\mathcal{P}_{\text{trace}}",
        formula_latex=r"\mathcal{P}_{\text{trace}} = \frac{|E_{\text{trace}} \cap E_{\text{oracle}}|}{|E_{\text{oracle}}|}",
        unit="recall in [0, 1]",
        direction="higher_is_better",
        description="Fraction of ground-truth causal mechanism IDs, context versions, and workflow side-effect edges captured in SystemicTrace.",
    ),
    MetricDimension.LATENCY: MetricDefinition(
        dimension=MetricDimension.LATENCY,
        symbol=r"\tau_{\text{step}}",
        formula_latex=r"\tau_{\text{step}} = \frac{T_{\text{wall}}}{N_{\text{steps}}} \times 10^3",
        unit="ms / step",
        direction="lower_is_better",
        description="Mean wall-clock execution time per simulated transition step in milliseconds.",
    ),
    MetricDimension.MEMORY: MetricDefinition(
        dimension=MetricDimension.MEMORY,
        symbol=r"\Delta M_{\text{peak}}",
        formula_latex=r"\Delta M_{\text{peak}} = \frac{\max_{t} M_{\text{alloc}}(t) - M_{\text{base}}}{1024}",
        unit="KB",
        direction="lower_is_better",
        description="Peak memory allocation delta during evaluation in kilobytes.",
    ),
    MetricDimension.THROUGHPUT: MetricDefinition(
        dimension=MetricDimension.THROUGHPUT,
        symbol=r"\Theta",
        formula_latex=r"\Theta = \frac{N_{\text{steps}}}{T_{\text{wall}}}",
        unit="steps / sec",
        direction="higher_is_better",
        description="Simulation transition throughput in steps per second.",
    ),
}


class MetricStatisticalSummary(BaseModel):
    """Rigorous statistical summary enforcing deterministic vs. stochastic protocol discipline.

    - For deterministic exact-oracle benchmarks (`is_deterministic=True`), reports exact value
      and numerical tolerance without manufacturing statistical significance.
    - For stochastic multi-seed benchmarks (`is_deterministic=False`), records every seed,
      central estimates (mean, median), uncertainty intervals (95% bootstrap CI, IQR),
      separates Monte Carlo sampling error from internal model uncertainty, and reports
      paired effect sizes (paired delta, Cohen's d_z, Holm-Bonferroni adjusted p-value)
      plus explicit failure counts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension: MetricDimension = Field(description="Evaluated metric dimension.")
    is_deterministic: bool = Field(
        description="True if evaluated on a deterministic exact-oracle scenario."
    )
    statistical_significance_applicable: bool = Field(
        description="False for deterministic exact-oracle benchmarks; True for multi-seed stochastic benchmarks."
    )
    mean: float = Field(
        description="Exact value (deterministic) or sample mean across seeds (stochastic)."
    )
    median: float = Field(description="Sample median across seeds.")
    std: float = Field(default=0.0, description="Sample standard deviation across seeds.")
    iqr: float = Field(default=0.0, description="Interquartile range (p75 - p25) across seeds.")
    ci_lower: float = Field(
        description="Lower bound of 95% bootstrap CI (or exact value if deterministic)."
    )
    ci_upper: float = Field(
        description="Upper bound of 95% bootstrap CI (or exact value if deterministic)."
    )
    confidence_level: float = Field(default=0.95, description="Nominal bootstrap confidence level.")
    mc_standard_error: float = Field(
        default=0.0,
        description="Monte Carlo standard error across seeds (std / sqrt(n_seeds)).",
    )
    model_uncertainty_spread: float = Field(
        default=0.0,
        description="Internal model predictive spread (mean predictive std or interval half-width).",
    )
    oracle_tolerance: float | None = Field(
        default=None,
        description="Numerical tolerance for deterministic exact-oracle checks (e.g. 1e-12).",
    )
    within_oracle_tolerance: bool | None = Field(
        default=None,
        description="Whether deterministic value matches oracle target within oracle_tolerance.",
    )
    seeds: tuple[int, ...] = Field(
        default_factory=tuple,
        description="Exact sequence of random seeds evaluated.",
    )
    seed_values: tuple[float, ...] = Field(
        default_factory=tuple,
        description="Raw metric values observed for each seed.",
    )
    paired_delta_vs_rules: float | None = Field(
        default=None,
        description="Paired mean difference (model - rules_only) across identical seeds.",
    )
    paired_ci_lower: float | None = Field(
        default=None,
        description="Lower 95% bootstrap CI of paired difference vs rules-only.",
    )
    paired_ci_upper: float | None = Field(
        default=None,
        description="Upper 95% bootstrap CI of paired difference vs rules-only.",
    )
    cohens_dz: float | None = Field(
        default=None,
        description="Paired standardized effect size Cohen's d_z vs rules-only.",
    )
    raw_p_value: float | None = Field(
        default=None,
        description="Two-sided paired permutation/bootstrap p-value before multiple-testing correction.",
    )
    holm_adjusted_p_value: float | None = Field(
        default=None,
        description="Holm-Bonferroni step-down adjusted p-value across baseline comparisons.",
    )
    failed_runs: int = Field(default=0, description="Number of failed or invalidated seed runs.")
    total_runs: int = Field(default=1, description="Total number of attempted seed runs.")

    @classmethod
    def from_deterministic(
        cls,
        dimension: MetricDimension,
        value: float,
        seed: int,
        oracle_target: float | None = None,
        oracle_tolerance: float = 1e-12,
        model_uncertainty_spread: float = 0.0,
        paired_reference_value: float | None = None,
        failed_runs: int = 0,
        total_runs: int = 1,
    ) -> MetricStatisticalSummary:
        """Construct a deterministic exact-oracle metric summary without manufactured p-values."""
        val = float(value)
        within_tol: bool | None = None
        if oracle_target is not None:
            within_tol = abs(val - float(oracle_target)) <= oracle_tolerance
        paired_delta = (
            float(val - paired_reference_value) if paired_reference_value is not None else None
        )
        return cls(
            dimension=dimension,
            is_deterministic=True,
            statistical_significance_applicable=False,
            mean=val,
            median=val,
            std=0.0,
            iqr=0.0,
            ci_lower=val,
            ci_upper=val,
            confidence_level=0.95,
            mc_standard_error=0.0,
            model_uncertainty_spread=float(model_uncertainty_spread),
            oracle_tolerance=oracle_tolerance if oracle_target is not None else None,
            within_oracle_tolerance=within_tol,
            seeds=(int(seed),),
            seed_values=(val,),
            paired_delta_vs_rules=paired_delta,
            paired_ci_lower=paired_delta,
            paired_ci_upper=paired_delta,
            cohens_dz=None,
            raw_p_value=None,
            holm_adjusted_p_value=None,
            failed_runs=failed_runs,
            total_runs=total_runs,
        )

    @classmethod
    def from_stochastic(
        cls,
        dimension: MetricDimension,
        seeds: Sequence[int],
        values: Sequence[float],
        model_spreads: Sequence[float] | None = None,
        paired_reference_values: Sequence[float] | None = None,
        failed_runs: int = 0,
        total_runs: int | None = None,
        bootstrap_seed: int = 42,
        n_resamples: int = 500,
    ) -> MetricStatisticalSummary:
        """Construct a multi-seed stochastic summary with bootstrap CI, MC error, and paired effect size."""
        arr = np.asarray(values, dtype=float)
        n = len(arr)
        if n == 0:
            return cls(
                dimension=dimension,
                is_deterministic=False,
                statistical_significance_applicable=True,
                mean=0.0,
                median=0.0,
                std=0.0,
                iqr=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                seeds=tuple(int(s) for s in seeds),
                seed_values=(),
                failed_runs=failed_runs,
                total_runs=total_runs or max(1, failed_runs),
            )

        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        q25, q75 = np.percentile(arr, [25.0, 75.0])
        iqr_val = float(q75 - q25)
        mc_se = float(std_val / math.sqrt(n)) if n > 1 else 0.0
        mod_spread = (
            float(np.mean(np.asarray(model_spreads, dtype=float)))
            if model_spreads is not None and len(model_spreads) > 0
            else 0.0
        )

        rng = np.random.default_rng(bootstrap_seed)
        if n > 1 and n_resamples > 0:
            boot_idx = rng.integers(0, n, size=(n_resamples, n))
            boot_means = np.mean(arr[boot_idx], axis=1)
            ci_low = float(np.percentile(boot_means, 2.5))
            ci_high = float(np.percentile(boot_means, 97.5))
        else:
            ci_low = mean_val
            ci_high = mean_val

        paired_delta: float | None = None
        paired_low: float | None = None
        paired_high: float | None = None
        cohens_dz: float | None = None
        raw_p: float | None = None

        if paired_reference_values is not None and len(paired_reference_values) == n:
            ref_arr = np.asarray(paired_reference_values, dtype=float)
            diffs = arr - ref_arr
            paired_delta = float(np.mean(diffs))
            diff_std = float(np.std(diffs, ddof=1)) if n > 1 else 0.0

            if n > 1 and diff_std > 1e-12:
                cohens_dz = float(paired_delta / diff_std)
                p_idx = rng.integers(0, n, size=(n_resamples, n))
                boot_diffs = np.mean(diffs[p_idx], axis=1)
                paired_low = float(np.percentile(boot_diffs, 2.5))
                paired_high = float(np.percentile(boot_diffs, 97.5))
                # Two-sided sign-flip / bootstrap tail probability
                prop_pos = float(np.mean(boot_diffs >= 0.0))
                prop_neg = float(np.mean(boot_diffs <= 0.0))
                raw_p = min(1.0, max(1.0 / float(n_resamples), 2.0 * min(prop_pos, prop_neg)))
            else:
                cohens_dz = (
                    0.0 if abs(paired_delta) <= 1e-12 else float(np.sign(paired_delta) * 10.0)
                )
                paired_low = paired_delta
                paired_high = paired_delta
                raw_p = 1.0 if abs(paired_delta) <= 1e-12 else (1.0 / float(max(1, n_resamples)))

        return cls(
            dimension=dimension,
            is_deterministic=False,
            statistical_significance_applicable=True,
            mean=mean_val,
            median=median_val,
            std=std_val,
            iqr=iqr_val,
            ci_lower=ci_low,
            ci_upper=ci_high,
            confidence_level=0.95,
            mc_standard_error=mc_se,
            model_uncertainty_spread=mod_spread,
            oracle_tolerance=None,
            within_oracle_tolerance=None,
            seeds=tuple(int(s) for s in seeds),
            seed_values=tuple(float(v) for v in arr),
            paired_delta_vs_rules=paired_delta,
            paired_ci_lower=paired_low,
            paired_ci_upper=paired_high,
            cohens_dz=cohens_dz,
            raw_p_value=raw_p,
            holm_adjusted_p_value=raw_p,
            failed_runs=failed_runs,
            total_runs=total_runs if total_runs is not None else (n + failed_runs),
        )

    def format_compact(self) -> str:
        """Render a compact, scientifically honest string representation."""
        if self.is_deterministic:
            return f"{self.mean:.4f} (det)"
        half_ci = 0.5 * (self.ci_upper - self.ci_lower)
        return f"{self.mean:.4f} ± {half_ci:.4f}"


def apply_holm_bonferroni(p_values: Mapping[str, float]) -> dict[str, float]:
    """Apply Holm-Bonferroni step-down multiple-comparison adjustment to a family of p-values."""
    if not p_values:
        return {}
    sorted_items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(sorted_items)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for idx, (key, p_val) in enumerate(sorted_items):
        multiplier = m - idx
        adj = min(1.0, max(0.0, float(p_val) * multiplier))
        running_max = max(running_max, adj)
        adjusted[key] = running_max
    return adjusted


class ShiftProtocolRecord(BaseModel):
    """Mandatory shift metadata recorded for deployment-shift and configuration-shift experiments."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    training_context_version: str = Field(
        description="Context version active during training/calibration."
    )
    training_context_fingerprint: str = Field(
        description="SHA-256 fingerprint of the training ContextSnapshot."
    )
    evaluation_context_version: str = Field(
        description="Context version active after deployment shift."
    )
    evaluation_context_fingerprint: str = Field(
        description="SHA-256 fingerprint of the post-shift ContextSnapshot."
    )
    shift_step: int = Field(
        description="Simulation step at which the deployment/regime shift occurred."
    )
    what_changed: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured diff describing which rules, workflows, parameters, or mechanisms changed.",
    )
    drift_categories: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Detected DriftKind categories (e.g. rule_constraint_change, parameter_drift).",
    )
    model_informed_explicitly: bool = Field(
        description="Whether the candidate model was explicitly handed the new ContextSnapshot at shift_step."
    )
    runtime_information_available: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Runtime audit/telemetry keys available to context discovery providers.",
    )
    audit_discovery_delay_steps: int = Field(
        default=0,
        description="Latency in steps between the actual shift and when runtime audit telemetry reflects it.",
    )
    adaptation_budget_steps: int = Field(
        description="Maximum allowed steps after shift_step for the model to recover below error threshold."
    )
    recovery_error_threshold: float = Field(
        description="Per-step MAE threshold defining post-shift recovery."
    )
    steps_until_recovery: int | None = Field(
        default=None,
        description="Steps after shift_step until error <= recovery_error_threshold (None if unrecovered).",
    )
    recovered_within_budget: bool = Field(
        description="True if steps_until_recovery is not None and <= adaptation_budget_steps."
    )
    pre_shift_mae: float = Field(description="Mean absolute error before shift_step.")
    peak_post_shift_mae: float = Field(
        description="Peak per-step MAE immediately following shift_step."
    )
    steady_state_post_shift_mae: float = Field(
        description="Mean absolute error after the adaptation window."
    )


class DataLeakageCertificate(BaseModel):
    """Cryptographic audit certificate proving hard temporal, state, and configuration non-leakage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train_partition_seed: int = Field(
        description="Isolated RNG seed used for training/induction data."
    )
    eval_partition_seed: int = Field(description="Isolated RNG seed used for evaluation scenarios.")
    train_sample_count: int = Field(description="Number of training transitions generated.")
    eval_sample_count: int = Field(description="Number of evaluation transitions generated.")
    state_fingerprint_overlap_count: int = Field(
        description="Count of ontic state fingerprints shared between training and evaluation partitions."
    )
    max_train_timestamp: float = Field(description="Maximum timestamp in the training partition.")
    min_eval_timestamp: float = Field(description="Minimum timestamp in the evaluation partition.")
    temporal_boundary_verified: bool = Field(
        description="True if max_train_timestamp < min_eval_timestamp."
    )
    held_out_config_versions: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Evaluation context versions strictly withheld from training.",
    )
    config_boundary_verified: bool = Field(
        description="True if no held-out evaluation context version appeared in training."
    )
    oracle_isolation_verified: bool = Field(
        description="True if candidate models never received ground-truth next_state or future schedule."
    )
    passed: bool = Field(description="True only if all leakage checks passed with zero violations.")


def verify_no_data_leakage(
    *,
    train_partition_seed: int,
    eval_partition_seed: int,
    train_state_fingerprints: Sequence[str],
    eval_state_fingerprints: Sequence[str],
    train_timestamps: Sequence[float],
    eval_timestamps: Sequence[float],
    train_context_versions: Sequence[str],
    held_out_eval_context_versions: Sequence[str],
    oracle_isolation_verified: bool = True,
) -> DataLeakageCertificate:
    """Verify hard temporal, state-fingerprint, and configuration boundaries between train and eval."""
    overlap = len(set(train_state_fingerprints).intersection(set(eval_state_fingerprints)))
    max_train_ts = float(max(train_timestamps)) if train_timestamps else 0.0
    min_eval_ts = float(min(eval_timestamps)) if eval_timestamps else 1.0
    temporal_ok = max_train_ts < min_eval_ts
    seed_ok = int(train_partition_seed) != int(eval_partition_seed)

    train_cfg_set = set(train_context_versions)
    held_out_set = set(held_out_eval_context_versions)
    cfg_ok = len(train_cfg_set.intersection(held_out_set)) == 0

    passed = seed_ok and overlap == 0 and temporal_ok and cfg_ok and bool(oracle_isolation_verified)
    return DataLeakageCertificate(
        train_partition_seed=int(train_partition_seed),
        eval_partition_seed=int(eval_partition_seed),
        train_sample_count=len(train_state_fingerprints),
        eval_sample_count=len(eval_state_fingerprints),
        state_fingerprint_overlap_count=overlap,
        max_train_timestamp=max_train_ts,
        min_eval_timestamp=min_eval_ts,
        temporal_boundary_verified=temporal_ok,
        held_out_config_versions=tuple(sorted(held_out_set)),
        config_boundary_verified=cfg_ok,
        oracle_isolation_verified=bool(oracle_isolation_verified),
        passed=passed,
    )


class ReproducibilityMetadata(BaseModel):
    """Complete reproducibility provenance attached to every enterprise benchmark result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository_commit: str = Field(description="Git commit SHA-1 of the repository.")
    benchmark_spec_version: str = Field(default=BENCHMARK_SPEC_VERSION)
    engine_version: str = Field(default=ENGINE_VERSION)
    metric_implementation_version: str = Field(default=METRIC_IMPLEMENTATION_VERSION)
    python_version: str = Field(description="Python runtime version.")
    platform_info: str = Field(description="OS and CPU architecture identifier.")
    dependencies: dict[str, str] = Field(description="Core runtime dependency versions.")
    hardware_backend: str = Field(default="cpu-numpy-deterministic")
    master_seed: int = Field(
        description="Master seed governing deterministic SeedSequence spawning."
    )
    evaluated_seeds: tuple[int, ...] = Field(description="All child seeds evaluated.")
    exact_command: str = Field(description="CLI command to reproduce this exact benchmark result.")
    dataset_scenario_fingerprint: str = Field(
        description="SHA-256 fingerprint of the generated scenario and dataset partitions."
    )
    output_fingerprint: str = Field(
        description="SHA-256 fingerprint of the canonicalized evaluation metrics."
    )
    timestamp: str = Field(description="ISO-8601 UTC timestamp of execution.")


def get_git_commit_hash() -> str:
    """Resolve current git commit SHA or return 'unknown' if outside a git worktree."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        ).strip()
        return out if out else "unknown"
    except Exception:
        return "unknown"


def capture_reproducibility_metadata(
    *,
    master_seed: int,
    evaluated_seeds: Sequence[int],
    exact_command: str,
    dataset_scenario_fingerprint: str,
    output_payload: Mapping[str, Any],
    timestamp: str | None = None,
) -> ReproducibilityMetadata:
    """Build a complete ReproducibilityMetadata record with canonical output hashing."""
    canonical_out = repr(sorted(output_payload.items())).encode("utf-8")
    out_fp = hashlib.sha256(canonical_out).hexdigest()
    return ReproducibilityMetadata(
        repository_commit=get_git_commit_hash(),
        benchmark_spec_version=BENCHMARK_SPEC_VERSION,
        engine_version=ENGINE_VERSION,
        metric_implementation_version=METRIC_IMPLEMENTATION_VERSION,
        python_version=sys.version.split()[0],
        platform_info=f"{platform.system()}-{platform.machine()}",
        dependencies={
            "numpy": np.__version__,
            "pydantic": pydantic.__version__,
            "pyyaml": getattr(yaml, "__version__", "6.0"),
        },
        hardware_backend="cpu-numpy-deterministic",
        master_seed=int(master_seed),
        evaluated_seeds=tuple(int(s) for s in evaluated_seeds),
        exact_command=exact_command,
        dataset_scenario_fingerprint=dataset_scenario_fingerprint,
        output_fingerprint=out_fp,
        timestamp=timestamp or datetime.now(UTC).isoformat(),
    )


class BaselineEvaluationRecord(BaseModel):
    """Comprehensive evaluation of one baseline paradigm on a benchmark family."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    paradigm: BaselineParadigm = Field(description="Canonical baseline paradigm identifier.")
    model_name: str = Field(description="Concrete model class or configuration name.")
    task_types_evaluated: tuple[BenchmarkTaskType, ...] = Field(
        description="Task types exercised on this baseline."
    )
    dimension_summaries: dict[MetricDimension, MetricStatisticalSummary] = Field(
        description="Separate statistical summaries for each of the 13 orthogonal metric dimensions."
    )
    task_scores: dict[BenchmarkTaskType, float] = Field(
        default_factory=dict,
        description="Task-specific primary score for each evaluated task type.",
    )
    raw_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Auxiliary diagnostic metrics (e.g. raw MAE, RMSE, CRPS, coverage_90).",
    )
    invariants_passed: bool = Field(description="True if zero hard invariant violations occurred.")
    invariant_violations: int = Field(description="Total hard invariant violations across runs.")
    failed_runs: int = Field(default=0, description="Number of failed/invalidated runs.")
    total_runs: int = Field(default=1, description="Total number of runs executed.")
    failure_modes_exposed: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Explicit failure modes surfaced for this baseline on this family.",
    )
    shift_protocol: ShiftProtocolRecord | None = Field(
        default=None,
        description="Shift protocol record if this family evaluates deployment/configuration shift.",
    )


class EnterpriseBenchmarkFamilyResult(BaseModel):
    """Complete, self-contained evaluation artifact for one Enterprise Benchmark Family (A-L)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    family_code: str = Field(description="Single-letter family code ('A' through 'L').")
    family: EnterpriseBenchmarkFamily = Field(description="Enterprise benchmark family identifier.")
    benchmark_name: str = Field(description="Descriptive benchmark scenario title.")
    is_deterministic_oracle: bool = Field(
        description="True if ground truth is an exact deterministic oracle; False if stochastic multi-seed."
    )
    task_types: tuple[BenchmarkTaskType, ...] = Field(
        description="Task types evaluated in this family."
    )
    primary_dimensions: tuple[MetricDimension, ...] = Field(
        description="Primary metric dimensions highlighted by this family."
    )
    scientific_question: str = Field(
        description="Core scientific question probed by this benchmark family."
    )
    phenomenon_surfaced: str = Field(
        description="Empirical failure mode and architectural tradeoff surfaced."
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Scenario generator parameters and horizon configuration.",
    )
    baselines: tuple[BaselineEvaluationRecord, ...] = Field(
        description="Comparative evaluation records across the four canonical baselines."
    )
    leakage_certificate: DataLeakageCertificate = Field(
        description="Cryptographic verification of zero train/eval data or configuration leakage."
    )
    reproducibility: ReproducibilityMetadata = Field(
        description="Full reproducibility metadata and fingerprints."
    )


class ModelEvaluationSummary(BaseModel):
    """Evaluation summary of an individual DynamicsModel on a benchmark family."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str = Field(description="Name or class of the dynamics model.")
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Numeric metrics computed for this model (e.g. MAE, divergence, violation rate).",
    )
    invariants_passed: bool = Field(
        default=True,
        description="Whether declared invariants (bounds, conservation) held during rollout.",
    )
    invariant_violations: int = Field(
        default=0,
        description="Total number of invariant breaches during evaluation.",
    )
    divergence_or_gap: float = Field(
        default=0.0,
        description="Primary gap or divergence metric (e.g. interventional gap, horizon drift).",
    )
    diagnostics: dict[str, Any] = Field(
        default_factory=dict,
        description="Model-specific diagnostics and execution traces.",
    )


class BenchmarkResult(BaseModel):
    """Machine-readable, seeded, and reproducible benchmark outcome with full provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark_name: str = Field(description="Specific benchmark instance title.")
    family: BenchmarkFamily = Field(description="Canonical benchmark family category.")
    seed: int = Field(description="Random master seed used for execution.")
    provenance_fingerprint: str = Field(
        description="SHA-256 hash sealing seed, parameters, library versions, and platform."
    )
    timestamp: str = Field(description="ISO-8601 UTC timestamp of execution.")
    engine_version: str = Field(default=ENGINE_VERSION, description="EWM Engine release version.")
    python_version: str = Field(
        default_factory=lambda: sys.version.split()[0],
        description="Python runtime environment version.",
    )
    platform_info: str = Field(
        default_factory=lambda: f"{platform.system()}-{platform.machine()}",
        description="Hardware and operating system architecture.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Experimental benchmark parameters and hyperparameters.",
    )
    models_evaluated: list[ModelEvaluationSummary] = Field(
        default_factory=list,
        description="Comparative metrics across all evaluated dynamics models.",
    )
    phenomenon_surfaced: str = Field(
        description="Scientific phenomenon revealed (e.g. interventional degradation, compounding drift).",
    )
    epistemic_disclaimer: str = Field(
        default=(
            "SYNTHETIC RESEARCH INSTRUMENT: Results reflect properties of artificial, "
            "synthetic worlds only. They do NOT imply real-world causal validity or automated "
            "decision readiness. Models must not be ranked as globally 'superior' without a "
            "domain-specific utility and constraint contract."
        ),
        description="Mandatory epistemic humility banner.",
    )


class Benchmark(Protocol):
    """Protocol for an executable scientific benchmark."""

    @property
    def name(self) -> str:
        """Benchmark identifier."""
        ...

    @property
    def family(self) -> BenchmarkFamily:
        """Associated benchmark family category."""
        ...

    def run(
        self,
        models: Sequence[DynamicsModel] | None = None,
    ) -> BenchmarkResult:
        """Execute the benchmark across candidate dynamics models."""
        ...


class EnterpriseBenchmark(Protocol):
    """Protocol for an executable Enterprise World-Model Benchmark Family (A-L)."""

    @property
    def family_code(self) -> str:
        """Single-letter code ('A' through 'L')."""
        ...

    @property
    def family(self) -> EnterpriseBenchmarkFamily:
        """Associated enterprise benchmark family enum."""
        ...

    @property
    def name(self) -> str:
        """Benchmark instance title."""
        ...

    def run_enterprise(self) -> EnterpriseBenchmarkFamilyResult:
        """Execute the benchmark across the four canonical baselines and return full results."""
        ...


def compute_benchmark_fingerprint(
    benchmark_name: str,
    family: str,
    seed: int,
    parameters: dict[str, Any],
) -> str:
    """Compute a deterministic SHA-256 fingerprint sealing benchmark configuration."""
    raw = f"{benchmark_name}:{family}:{seed}:{ENGINE_VERSION}:{sorted(parameters.items())}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
