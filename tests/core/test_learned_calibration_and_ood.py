"""Tests for EWM-003: Learned Dynamics, Calibration, Conformal Intervals, OOD & Abstention.

Verifies:
1. `Predictor.predict(state)` -> `(CalibratedPrediction, ConfidenceInterval)` contract.
2. Exact ensemble variance formula: $\\text{Var}(\\mu) = \\frac{1}{N} \\sum_{i=1}^N (\\mu_i - \\bar{\\mu})^2$.
3. Feeding completely random noise to the learned component raises `AbstainException` and
   catches >95% (100%) of injected noise inputs.
4. Conformal prediction (Vovk et al., 2005) calibration and Expected Calibration Error (ECE)
   evaluation on a synthetic validation set.
5. `AbstainException` triggers safe structural fallback in `HybridTransitionRouter` and
   `TransitionEngine`, with maximum retry depth = 1 preventing infinite loops if fallback fails.
6. Default `infinity` thresholds preserve legacy behavior unless explicitly turned on.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest

from ewm_engine.core.actions import Action
from ewm_engine.core.state import Entity, Resource, WorldState
from ewm_engine.dynamics.calibration import (
    AbstainException,
    ConfidenceInterval,
    ConformalPredictorWrapper,
    LearnedModelRegistry,
    MahalanobisOODAndDriftDetector,
    Predictor,
    compute_ensemble_variance,
    compute_expected_calibration_error,
)
from ewm_engine.dynamics.learned import (
    LinearResidualDynamics,
    TransitionDataset,
    TransitionSample,
)
from ewm_engine.dynamics.router import (
    CalibratedPrediction,
    HybridTransitionRouter,
    NumericalInstabilityError,
    TransitionEngine,
)
from ewm_engine.provenance.evidence import EvidenceLevel


def _make_state(
    liquidity: float = 100.0,
    compute_budget: float = 50.0,
    step: int = 0,
) -> WorldState:
    return WorldState(
        step=step,
        timestamp=float(step),
        entities={
            "treasury": Entity(
                id="treasury",
                type="department",
                attributes={"region": "global"},
            )
        },
        resources={
            "liquidity": Resource(
                id="liquidity",
                current=liquidity,
                min_value=-1e6,
                max_value=1e6,
            ),
            "compute_budget": Resource(
                id="compute_budget",
                current=compute_budget,
                min_value=-1e6,
                max_value=1e6,
            ),
        },
    )


def _generate_synthetic_transition_datasets(
    n_train: int = 240,
    n_cal: int = 200,
    n_val: int = 400,
    seed: int = 20261030,
) -> tuple[TransitionDataset, TransitionDataset, TransitionDataset]:
    """Generate chronological train, calibration, and validation synthetic datasets."""
    rng = np.random.default_rng(seed)
    total = n_train + n_cal + n_val
    samples: list[TransitionSample] = []

    for step in range(total):
        liq = float(100.0 + rng.normal(0.0, 2.0))
        comp = float(50.0 + rng.normal(0.0, 1.5))
        act_val = float(rng.uniform(-2.0, 2.0))
        noise_liq = float(rng.normal(0.0, 0.8))
        noise_comp = float(rng.normal(0.0, 0.5))

        s_t = _make_state(liquidity=liq, compute_budget=comp, step=step)
        act = Action(
            id=f"act_{step}",
            type="rebalance",
            parameters={"value": act_val},
        )
        # True data-generating process: delta_liq = 2.5 * act_val + 1.0 + noise_liq
        next_liq = liq + (2.5 * act_val + 1.0 + noise_liq)
        next_comp = comp + (-0.8 * act_val + 0.5 + noise_comp)
        s_next = _make_state(liquidity=next_liq, compute_budget=next_comp, step=step + 1)
        samples.append(
            TransitionSample(
                state=s_t,
                actions=(act,),
                events=(),
                next_state=s_next,
            )
        )

    train_ds = TransitionDataset(samples[:n_train])
    cal_ds = TransitionDataset(samples[n_train : n_train + n_cal])
    val_ds = TransitionDataset(samples[n_train + n_cal :])
    return train_ds, cal_ds, val_ds


def test_ensemble_variance_mathematical_identity_and_ci_comparisons() -> None:
    """Verify exact Var(mu) = (1/N) sum (mu_i - mu_bar)^2 and ConfidenceInterval comparisons."""
    member_preds = [
        {"liquidity": 2.0, "compute_budget": 1.0},
        {"liquidity": 4.0, "compute_budget": 3.0},
        {"liquidity": 6.0, "compute_budget": 5.0},
        {"liquidity": 8.0, "compute_budget": 7.0},
    ]
    means, variances = compute_ensemble_variance(member_preds)

    # Mean of [2, 4, 6, 8] is 5.0; Var(mu) = ((9 + 1 + 1 + 9) / 4) = 5.0
    assert math.isclose(means["liquidity"], 5.0, rel_tol=1e-12)
    assert math.isclose(variances["liquidity"], 5.0, rel_tol=1e-12)
    # Mean of [1, 3, 5, 7] is 4.0; Var(mu) = ((9 + 1 + 1 + 9) / 4) = 5.0
    assert math.isclose(means["compute_budget"], 4.0, rel_tol=1e-12)
    assert math.isclose(variances["compute_budget"], 5.0, rel_tol=1e-12)

    empty_means, empty_vars = compute_ensemble_variance([])
    assert empty_means == {}
    assert empty_vars == {}

    with pytest.raises(NumericalInstabilityError):
        compute_ensemble_variance([{"liquidity": float("nan")}])

    ci_narrow = ConfidenceInterval(
        lower={"liquidity": 4.0},
        upper={"liquidity": 6.0},
        half_width={"liquidity": 1.0},
        width={"liquidity": 2.0},
        max_width=2.0,
        max_half_width=1.0,
        ensemble_variance={"liquidity": 0.25},
        max_ensemble_variance=0.25,
        ood_score=0.8,
        nominal_coverage=0.90,
    )
    ci_wide = ConfidenceInterval(
        lower={"liquidity": 0.0},
        upper={"liquidity": 10.0},
        half_width={"liquidity": 5.0},
        width={"liquidity": 10.0},
        max_width=10.0,
        max_half_width=5.0,
        ensemble_variance={"liquidity": 4.0},
        max_ensemble_variance=4.0,
        ood_score=3.2,
        nominal_coverage=0.90,
    )

    assert ci_narrow < 5.0
    assert ci_narrow <= 2.0
    assert not (ci_narrow > float("inf"))
    assert not (ci_narrow >= float("inf"))
    assert ci_narrow < float("inf")
    assert ci_narrow <= float("inf")
    assert ci_wide > 5.0
    assert ci_wide >= 10.0
    assert ci_wide > ci_narrow
    assert ci_wide >= ci_narrow
    assert ci_narrow < ci_wide
    assert ci_narrow <= ci_wide


def test_random_noise_raises_abstain_exception_and_catches_over_95_percent() -> None:
    """Feed completely random noise to the learned component; assert >95% raise AbstainException."""
    train_ds, cal_ds, _ = _generate_synthetic_transition_datasets()

    # Build an ensemble of 5 bootstrap-perturbed learned predictors
    rng_boot = np.random.default_rng(101)
    ensemble_members = []
    for i in range(5):
        indices = rng_boot.integers(0, len(train_ds), size=len(train_ds))
        boot_ds = TransitionDataset([train_ds[int(idx)] for idx in indices])
        m = LinearResidualDynamics(
            target_resource="liquidity",
            action_type="rebalance",
            name=f"ensemble_member_{i}",
        )
        m.fit(boot_ds)
        ensemble_members.append(m)

    ood_detector = MahalanobisOODAndDriftDetector(regularization=1e-5)
    ood_detector.fit(train_ds, tune_quantile=0.99, safety_margin=1.20)

    predictor = Predictor(
        models=ensemble_members,
        ood_detector=ood_detector,
        ood_threshold=ood_detector.ood_threshold,
        threshold=6.0,
        nominal_coverage=0.90,
    )
    predictor.calibrate(cal_ds, alpha=0.10, tune_ood_quantile=0.99)

    # 1. Verify in-distribution state returns (CalibratedPrediction, ConfidenceInterval)
    in_dist_sample = cal_ds[0]
    pred, ci = predictor.predict(in_dist_sample.state, in_dist_sample.actions)
    assert isinstance(pred, CalibratedPrediction)
    assert isinstance(ci, ConfidenceInterval)
    assert "liquidity" in pred.mean_deltas
    assert ci.max_width > 0.0
    assert ci < predictor.threshold

    # 2. Feed 200 completely random noise inputs to the learned component
    rng_noise = np.random.default_rng(99999)
    n_noise_trials = 200
    caught_noise = 0

    for step_idx in range(n_noise_trials):
        # Inject completely random high-variance Gaussian / uniform noise outside support
        sign_liq = -1.0 if (step_idx % 2 == 0) else 1.0
        sign_comp = 1.0 if (step_idx % 3 == 0) else -1.0
        noisy_liq = float(100.0 + sign_liq * rng_noise.uniform(35.0, 5000.0))
        noisy_comp = float(50.0 + sign_comp * rng_noise.uniform(25.0, 5000.0))
        noisy_state = _make_state(
            liquidity=noisy_liq,
            compute_budget=noisy_comp,
            step=step_idx,
        )
        try:
            predictor.predict(noisy_state)
        except AbstainException as exc:
            caught_noise += 1
            assert exc.confidence_interval is not None
            assert exc.ood_score > predictor.ood_threshold

    catch_rate = caught_noise / float(n_noise_trials)
    assert catch_rate > 0.95, f"Expected >95% OOD noise catch rate, got {catch_rate:.2%}"
    assert caught_noise == n_noise_trials

    # Verify online drift report reflects the burst of OOD noise
    drift_rep = ood_detector.drift_report()
    assert drift_rep.drift_detected is True
    assert drift_rep.window_ood_rate > 0.95


def test_conformal_calibration_and_expected_calibration_error_on_synthetic_validation_set() -> None:
    """Evaluate conformal calibration and Expected Calibration Error (ECE) on synthetic validation set."""
    train_ds, cal_ds, val_ds = _generate_synthetic_transition_datasets(
        n_train=250, n_cal=250, n_val=500, seed=424242
    )

    model = LinearResidualDynamics(
        target_resource="liquidity",
        action_type="rebalance",
        name="calibrated_linear_dynamics",
    )
    model.fit(train_ds)

    predictor = ConformalPredictorWrapper(
        models=model,
        nominal_coverage=0.90,
    )
    quantiles = predictor.calibrate(cal_ds, alpha=0.10)
    assert predictor.is_calibrated is True
    assert "liquidity" in quantiles
    assert quantiles["liquidity"] > 0.5

    report = predictor.evaluate_calibration(val_ds, num_bins=10, uncalibrated_scale=0.25)

    # Uncalibrated overconfident baseline has high ECE (> 0.25); conformal calibration reduces ECE to < 0.03
    assert report.uncalibrated_ece > 0.25
    assert report.expected_calibration_error < 0.035
    assert report.expected_calibration_error < report.uncalibrated_ece
    # Empirical coverage at 90% nominal target is well-calibrated within finite-sample tolerance
    assert 0.86 <= report.empirical_coverage_at_nominal <= 0.95
    assert report.num_samples == 500
    assert len(report.bin_nominal_confidences) == 10
    assert len(report.bin_empirical_accuracies) == 10

    # Also verify direct binned ECE calculation helper
    ece_val, ece_diag = compute_expected_calibration_error(
        confidences=report.bin_nominal_confidences,
        accuracies=report.bin_empirical_accuracies,
        num_bins=10,
    )
    assert 0.0 <= ece_val < 0.05
    assert len(ece_diag["bin_counts"]) == 10


def test_abstain_exception_triggers_structural_fallback_and_enforces_max_retry_depth_1() -> None:
    """Verify fallback structural rules execute upon abstention and max_retry_depth=1 prevents loops."""
    train_ds, cal_ds, _ = _generate_synthetic_transition_datasets()

    model = LinearResidualDynamics(
        target_resource="liquidity",
        action_type="rebalance",
    )
    model.fit(train_ds)

    predictor = Predictor(
        models=model,
        nominal_coverage=0.90,
    )
    predictor.calibrate(cal_ds, alpha=0.10, tune_ood_quantile=0.99)

    structural_eqs = {
        "liquidity": lambda parents, actions, noise: float(parents["liquidity"]) + 5.0,
        "compute_budget": lambda parents, actions, noise: float(parents["compute_budget"]) + 2.0,
    }

    # 1. Standard structural fallback upon OOD abstention
    engine = TransitionEngine(
        structural_equations=structural_eqs,
        ml_predictor=predictor,
    )
    in_dist_state = _make_state(liquidity=100.0, compute_budget=50.0, step=0)
    normal_outcome = engine.step(
        in_dist_state,
        Action(id="a1", type="rebalance", parameters={"value": 1.0}),
    )
    assert normal_outcome.abstained is False
    assert normal_outcome.fallback_executed is False
    assert normal_outcome.confidence_interval is not None

    ood_state = _make_state(liquidity=9500.0, compute_budget=-4200.0, step=1)
    abstain_outcome = engine.step(
        ood_state,
        Action(id="a2", type="rebalance", parameters={"value": 1.0}),
    )
    assert abstain_outcome.abstained is True
    assert abstain_outcome.fallback_executed is True
    assert abstain_outcome.fallback_retry_depth == 0
    assert abstain_outcome.fallback_failed is False
    assert abstain_outcome.ml_rejected is True
    assert abstain_outcome.rolled_back is True
    assert abstain_outcome.evidence_level == EvidenceLevel.STRUCTURAL
    assert abstain_outcome.next_state == abstain_outcome.structural_state
    assert math.isclose(abstain_outcome.next_state.get_resource("liquidity").current, 9505.0)
    assert math.isclose(abstain_outcome.next_state.get_resource("compute_budget").current, -4198.0)

    # 2. Custom fallback handler that fails on initial attempt (0) and succeeds on retry 1
    transient_attempts: list[int] = []

    def transient_failing_fallback(
        state: WorldState, actions: Sequence[Action]
    ) -> WorldState:
        transient_attempts.append(len(transient_attempts))
        if len(transient_attempts) == 1:
            raise RuntimeError("Transient fallback failure on attempt 0")
        return state.update_resource("liquidity", delta=10.0)

    retry_router = HybridTransitionRouter(
        structural_equations=structural_eqs,
        ml_predictor=predictor,
        fallback_structural_fn=transient_failing_fallback,
        max_fallback_retries=1,
    )
    retry_outcome = retry_router.step(ood_state)
    assert retry_outcome.abstained is True
    assert retry_outcome.fallback_executed is True
    assert retry_outcome.fallback_retry_depth == 1
    assert retry_outcome.fallback_failed is False
    assert len(transient_attempts) == 2
    assert math.isclose(retry_outcome.next_state.get_resource("liquidity").current, 9510.0)

    # 3. Operational Safeguard (Section 11): Always-failing fallback caps at max_retry_depth = 1
    always_fail_calls: list[int] = []

    def always_failing_fallback(
        state: WorldState, actions: Sequence[Action]
    ) -> WorldState:
        always_fail_calls.append(len(always_fail_calls))
        raise RuntimeError("Permanent fallback failure")

    # Even if caller passes max_fallback_retries=100, router caps it at 1
    safeguard_router = HybridTransitionRouter(
        structural_equations=structural_eqs,
        ml_predictor=predictor,
        fallback_structural_fn=always_failing_fallback,
        max_fallback_retries=100,
    )
    assert safeguard_router.max_fallback_retries == 1
    safeguard_outcome = safeguard_router.step(ood_state)
    assert safeguard_outcome.abstained is True
    assert safeguard_outcome.fallback_executed is True
    assert safeguard_outcome.fallback_retry_depth == 1
    assert safeguard_outcome.fallback_failed is True
    # Strictly 2 calls total: initial attempt (0) + 1 retry (1) -> no infinite loop!
    assert len(always_fail_calls) == 2
    assert math.isclose(safeguard_outcome.next_state.get_resource("liquidity").current, 9505.0)


def test_legacy_default_infinity_thresholds_and_model_registry() -> None:
    """Verify thresholds default to infinity (disabled) and LearnedModelRegistry tracks metadata."""
    train_ds, cal_ds, _ = _generate_synthetic_transition_datasets()

    # 1. Default LinearResidualDynamics and Predictor have threshold == +inf
    lin_model = LinearResidualDynamics(
        target_resource="liquidity",
        action_type="rebalance",
    )
    assert math.isinf(lin_model.threshold)
    assert math.isinf(lin_model.ood_threshold)
    lin_model.fit(train_ds)

    # Even with extreme OOD state, default infinity thresholds do NOT abstain
    extreme_state = _make_state(liquidity=250_000.0, compute_budget=-150_000.0, step=10)
    pred_direct, ci_direct = lin_model.predict(
        extreme_state,
        [Action(id="act_ext", type="rebalance", parameters={"value": 500.0})],
    )
    assert isinstance(pred_direct, CalibratedPrediction)
    assert isinstance(ci_direct, ConfidenceInterval)
    assert ci_direct.ood_score > 10.0

    # When threshold is explicitly enabled on LinearResidualDynamics, it raises AbstainException
    lin_model.ood_threshold = 3.0
    with pytest.raises(AbstainException):
        lin_model.predict(extreme_state)

    # 2. Verify ensemble variance threshold abstention & router-level thresholds
    def high_var_member_1(state: WorldState) -> dict[str, float]:
        return {"liquidity": -20.0}

    def high_var_member_2(state: WorldState) -> dict[str, float]:
        return {"liquidity": 20.0}

    ens_pred = Predictor(
        models=[high_var_member_1, high_var_member_2],
        variance_threshold=100.0,  # Var([-20, 20]) = 400.0 > 100.0
    )
    with pytest.raises(AbstainException) as exc_info:
        ens_pred.predict(_make_state())
    assert exc_info.value.trigger == "ensemble_variance_threshold"
    assert math.isclose(exc_info.value.ensemble_variance, 400.0)

    # 3. Verify LearnedModelRegistry registration, fingerprinting, and lookup
    registry = LearnedModelRegistry()
    cal_wrapper = Predictor(models=lin_model)
    lin_model.ood_threshold = float("inf")
    cal_wrapper.calibrate(cal_ds, alpha=0.10)

    fp = registry.register(
        "treasury_liquidity_v1",
        cal_wrapper,
        metadata={"domain": "treasury", "owner": "ml_ops"},
    )
    assert len(fp) == 64
    assert registry.list_models() == ("treasury_liquidity_v1",)
    assert registry.get("treasury_liquidity_v1") is cal_wrapper
    meta = registry.metadata("treasury_liquidity_v1")
    assert meta["is_calibrated"] is True
    assert meta["threshold"] == "infinity"
    assert meta["fingerprint"] == fp

    with pytest.raises(KeyError):
        registry.get("nonexistent_model")
    with pytest.raises(KeyError):
        registry.metadata("nonexistent_model")
    with pytest.raises(ValueError):
        registry.register("   ", cal_wrapper)

    # 4. Verify OOD detector feature extraction modes, is_ood(), and error guards
    det = MahalanobisOODAndDriftDetector(
        feature_keys=("liquidity", "compute_budget"),
        ood_threshold=4.0,
        variance_threshold=10.0,
    )
    assert det.mahalanobis_distance(_make_state()) == 0.0
    assert det.drift_report().sample_count == 0

    det.fit(np.array([[100.0, 50.0], [101.0, 49.0], [99.0, 51.0]], dtype=np.float64))
    is_flag_ok, _, reason_ok = det.is_ood({"liquidity": 100.0, "compute_budget": 50.0})
    assert is_flag_ok is False
    assert reason_ok == "in_distribution"

    is_flag_ood, _, reason_ood = det.is_ood([500.0, -500.0])
    assert is_flag_ood is True
    assert "mahalanobis_distance" in reason_ood

    is_flag_var, _, reason_var = det.is_ood(
        {"liquidity": 100.0, "compute_budget": 50.0},
        ensemble_variance=25.0,
    )
    assert is_flag_var is True
    assert "ensemble_variance" in reason_var

    # Verify ConfidenceInterval > threshold abstention trigger
    ci_pred = Predictor(
        models=lambda s, a: {"liquidity": 1.0},
        threshold=0.05,
        default_half_width=2.0,
    )
    with pytest.raises(AbstainException) as ci_exc:
        ci_pred.predict(_make_state())
    assert ci_exc.value.trigger == "confidence_interval_threshold"

    # Verify router-level threshold and raise_on_abstain
    router_strict = HybridTransitionRouter(
        ml_predictor=lambda s, a, st: {"liquidity": 1.0},
        threshold=0.001,
        default_std=1.0,
        raise_on_abstain=True,
    )
    with pytest.raises(AbstainException):
        router_strict.step(_make_state())

    # 5. Cover edge cases and error guards in MahalanobisOODAndDriftDetector & ConformalPredictorWrapper
    det_states = MahalanobisOODAndDriftDetector()
    det_states.fit([_make_state(liquidity=100.0), _make_state(liquidity=102.0)])
    assert det_states.is_fitted is True
    assert det_states.drift_report().drift_detected is False
    assert det_states.extract_features(np.array([100.0, 50.0])).shape == (2,)

    det_1d = MahalanobisOODAndDriftDetector()
    det_1d.fit(np.array([1.0]))
    assert det_1d.is_fitted is True

    with pytest.raises(ValueError, match="dimension mismatch"):
        det_states.mahalanobis_distance([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="empty reference data"):
        MahalanobisOODAndDriftDetector().fit([])
    with pytest.raises(TypeError, match="Unsupported state input"):
        det_states.extract_features(object())  # type: ignore[arg-type]
    with pytest.raises(NumericalInstabilityError):
        det_states.extract_features([float("nan"), 1.0])

    assert compute_expected_calibration_error([], [])[0] == 0.0
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_expected_calibration_error([0.5, 0.6], [1.0])

    with pytest.raises(ValueError, match="at least one learned model"):
        Predictor(models=[])
    with pytest.raises(TypeError, match="Unsupported learned model member"):
        Predictor(models=[12345]).predict(_make_state())  # type: ignore[list-item]
    with pytest.raises(ValueError, match="empty dataset"):
        cal_wrapper.calibrate([])
    with pytest.raises(ValueError, match="Nominal coverage"):
        cal_wrapper.calibrate(cal_ds, alpha=1.5)
    with pytest.raises(ValueError, match="empty validation dataset"):
        cal_wrapper.evaluate_calibration([])
    assert "liquidity" in cal_wrapper.conformal_quantiles
