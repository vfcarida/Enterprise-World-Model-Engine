"""Integration tests for optional statistical and optimization backends.

Conforms to Track T4 (Extras):
- SALib: Sobol and Morris global sensitivity analysis.
- DistanceCalibrator: Method of Simulated Moments (MSM) parameter recovery.
- pycma & Nevergrad: Evolutionary black-box optimization.
- pymoo: NSGA-II multi-objective Pareto front.
- scikit-learn: GaussianProcessSurrogate emulator.
- Quarantined adapter stubs: SBINeuralInferenceAdapter, AxBayesianOptimizerAdapter.
"""

from __future__ import annotations

import numpy as np
import pytest

from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.experimentation.adapters import (
    AxBayesianOptimizerAdapter,
    SBINeuralInferenceAdapter,
)
from ewm_engine.experimentation.calibration import (
    CalibrationResult,
    DistanceCalibrator,
    compute_msm_moments,
    is_scipy_available,
)
from ewm_engine.experimentation.evolutionary import (
    NevergradOptimizer,
    PyCMAOptimizer,
    is_cma_available,
    is_nevergrad_available,
)
from ewm_engine.experimentation.params import (
    ParameterDef,
    ParameterSpace,
)
from ewm_engine.experimentation.pareto_opt import (
    PymooParetoOptimizer,
    is_pymoo_available,
)
from ewm_engine.experimentation.protocols import (
    ObjectiveVector,
    OptimizationResult,
)
from ewm_engine.experimentation.sensitivity import (
    GlobalSensitivityAnalyzer,
    MorrisResult,
    SobolResult,
    is_salib_available,
)
from ewm_engine.experimentation.surrogate_sklearn import (
    GaussianProcessSurrogate,
    is_sklearn_available,
)


def test_salib_sobol_and_morris_analysis() -> None:
    """Test Sobol and Morris sensitivity analysis using SALib."""
    if not is_salib_available():
        pytest.skip("SALib extra not installed")

    # Linear model: y = 10 * x1 + 1 * x2
    space = ParameterSpace(
        parameters=(
            ParameterDef(name="x1", type="continuous", bounds=(0.0, 1.0), default=0.5),
            ParameterDef(name="x2", type="continuous", bounds=(0.0, 1.0), default=0.5),
        )
    )

    def model_fn(pt: dict[str, float]) -> float:
        return 10.0 * pt["x1"] + 1.0 * pt["x2"]

    analyzer = GlobalSensitivityAnalyzer(space)

    # Sobol analysis with second order indices
    sobol: SobolResult = analyzer.analyze_sobol(
        model_fn, n_samples=64, calc_second_order=True, seed=42
    )
    assert "x1" in sobol.s1
    assert "x2" in sobol.s1
    # x1 has 10x the weight of x2, so S1(x1) must be substantially larger than S1(x2)
    assert sobol.s1["x1"] > sobol.s1["x2"]
    assert sobol.st["x1"] > sobol.st["x2"]
    assert sobol.s2 is not None
    assert "x1" in sobol.s2
    assert "x2" in sobol.s2["x1"]

    # Morris analysis
    morris: MorrisResult = analyzer.analyze_morris(model_fn, n_trajectories=10, seed=42)
    assert "x1" in morris.mu_star
    assert "x2" in morris.mu_star
    assert morris.mu_star["x1"] > morris.mu_star["x2"]

    # Space with no numeric parameters raises ValueError
    cat_space = ParameterSpace(
        parameters=(
            ParameterDef(name="cat", type="categorical", categories=("a", "b"), default="a"),
        )
    )
    with pytest.raises(ValueError, match="at least one numeric parameter"):
        GlobalSensitivityAnalyzer(cat_space)


def test_distance_calibrator_recovers_known_parameters() -> None:
    """Test DistanceCalibrator recovering known synthetic parameter target via MSM loss."""
    if not is_scipy_available():
        pytest.skip("scipy extra not installed")

    # True parameter: shift = 25.0
    true_shift = 25.0

    space = ParameterSpace(
        parameters=(
            ParameterDef(name="shift", type="continuous", bounds=(0.0, 50.0), default=10.0),
        )
    )

    def simulator(pt: dict[str, float]) -> list[float]:
        # Return samples centered at pt['shift']
        return [pt["shift"] - 1.0, pt["shift"], pt["shift"] + 1.0]

    # Target moments from true parameter
    target_data = [true_shift - 1.0, true_shift, true_shift + 1.0]
    target_moments = compute_msm_moments(target_data)

    calibrator = DistanceCalibrator(
        space=space,
        simulator_fn=simulator,
        target_moments=target_moments,
    )

    result: CalibrationResult = calibrator.calibrate(max_evaluations=40, method="Nelder-Mead")

    recovered_shift = result.calibrated_parameters["shift"]
    assert recovered_shift == pytest.approx(true_shift, abs=0.5)
    assert result.loss_value < 1e-2


def test_pycma_optimizer_execution() -> None:
    """Test CMA-ES optimizer minimizing sphere objective."""
    if not is_cma_available():
        pytest.skip("pycma extra not installed")

    space = ParameterSpace(
        parameters=(
            ParameterDef(name="p1", type="continuous", bounds=(-2.0, 2.0), default=1.5),
            ParameterDef(name="p2", type="continuous", bounds=(-2.0, 2.0), default=1.5),
        )
    )

    def obj_fn(pt: dict[str, float]) -> ObjectiveVector:
        # Minimum at (0, 0)
        val = pt["p1"] ** 2 + pt["p2"] ** 2
        return ObjectiveVector(values={"loss": val}, directions={"loss": "minimize"})

    opt = PyCMAOptimizer(sigma0=0.3)
    res: OptimizationResult = opt.optimize(obj_fn, space, n_evaluations=30, seed=42)

    assert res.evaluations_count > 0
    # Initial score at (1.5, 1.5) was 4.5; CMA should find a point closer to 0
    assert res.best_objective.values["loss"] < 4.5


def test_nevergrad_optimizer_execution() -> None:
    """Test Nevergrad optimizer minimizing quadratic objective."""
    if not is_nevergrad_available():
        pytest.skip("nevergrad extra not installed")

    space = ParameterSpace(
        parameters=(ParameterDef(name="a", type="continuous", bounds=(0.0, 10.0), default=8.0),)
    )

    def obj_fn(pt: dict[str, float]) -> ObjectiveVector:
        # Minimum at a = 3.0
        val = (pt["a"] - 3.0) ** 2
        return ObjectiveVector(values={"loss": val}, directions={"loss": "minimize"})

    opt = NevergradOptimizer(algorithm="NGOpt")
    res: OptimizationResult = opt.optimize(obj_fn, space, n_evaluations=25, seed=42)

    assert res.evaluations_count == 25
    assert res.best_objective.values["loss"] < 25.0


def test_pymoo_pareto_optimizer_execution() -> None:
    """Test pymoo NSGA-II multi-objective optimization returning a ParetoFront."""
    if not is_pymoo_available():
        pytest.skip("pymoo extra not installed")

    space = ParameterSpace(
        parameters=(ParameterDef(name="x", type="continuous", bounds=(0.0, 2.0), default=1.0),)
    )

    # Conflicting objectives: f1 = x^2, f2 = (x - 2)^2
    def obj_fn(pt: dict[str, float]) -> ObjectiveVector:
        x = pt["x"]
        return ObjectiveVector(
            values={"f1": x**2, "f2": (x - 2.0) ** 2},
            directions={"f1": "minimize", "f2": "minimize"},
        )

    opt = PymooParetoOptimizer(population_size=10, n_generations=5)
    res: OptimizationResult = opt.optimize(obj_fn, space, seed=42)

    assert res.pareto_front is not None
    assert res.pareto_front.size >= 1
    for policy in res.pareto_front.policies:
        assert 0.0 <= policy.parameters["x"] <= 2.0
        assert "f1" in policy.objectives.values
        assert "f2" in policy.objectives.values


def test_gaussian_process_surrogate_execution() -> None:
    """Test GaussianProcessSurrogate fitting and predicting on synthetic data."""
    if not is_sklearn_available():
        pytest.skip("scikit-learn extra not installed")

    X_train = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y_train = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float64)

    surrogate = GaussianProcessSurrogate(random_state=42)
    surrogate.fit(X_train, y_train)

    X_test = np.array([[0.0], [2.0]], dtype=np.float64)
    mean, std = surrogate.predict(X_test)

    assert len(mean) == 2
    assert len(std) == 2
    # At training points, prediction should match closely with low uncertainty
    assert mean[0] == pytest.approx(0.0, abs=0.2)
    assert mean[1] == pytest.approx(4.0, abs=0.2)
    assert std[0] < 0.1


def test_quarantined_adapter_stubs() -> None:
    """Test heavy quarantined adapter stubs without external runtime coupling."""
    with pytest.raises(SimulationConfigurationError, match="sbi and torch are required"):
        SBINeuralInferenceAdapter()

    with pytest.raises(SimulationConfigurationError, match="ax-platform and botorch are required"):
        AxBayesianOptimizerAdapter()
