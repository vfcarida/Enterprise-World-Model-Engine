"""Tests for runtime validation discipline, boundary checks, and beartype integration."""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype
from beartype.roar import BeartypeCallHintViolation

from ewm_engine.core.boundary import validate_numeric_array_boundary


@pytest.mark.packaging
def test_validate_numeric_array_boundary_success() -> None:
    """Valid floating and integer arrays pass boundary validation efficiently."""
    arr1d = np.array([1.0, 2.5, 3.8], dtype=np.float64)
    res = validate_numeric_array_boundary(arr1d, expected_ndim=1)
    assert res is arr1d

    arr2d = np.array([[1, 2], [3, 4]], dtype=np.int64)
    res2 = validate_numeric_array_boundary(arr2d, expected_ndim=2)
    assert res2 is arr2d


@pytest.mark.packaging
def test_validate_numeric_array_boundary_failures() -> None:
    """Non-arrays, dimension mismatches, NaNs, and Infs raise clean boundary errors."""
    # 1. Non-array input
    with pytest.raises(TypeError, match=r"must be a numpy\.ndarray"):
        validate_numeric_array_boundary([1.0, 2.0])

    # 2. Dimension mismatch
    arr1d = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="must have ndim=2"):
        validate_numeric_array_boundary(arr1d, expected_ndim=2)

    # 3. NaN rejection
    nan_arr = np.array([1.0, np.nan, 3.0])
    with pytest.raises(ValueError, match="contains NaN values"):
        validate_numeric_array_boundary(nan_arr, allow_nan=False)

    # 4. Inf rejection
    inf_arr = np.array([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="contains infinite values"):
        validate_numeric_array_boundary(inf_arr, allow_inf=False)

    # 5. Non-numeric dtype
    str_arr = np.array(["a", "b"])
    with pytest.raises(TypeError, match="must have numeric dtype"):
        validate_numeric_array_boundary(str_arr)


@pytest.mark.packaging
def test_beartype_decoration_on_boundary_functions() -> None:
    """Beartype successfully decorates public boundary functions without array element loops."""

    @beartype
    def boundary_endpoint(state_id: str, values: np.ndarray, tolerance: float = 1e-4) -> float:
        validate_numeric_array_boundary(values, expected_ndim=1)
        return float(np.sum(values))

    # Valid call
    total = boundary_endpoint("state_a", np.array([10.0, 20.0, 30.0]))
    assert total == 60.0

    # Type error on primitive parameter caught by beartype at O(1)
    with pytest.raises(BeartypeCallHintViolation):
        boundary_endpoint(12345, np.array([10.0, 20.0]))  # type: ignore[arg-type]
