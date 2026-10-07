"""Runtime validation discipline for numerical hot paths and public boundaries.

Implements boundary-only array validation (type, ndim, finiteness) in a single vectorized
pass, eliminating per-element validation overhead in simulation rollout hot paths.
"""

from __future__ import annotations

import numpy as np


def validate_numeric_array_boundary(
    arr: object,
    expected_ndim: int | None = None,
    allow_nan: bool = False,
    allow_inf: bool = False,
    dtype_kind: str = "fiub",  # float (f), signed int (i), unsigned int (u), bool (b)
    param_name: str = "array",
) -> np.ndarray:
    """Validate a numeric array at the system boundary in a single vectorized pass.

    Never perform per-element or Pydantic validation inside numeric simulation hot paths.
    Validate once at the boundary (type, ndim, finiteness), then trust internally.

    Args:
        arr: Array candidate to validate.
        expected_ndim: Optional expected dimension count (e.g., 1 for 1D, 2 for 2D).
        allow_nan: If False, raises ValueError if any NaN values exist.
        allow_inf: If False, raises ValueError if any infinite values exist.
        dtype_kind: Allowed numpy dtype kind characters (default: float, int, uint, bool).
        param_name: Parameter name for informative error messages.

    Returns:
        Validated numpy.ndarray instance.

    Raises:
        TypeError: If input is not a numpy array or has invalid dtype.
        ValueError: If ndim doesn't match or invalid NaN/inf values are detected.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"'{param_name}' must be a numpy.ndarray, got {type(arr).__name__}")

    if expected_ndim is not None and arr.ndim != expected_ndim:
        raise ValueError(
            f"'{param_name}' must have ndim={expected_ndim}, got ndim={arr.ndim} (shape={arr.shape})"
        )

    if arr.dtype.kind not in dtype_kind:
        raise TypeError(
            f"'{param_name}' must have numeric dtype (kinds: '{dtype_kind}'), got dtype={arr.dtype}"
        )

    if not allow_nan and bool(np.isnan(arr).any()):
        raise ValueError(f"'{param_name}' contains NaN values at the system boundary")

    if not allow_inf and bool(np.isinf(arr).any()):
        raise ValueError(f"'{param_name}' contains infinite values at the system boundary")

    return arr
