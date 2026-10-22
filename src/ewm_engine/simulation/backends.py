"""Array API standard abstraction and cross-backend execution backends.

Provides backend-agnostic tensor operations conforming to the Python Array API standard
(2025.12) with an explicit three-tier numerical reproducibility contract:
- BITWISE_REPRODUCIBLE (NumPy / Pure-Python under a pinned runtime: identical Python,
  NumPy, IEEE-754 float64 dtype, single-threaded BLAS `OMP_NUM_THREADS=1`, and
  SeedSequence tree). Does NOT claim universal bit-identity across arbitrary hardware,
  compilers, ISA extensions, or multi-threaded BLAS reductions.
- NUMERICALLY_EQUIVALENT_WITH_TOLERANCE (JAX / compiled XLA backends): Threefry counter
  PRNG provides deterministic random streams, while XLA fusion, FMA selection, and
  device execution agree with the reference kernel within an explicit `(rtol, atol)`
  tolerance contract.
- NON_CERTIFIED_BACKEND (CuPy / raw CUDA kernels): Hardware float atomic reductions may
  be non-deterministic across runs; suitable for throughput exploration only and refused
  from certified audit paths when `certify_equivalence=True`.
"""

from __future__ import annotations

import importlib.util
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ewm_engine.exceptions import SimulationConfigurationError


class BackendType(StrEnum):
    """Identifier for array execution backends."""

    PURE_PYTHON = "pure_python"
    NUMPY = "numpy"
    JAX = "jax"
    CUPY = "cupy"


class CertificationStatus(StrEnum):
    """Three-tier numerical reproducibility certification contract for execution backends."""

    BITWISE_REPRODUCIBLE = "bitwise_reproducible"
    NUMERICALLY_EQUIVALENT_WITH_TOLERANCE = "numerically_equivalent_with_tolerance"
    NON_CERTIFIED_BACKEND = "non_certified_backend"

    # Backwards-compatible aliases
    CERTIFIED_BIT_IDENTICAL = "bitwise_reproducible"
    THROUGHPUT_ONLY = "non_certified_backend"


@runtime_checkable
class ArrayBackend(Protocol):
    """Protocol defining the standard Array API surface for simulation rollouts."""

    @property
    def backend_type(self) -> BackendType:
        """Type of this backend."""
        ...

    @property
    def certification(self) -> CertificationStatus:
        """Three-tier numerical reproducibility certification level."""
        ...

    @property
    def certification_status(self) -> CertificationStatus:
        """Alias for certification."""
        ...

    @property
    def tolerance_contract(self) -> dict[str, float] | None:
        """Explicit (rtol, atol) numerical tolerance contract, or None if uncertified."""
        ...

    @property
    def is_available(self) -> bool:
        """Whether required third-party libraries for this backend are installed."""
        ...

    def asarray(self, obj: Any, dtype: Any = None) -> Any:
        """Convert input object to a backend array."""
        ...

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        """Create an array of zeros with the specified shape."""
        ...

    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        """Create an array of ones with the specified shape."""
        ...

    def clip(self, x: Any, min_val: float | None = None, max_val: float | None = None) -> Any:
        """Clip array elements to lie within [min_val, max_val]."""
        ...

    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication of two arrays."""
        ...

    def to_numpy(self, x: Any) -> np.ndarray:
        """Convert backend array to a standard NumPy ndarray."""
        ...


class NumPyBackend:
    """Default standard backend using NumPy (Array API compliant, zero extra deps).

    Certified as BITWISE_REPRODUCIBLE under a pinned runtime (same Python/NumPy
    version, IEEE-754 float64, OMP_NUM_THREADS=1, and canonical SeedSequence).
    """

    def __init__(self) -> None:
        self._type = BackendType.NUMPY
        self._cert = CertificationStatus.BITWISE_REPRODUCIBLE
        self._tolerance: dict[str, float] = {"rtol": 0.0, "atol": 0.0}

    @property
    def backend_type(self) -> BackendType:
        return self._type

    @property
    def certification(self) -> CertificationStatus:
        return self._cert

    @property
    def certification_status(self) -> CertificationStatus:
        return self._cert

    @property
    def tolerance_contract(self) -> dict[str, float] | None:
        return dict(self._tolerance)

    @property
    def is_available(self) -> bool:
        return True

    def asarray(self, obj: Any, dtype: Any = None) -> np.ndarray:
        return np.asarray(obj, dtype=dtype if dtype is not None else np.float64)

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> np.ndarray:
        return np.zeros(shape, dtype=dtype if dtype is not None else np.float64)

    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> np.ndarray:
        return np.ones(shape, dtype=dtype if dtype is not None else np.float64)

    def clip(
        self, x: Any, min_val: float | None = None, max_val: float | None = None
    ) -> np.ndarray:
        arr = np.asarray(x)
        return np.clip(arr, a_min=min_val, a_max=max_val)

    def matmul(self, a: Any, b: Any) -> np.ndarray:
        return np.asarray(np.matmul(np.asarray(a), np.asarray(b)))

    def to_numpy(self, x: Any) -> np.ndarray:
        return np.asarray(x)


class JaxBackend:
    """Optional JAX backend leveraging JIT compilation and Threefry counter-based PRNG.

    Classified as NUMERICALLY_EQUIVALENT_WITH_TOLERANCE: while Threefry-2x32 PRNG is
    deterministic across devices, XLA compiler fusion, FMA instruction selection, and
    device-specific reductions agree with the reference kernel within explicit
    floating-point tolerances (`rtol=1e-6`, `atol=1e-7`) rather than universal
    cross-hardware bit-identity.
    """

    def __init__(
        self,
        platform: str | None = None,
        rtol: float = 1e-6,
        atol: float = 1e-7,
    ) -> None:
        self._type = BackendType.JAX
        self._platform = platform
        self._cert = CertificationStatus.NUMERICALLY_EQUIVALENT_WITH_TOLERANCE
        self._tolerance: dict[str, float] = {"rtol": float(rtol), "atol": float(atol)}

    @property
    def backend_type(self) -> BackendType:
        return self._type

    @property
    def certification(self) -> CertificationStatus:
        return self._cert

    @property
    def certification_status(self) -> CertificationStatus:
        return self._cert

    @property
    def tolerance_contract(self) -> dict[str, float] | None:
        return dict(self._tolerance)

    @property
    def is_available(self) -> bool:
        return importlib.util.find_spec("jax") is not None

    def _get_jnp(self) -> Any:
        if not self.is_available:
            raise SimulationConfigurationError(
                "JAX backend requested but 'jax' is not installed. "
                "Install via `pip install ewm-engine[jax]`."
            )
        import jax
        import jax.numpy as jnp

        # Enable float64 precision when JAX is used for scientific rollouts
        jax.config.update("jax_enable_x64", True)
        return jnp

    def asarray(self, obj: Any, dtype: Any = None) -> Any:
        jnp = self._get_jnp()
        return jnp.asarray(obj, dtype=dtype if dtype is not None else jnp.float64)

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        jnp = self._get_jnp()
        return jnp.zeros(shape, dtype=dtype if dtype is not None else jnp.float64)

    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        jnp = self._get_jnp()
        return jnp.ones(shape, dtype=dtype if dtype is not None else jnp.float64)

    def clip(self, x: Any, min_val: float | None = None, max_val: float | None = None) -> Any:
        jnp = self._get_jnp()
        return jnp.clip(x, a_min=min_val, a_max=max_val)

    def matmul(self, a: Any, b: Any) -> Any:
        jnp = self._get_jnp()
        return jnp.matmul(a, b)

    def to_numpy(self, x: Any) -> np.ndarray:
        return np.asarray(x)


class CuPyBackend:
    """Optional CuPy GPU backend for massive CUDA Monte Carlo ensembles.

    Classified as NON_CERTIFIED_BACKEND because raw GPU atomic floating-point
    reductions are not bitwise or tolerance-certified across arbitrary thread schedules.
    """

    def __init__(self) -> None:
        self._type = BackendType.CUPY
        self._cert = CertificationStatus.NON_CERTIFIED_BACKEND

    @property
    def backend_type(self) -> BackendType:
        return self._type

    @property
    def certification(self) -> CertificationStatus:
        return self._cert

    @property
    def certification_status(self) -> CertificationStatus:
        return self._cert

    @property
    def tolerance_contract(self) -> dict[str, float] | None:
        return None

    @property
    def is_available(self) -> bool:
        return importlib.util.find_spec("cupy") is not None

    def _get_cp(self) -> Any:
        if not self.is_available:
            raise SimulationConfigurationError(
                "CuPy backend requested but 'cupy' is not installed. "
                "Install via `pip install ewm-engine[cuda]`."
            )
        import cupy as cp

        return cp

    def asarray(self, obj: Any, dtype: Any = None) -> Any:
        cp = self._get_cp()
        return cp.asarray(obj, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        cp = self._get_cp()
        return cp.zeros(shape, dtype=dtype)

    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        cp = self._get_cp()
        return cp.ones(shape, dtype=dtype)

    def clip(self, x: Any, min_val: float | None = None, max_val: float | None = None) -> Any:
        cp = self._get_cp()
        return cp.clip(x, a_min=min_val, a_max=max_val)

    def matmul(self, a: Any, b: Any) -> Any:
        cp = self._get_cp()
        return cp.matmul(a, b)

    def to_numpy(self, x: Any) -> np.ndarray:
        if hasattr(x, "get"):
            return np.asarray(x.get())
        return np.asarray(x)


def resolve_backend(backend: str | BackendType | ArrayBackend | None) -> ArrayBackend:
    """Resolve backend specifier to an initialized ArrayBackend instance."""
    if backend is None or backend in (BackendType.NUMPY, "numpy"):
        return NumPyBackend()
    if backend in (BackendType.JAX, "jax"):
        b_jax = JaxBackend()
        if not b_jax.is_available:
            raise SimulationConfigurationError(
                "The 'jax' backend is not installed in the current environment. "
                "Install with `pip install ewm-engine[jax]`."
            )
        return b_jax
    if backend in (BackendType.CUPY, "cupy", "cuda"):
        b_cp = CuPyBackend()
        if not b_cp.is_available:
            raise SimulationConfigurationError(
                "The 'cupy' backend is not installed in the current environment. "
                "Install with `pip install ewm-engine[cuda]`."
            )
        return b_cp
    if isinstance(backend, ArrayBackend):
        return backend

    raise SimulationConfigurationError(
        f"Unknown simulation backend: {backend!r}. "
        f"Expected one of 'numpy', 'jax', 'cupy', or an ArrayBackend instance."
    )
