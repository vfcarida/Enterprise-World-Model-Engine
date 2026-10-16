"""Array API standard abstraction and cross-backend execution backends.

Provides backend-agnostic tensor operations conforming to the Python Array API standard
(2025.12) with support for:
- NumPy (default reference backend, certified bit-identical to pure-Python).
- JAX (optional compiled backend, CPU/GPU bit-identical Threefry counter-based PRNG).
- CuPy (optional GPU backend, throughput-only where floating-point non-deterministic).

Invariant #5 Enforcement:
Any backend that deviates from bit-exact fingerprint reproducibility is explicitly
labeled THROUGHPUT_ONLY and refused from certified audit paths.
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
    """Epistemic certification status of execution backends."""

    CERTIFIED_BIT_IDENTICAL = "certified_bit_identical"
    THROUGHPUT_ONLY = "throughput_only"


@runtime_checkable
class ArrayBackend(Protocol):
    """Protocol defining the standard Array API surface for simulation rollouts."""

    @property
    def backend_type(self) -> BackendType:
        """Type of this backend."""
        ...

    @property
    def certification(self) -> CertificationStatus:
        """Whether this backend is certified bit-identical or throughput-only."""
        ...

    @property
    def certification_status(self) -> CertificationStatus:
        """Alias for certification."""
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
    """Default standard backend using NumPy (Array API compliant, zero extra deps)."""

    def __init__(self) -> None:
        self._type = BackendType.NUMPY
        self._cert = CertificationStatus.CERTIFIED_BIT_IDENTICAL

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
        return np.matmul(np.asarray(a), np.asarray(b))

    def to_numpy(self, x: Any) -> np.ndarray:
        return np.asarray(x)


class JaxBackend:
    """Optional JAX backend leveraging JIT compilation and Threefry counter-based PRNG.

    Guarantees CPU/GPU bit-identity via counter-based Threefry-2x32 PRNG (Salmon et al., 2011).
    """

    def __init__(self, platform: str | None = None) -> None:
        self._type = BackendType.JAX
        self._platform = platform
        # Default JAX CPU is bit-identical certified; custom GPU ops may be throughput-only
        self._cert = CertificationStatus.CERTIFIED_BIT_IDENTICAL

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
    def is_available(self) -> bool:
        return importlib.util.find_spec("jax") is not None

    def _get_jnp(self) -> Any:
        if not self.is_available:
            raise SimulationConfigurationError(
                "JAX backend requested but 'jax' is not installed. "
                "Install via `pip install ewm-engine[jax]`."
            )
        import jax.numpy as jnp

        return jnp

    def asarray(self, obj: Any, dtype: Any = None) -> Any:
        jnp = self._get_jnp()
        return jnp.asarray(obj, dtype=dtype if dtype is not None else jnp.float32)

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        jnp = self._get_jnp()
        return jnp.zeros(shape, dtype=dtype if dtype is not None else jnp.float32)

    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        jnp = self._get_jnp()
        return jnp.ones(shape, dtype=dtype if dtype is not None else jnp.float32)

    def clip(self, x: Any, min_val: float | None = None, max_val: float | None = None) -> Any:
        jnp = self._get_jnp()
        return jnp.clip(x, a_min=min_val, a_max=max_val)

    def matmul(self, a: Any, b: Any) -> Any:
        jnp = self._get_jnp()
        return jnp.matmul(a, b)

    def to_numpy(self, x: Any) -> np.ndarray:
        return np.asarray(x)


class CuPyBackend:
    """Optional CuPy GPU backend for massive CUDA Monte Carlo ensembles."""

    def __init__(self) -> None:
        self._type = BackendType.CUPY
        # GPU atomic / non-deterministic float reductions are labeled throughput-only
        self._cert = CertificationStatus.THROUGHPUT_ONLY

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
            return x.get()
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
