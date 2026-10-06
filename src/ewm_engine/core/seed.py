"""Global and environment-level pseudo-random seed configuration."""

from __future__ import annotations

import os
import random
import sys
import warnings
from typing import Any

import numpy as np


def seed_everything(
    seed: int,
    deterministic: bool = True,
    warn_cuda: bool = True,
) -> dict[str, Any]:
    """Seed Python standard library, NumPy, and optional ML frameworks (PyTorch).

    Implements the NeurIPS reproducibility checklist standards and AC-004 contract.

    Args:
        seed: Master integer seed.
        deterministic: If True, sets strict deterministic flags across runtime environments.
        warn_cuda: If True, warns when CUBLAS_WORKSPACE_CONFIG is not properly configured for CUDA.

    Returns:
        Dictionary detailing active determinism flags and seed status.
    """
    # 1. Python standard library random & hash seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. NumPy global random state (Generator should still be instantiated locally)
    np.random.seed(seed)

    # 3. Thread pinning for BLAS reduction determinism
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    flags: dict[str, Any] = {
        "master_seed": seed,
        "python_hash_seed": str(seed),
        "numpy_seeded": True,
        "blas_pinned": True,
        "torch_available": False,
        "cuda_deterministic": False,
    }

    # 4. Optional PyTorch deterministic configuration (lazy check)
    if "torch" in sys.modules:
        _configure_torch(seed, deterministic, warn_cuda, flags)
    else:
        try:
            import torch  # noqa: F401

            _configure_torch(seed, deterministic, warn_cuda, flags)
        except ImportError:
            pass

    return flags


def _configure_torch(
    seed: int,
    deterministic: bool,
    warn_cuda: bool,
    flags: dict[str, Any],
) -> None:
    """Configure PyTorch seed and algorithmic determinism."""
    import torch

    flags["torch_available"] = True
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
            flags["torch_deterministic_algorithms"] = True
        except Exception as exc:  # pragma: no cover
            flags["torch_deterministic_algorithms"] = False
            flags["torch_deterministic_error"] = str(exc)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            flags["cuda_deterministic"] = True

            cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
            if cublas_config not in (":4096:8", ":16:8"):
                flags["cublas_warning"] = True
                if warn_cuda:
                    warnings.warn(
                        "CUBLAS_WORKSPACE_CONFIG is not set to ':4096:8' or ':16:8'. "
                        "CUDA atomic operations and convolutions may fail or be non-deterministic.",
                        UserWarning,
                        stacklevel=3,
                    )
