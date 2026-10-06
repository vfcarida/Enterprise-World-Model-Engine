"""OmegaConf configuration loader with variable interpolation (CONFIG extra).

Quarantined behind the [config] extra (OmegaConf).
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.exceptions import SimulationConfigurationError


def is_omegaconf_available() -> bool:
    """Check if OmegaConf configuration package is installed."""
    return importlib.util.find_spec("omegaconf") is not None


class OmegaConfConfigLoader:
    """Configuration loader supporting `${...}` interpolation and canonical hashing."""

    def __init__(self) -> None:
        if not is_omegaconf_available():
            raise SimulationConfigurationError(
                "OmegaConf is required for OmegaConfConfigLoader. Install via `pip install ewm-engine[config]`."
            )

    def load_config_file(self, config_path: str | Path) -> dict[str, Any]:
        """Load and interpolate YAML/JSON config file, returning a resolved dict."""
        omega = importlib.import_module("omegaconf")
        p = Path(config_path)
        cfg = omega.OmegaConf.load(str(p))
        resolved = omega.OmegaConf.to_container(cfg, resolve=True)
        return dict(resolved) if isinstance(resolved, dict) else {"config": resolved}

    def load_config_dict(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Create and interpolate configuration from a dictionary."""
        omega = importlib.import_module("omegaconf")
        cfg = omega.OmegaConf.create(config_dict)
        resolved = omega.OmegaConf.to_container(cfg, resolve=True)
        return dict(resolved) if isinstance(resolved, dict) else {"config": resolved}

    def compute_config_hash(self, config: dict[str, Any]) -> str:
        """Compute canonical SHA-256 fingerprint of the resolved configuration."""
        return canonical_sha256(config)
