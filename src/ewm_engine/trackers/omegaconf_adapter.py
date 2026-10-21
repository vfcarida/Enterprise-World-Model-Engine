"""OmegaConf configuration loader with variable interpolation (CONFIG extra).

Quarantined behind the [config] extra (OmegaConf).
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import re
from pathlib import Path
from typing import Any

import yaml

from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.exceptions import SimulationConfigurationError

_INTERP_FULL_RE = re.compile(r"^\$\{([a-zA-Z0-9_.-]+)\}$")
_INTERP_PARTIAL_RE = re.compile(r"\$\{([a-zA-Z0-9_.-]+)\}")


def is_omegaconf_available() -> bool:
    """Check if OmegaConf configuration package is installed."""
    return importlib.util.find_spec("omegaconf") is not None


class OmegaConfConfigLoader:
    """Configuration loader supporting `${...}` interpolation and canonical hashing."""

    def __init__(self, require_extra: bool = False) -> None:
        if require_extra and not is_omegaconf_available():
            raise SimulationConfigurationError(
                "OmegaConf is required for OmegaConfConfigLoader. Install via `pip install ewm-engine[config]`."
            )

    def load_config_file(self, config_path: str | Path) -> dict[str, Any]:
        """Load and interpolate YAML/JSON config file, returning a resolved dict."""
        p = Path(config_path)
        if is_omegaconf_available():
            omega = importlib.import_module("omegaconf")
            cfg = omega.OmegaConf.load(str(p))
            resolved = omega.OmegaConf.to_container(cfg, resolve=True)
            return dict(resolved) if isinstance(resolved, dict) else {"config": resolved}

        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        data = dict(raw) if isinstance(raw, dict) else {"config": raw}
        return self._resolve_dict(data)

    def load_config_dict(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        """Create and interpolate configuration from a dictionary."""
        if is_omegaconf_available():
            omega = importlib.import_module("omegaconf")
            cfg = omega.OmegaConf.create(config_dict)
            resolved = omega.OmegaConf.to_container(cfg, resolve=True)
            return dict(resolved) if isinstance(resolved, dict) else {"config": resolved}

        return self._resolve_dict(config_dict)

    def _lookup_path(self, root: dict[str, Any], dotted_path: str) -> Any:
        curr: Any = root
        for part in dotted_path.split("."):
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                raise KeyError(f"Interpolation key not found: {dotted_path}")
        return curr

    def _resolve_node(self, node: Any, root: dict[str, Any]) -> Any:
        if isinstance(node, dict):
            return {k: self._resolve_node(v, root) for k, v in node.items()}
        if isinstance(node, list):
            return [self._resolve_node(item, root) for item in node]
        if isinstance(node, str):
            m_full = _INTERP_FULL_RE.match(node)
            if m_full is not None:
                return self._lookup_path(root, m_full.group(1))
            if "${" in node:
                return _INTERP_PARTIAL_RE.sub(
                    lambda m: str(self._lookup_path(root, m.group(1))), node
                )
        return node

    def _resolve_dict(self, config_dict: dict[str, Any]) -> dict[str, Any]:
        resolved: dict[str, Any] = copy.deepcopy(config_dict)
        for _ in range(5):
            resolved = self._resolve_node(resolved, resolved)
        return resolved

    def compute_config_hash(self, config: dict[str, Any]) -> str:
        """Compute canonical SHA-256 fingerprint of the resolved configuration."""
        return canonical_sha256(config)
