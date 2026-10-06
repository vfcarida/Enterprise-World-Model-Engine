"""Static image export utilities via Kaleido (VIZ-EXPORT extra).

Quarantined behind the [viz-export] extra (Kaleido).
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

from ewm_engine.exceptions import SimulationConfigurationError


def is_kaleido_available() -> bool:
    """Check if Kaleido static image export library is installed."""
    return importlib.util.find_spec("kaleido") is not None


def export_static_image(
    fig: Any,
    output_path: str | Path,
    img_format: str = "png",
    width: int = 1200,
    height: int = 800,
) -> None:
    """Export Plotly figure to static image (PNG, SVG, PDF) via Kaleido."""
    if not is_kaleido_available():
        raise SimulationConfigurationError(
            "Kaleido is required for static image export. Install via `pip install ewm-engine[viz-export]`."
        )

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(p), format=img_format, width=width, height=height)
