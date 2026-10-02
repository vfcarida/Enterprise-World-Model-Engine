"""Security tests verifying rejection of non-standard, insecure floating-point constants."""

from __future__ import annotations

import pytest

from ewm_engine.exceptions import SerializationSecurityError
from ewm_engine.serialization.json import canonical_dumps, canonical_loads


@pytest.mark.security
def test_canonical_loads_rejects_nan() -> None:
    """AC-018 & security: canonical_loads must reject JSON NaN."""
    with pytest.raises(SerializationSecurityError, match="rejects non-standard constant 'NaN'"):
        canonical_loads('{"metric": NaN}')


@pytest.mark.security
def test_canonical_loads_rejects_infinity() -> None:
    """AC-018 & security: canonical_loads must reject JSON Infinity."""
    with pytest.raises(
        SerializationSecurityError, match="rejects non-standard constant 'Infinity'"
    ):
        canonical_loads('{"capacity": Infinity}')

    with pytest.raises(
        SerializationSecurityError, match="rejects non-standard constant '-Infinity'"
    ):
        canonical_loads('{"temperature": -Infinity}')


@pytest.mark.security
def test_canonical_dumps_rejects_nan_and_infinity() -> None:
    """AC-018 & security: canonical_dumps must reject non-finite float values."""
    with pytest.raises(SerializationSecurityError, match="rejects non-finite float value"):
        canonical_dumps({"value": float("nan")})

    with pytest.raises(SerializationSecurityError, match="rejects non-finite float value"):
        canonical_dumps({"value": float("inf")})

    with pytest.raises(SerializationSecurityError, match="rejects non-finite float value"):
        canonical_dumps({"value": float("-inf")})


@pytest.mark.security
def test_canonical_dumps_rejects_nested_nan_in_containers() -> None:
    """Security: deep search identifies NaN/Infinity even when nested in sequences."""
    nested_data = {
        "metrics": [1.0, 2.0, [3.0, float("nan")]],
    }
    with pytest.raises(SerializationSecurityError, match="rejects non-finite float value"):
        canonical_dumps(nested_data)
