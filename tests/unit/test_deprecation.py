"""Unit tests for the deprecation workflow and NEP-23 warning compliance."""

from __future__ import annotations

from typing import Any, cast

import pytest

from ewm_engine.core.deprecation import deprecate, deprecated


def test_deprecate_function_emits_deprecation_warning() -> None:
    """deprecate() emits a standard DeprecationWarning with metadata."""
    with pytest.warns(
        DeprecationWarning,
        match=r"'old_feature' was deprecated in v1\.5\.0 and is scheduled for removal in v1\.7\.0\. Use new_feature\(\) instead\.",
    ):
        deprecate(
            feature="old_feature",
            since="1.5.0",
            removed_in="1.7.0",
            alternative="Use new_feature() instead.",
        )


def test_deprecated_decorator_emits_warning_and_preserves_execution() -> None:
    """@deprecated decorates callables, emits warnings on invocation, and forwards results."""

    @deprecated(since="1.5.0", removed_in="1.7.0", alternative="Use compute_v2() instead.")
    def sample_func(x: int, y: int = 10) -> int:
        """Sample calculation docstring."""
        return x + y

    # Verify metadata attributes
    any_func = cast(Any, sample_func)
    assert any_func.__deprecated__ is True
    assert any_func.__deprecated_since__ == "1.5.0"
    assert any_func.__deprecated_removed_in__ == "1.7.0"
    assert any_func.__deprecated_alternative__ == "Use compute_v2() instead."

    # Verify docstring update with .. deprecated:: directive
    assert sample_func.__doc__ is not None
    assert ".. deprecated:: 1.5.0" in sample_func.__doc__
    assert "Scheduled for removal in v1.7.0. Use compute_v2() instead." in sample_func.__doc__

    # Verify invocation emits DeprecationWarning and returns correct value
    with pytest.warns(DeprecationWarning, match=r"'sample_func' was deprecated in v1\.5\.0"):
        res = sample_func(5, y=15)
    assert res == 20


def test_deprecated_decorator_on_method() -> None:
    """@deprecated works transparently on class methods."""

    class LegacyService:
        @deprecated(since="1.5.0", removed_in="1.7.0")
        def legacy_method(self, value: float) -> float:
            return value * 2.0

    service = LegacyService()
    with pytest.warns(
        DeprecationWarning,
        match=r"was deprecated in v1\.5\.0 and is scheduled for removal in v1\.7\.0",
    ):
        out = service.legacy_method(3.5)
    assert out == 7.0
