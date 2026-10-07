"""Deprecation primitives adhering to NEP-23 and SemVer 2.0.0 guidelines.

Provides explicit, inspectable deprecation warnings at runtime using standard
Python `DeprecationWarning` with stacklevel=2 to preserve traceback integrity.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def deprecate(
    feature: str,
    since: str,
    removed_in: str,
    alternative: str | None = None,
    stacklevel: int = 2,
) -> None:
    """Emit a standard DeprecationWarning for a deprecated API or parameter.

    Parameters
    ----------
    feature : str
        Name of the deprecated function, method, parameter, or module.
    since : str
        The version in which the feature was marked as deprecated (e.g. '1.5.0').
    removed_in : str
        The scheduled version where the feature will be removed (must be >= 2 minor
        versions ahead or >= 1 calendar year per NEP-23, e.g. '1.7.0').
    alternative : str, optional
        Guidance on what alternative API to use instead.
    stacklevel : int, default 2
        Stacklevel pointing to the user's callsite.
    """
    msg = f"'{feature}' was deprecated in v{since} and is scheduled for removal in v{removed_in}."
    if alternative:
        msg += f" {alternative}"
    warnings.warn(msg, category=DeprecationWarning, stacklevel=stacklevel)


def deprecated(
    since: str,
    removed_in: str,
    alternative: str | None = None,
) -> Callable[[F], F]:
    """Decorator to mark a function, method, or class as deprecated.

    Emits `DeprecationWarning` with `stacklevel=2` whenever the decorated callable is invoked.

    Parameters
    ----------
    since : str
        The version in which the feature was marked as deprecated.
    removed_in : str
        The scheduled version for removal (NEP-23: at least 2 minor versions ahead).
    alternative : str, optional
        Alternative method, function, or pattern to use instead.

    Returns
    -------
    Callable[[F], F]
        The decorated callable emitting DeprecationWarning upon invocation.
    """

    def decorator(func: F) -> F:
        name = getattr(func, "__name__", "callable")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            deprecate(
                feature=name,
                since=since,
                removed_in=removed_in,
                alternative=alternative,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        # Set deprecation attributes for inspection
        any_wrapper = cast(Any, wrapper)
        any_wrapper.__deprecated__ = True
        any_wrapper.__deprecated_since__ = since
        any_wrapper.__deprecated_removed_in__ = removed_in
        any_wrapper.__deprecated_alternative__ = alternative

        # Append Sphinx/Diátaxis deprecation marker to docstring
        doc = wrapper.__doc__ or ""
        marker = f"\n\n.. deprecated:: {since}\n   Scheduled for removal in v{removed_in}."
        if alternative:
            marker += f" {alternative}"
        wrapper.__doc__ = doc + marker

        return cast(F, wrapper)

    return decorator
