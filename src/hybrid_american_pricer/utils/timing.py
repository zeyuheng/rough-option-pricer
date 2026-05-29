from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

from hybrid_american_pricer.models.base import PricingResult


F = TypeVar("F", bound=Callable[..., PricingResult])


def timed(fn: F) -> F:
    """Decorator that stores runtime on a PricingResult."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> PricingResult:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        return PricingResult(
            price=result.price,
            std_error=result.std_error,
            runtime_seconds=time.perf_counter() - start,
            metadata=result.metadata,
        )

    return wrapper  # type: ignore[return-value]

