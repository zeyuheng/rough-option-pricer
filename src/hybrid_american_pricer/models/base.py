from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from hybrid_american_pricer.options.instruments import MarketState, OptionContract


@dataclass(frozen=True)
class PricingResult:
    price: float
    std_error: float | None = None
    runtime_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Pricer(Protocol):
    def price(self, contract: OptionContract, market: MarketState) -> PricingResult:
        """Price an option contract under the supplied market state."""

