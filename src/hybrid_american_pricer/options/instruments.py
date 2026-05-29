from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OptionKind = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]


@dataclass(frozen=True)
class OptionContract:
    """Vanilla option contract definition."""

    strike: float
    maturity: float
    kind: OptionKind = "put"
    exercise: ExerciseStyle = "american"

    def validate(self) -> None:
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.maturity <= 0:
            raise ValueError("maturity must be positive")


@dataclass(frozen=True)
class MarketState:
    """Market inputs shared by pricing engines."""

    spot: float
    rate: float
    volatility: float
    dividend: float = 0.0

    def validate(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")

