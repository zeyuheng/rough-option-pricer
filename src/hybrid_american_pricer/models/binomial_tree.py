from __future__ import annotations

import math

import numpy as np

from hybrid_american_pricer.models.base import PricingResult
from hybrid_american_pricer.options.instruments import MarketState, OptionContract
from hybrid_american_pricer.options.payoffs import payoff
from hybrid_american_pricer.utils.timing import timed


class BinomialTreePricer:
    """Cox-Ross-Rubinstein binomial tree for European and American options."""

    def __init__(self, steps: int = 500) -> None:
        if steps < 1:
            raise ValueError("steps must be positive")
        self.steps = steps

    @timed
    def price(self, contract: OptionContract, market: MarketState) -> PricingResult:
        contract.validate()
        market.validate()
        dt = contract.maturity / self.steps
        if market.volatility == 0:
            terminal = market.spot * math.exp((market.rate - market.dividend) * contract.maturity)
            value = math.exp(-market.rate * contract.maturity) * payoff(terminal, contract)
            return PricingResult(price=float(value))

        up = math.exp(market.volatility * math.sqrt(dt))
        down = 1.0 / up
        growth = math.exp((market.rate - market.dividend) * dt)
        prob = (growth - down) / (up - down)
        if not 0.0 <= prob <= 1.0:
            raise ValueError("invalid risk-neutral probability")

        nodes = np.arange(self.steps + 1)
        spots = market.spot * (up ** (self.steps - nodes)) * (down**nodes)
        values = payoff(spots, contract).astype(float)
        disc = math.exp(-market.rate * dt)

        for step in range(self.steps - 1, -1, -1):
            values = disc * (prob * values[:-1] + (1.0 - prob) * values[1:])
            if contract.exercise == "american":
                nodes = np.arange(step + 1)
                spots = market.spot * (up ** (step - nodes)) * (down**nodes)
                values = np.maximum(values, payoff(spots, contract))
        return PricingResult(price=float(values[0]), metadata={"steps": self.steps})

