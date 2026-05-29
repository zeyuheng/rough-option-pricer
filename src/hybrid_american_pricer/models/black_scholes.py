from __future__ import annotations

import math

from scipy.stats import norm

from hybrid_american_pricer.models.base import PricingResult
from hybrid_american_pricer.options.instruments import MarketState, OptionContract
from hybrid_american_pricer.utils.timing import timed


class BlackScholesPricer:
    """Closed-form European Black-Scholes benchmark."""

    @timed
    def price(self, contract: OptionContract, market: MarketState) -> PricingResult:
        contract.validate()
        market.validate()
        if market.volatility == 0:
            terminal = market.spot * math.exp((market.rate - market.dividend) * contract.maturity)
            intrinsic = max(terminal - contract.strike, 0.0)
            if contract.kind == "put":
                intrinsic = max(contract.strike - terminal, 0.0)
            return PricingResult(price=math.exp(-market.rate * contract.maturity) * intrinsic)

        sigma_sqrt_t = market.volatility * math.sqrt(contract.maturity)
        d1 = (
            math.log(market.spot / contract.strike)
            + (market.rate - market.dividend + 0.5 * market.volatility**2) * contract.maturity
        ) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t
        discounted_spot = market.spot * math.exp(-market.dividend * contract.maturity)
        discounted_strike = contract.strike * math.exp(-market.rate * contract.maturity)

        if contract.kind == "call":
            price = discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
        elif contract.kind == "put":
            price = discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)
        else:
            raise ValueError(f"unsupported option kind: {contract.kind}")
        return PricingResult(price=float(price), metadata={"d1": d1, "d2": d2})

