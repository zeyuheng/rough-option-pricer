from __future__ import annotations

import math

import numpy as np

from hybrid_american_pricer.models.base import PricingResult
from hybrid_american_pricer.models.path_simulation import simulate_black_scholes_paths
from hybrid_american_pricer.options.instruments import MarketState, OptionContract
from hybrid_american_pricer.options.payoffs import payoff
from hybrid_american_pricer.utils.timing import timed


class MonteCarloPricer:
    """Plain Monte Carlo pricer for European payoff benchmarks."""

    def __init__(self, n_paths: int = 20000, n_steps: int = 50, seed: int | None = 42) -> None:
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed

    @timed
    def price(self, contract: OptionContract, market: MarketState) -> PricingResult:
        paths = simulate_black_scholes_paths(
            market, contract.maturity, self.n_paths, self.n_steps, seed=self.seed
        )
        discounted = math.exp(-market.rate * contract.maturity) * payoff(paths[:, -1], contract)
        return PricingResult(
            price=float(np.mean(discounted)),
            std_error=float(np.std(discounted, ddof=1) / math.sqrt(self.n_paths)),
            metadata={"n_paths": self.n_paths, "n_steps": self.n_steps},
        )

