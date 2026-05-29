from __future__ import annotations

import math

import numpy as np

from hybrid_american_pricer.models.base import PricingResult
from hybrid_american_pricer.models.lsmc import LSMCPricer
from hybrid_american_pricer.models.path_simulation import simulate_rough_bergomi_paths
from hybrid_american_pricer.options.instruments import MarketState, OptionContract
from hybrid_american_pricer.options.payoffs import payoff
from hybrid_american_pricer.utils.timing import timed


class RoughBergomiPricer:
    """rough Bergomi-style Monte Carlo pricer with optional American LSMC exercise."""

    def __init__(
        self,
        n_paths: int = 20000,
        n_steps: int = 50,
        hurst: float = 0.1,
        eta: float = 1.8,
        rho: float = -0.7,
        xi0: float | None = None,
        seed: int | None = 42,
    ) -> None:
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.hurst = hurst
        self.eta = eta
        self.rho = rho
        self.xi0 = xi0
        self.seed = seed

    @timed
    def price(self, contract: OptionContract, market: MarketState) -> PricingResult:
        paths, variance_paths = simulate_rough_bergomi_paths(
            market=market,
            maturity=contract.maturity,
            n_paths=self.n_paths,
            n_steps=self.n_steps,
            hurst=self.hurst,
            eta=self.eta,
            rho=self.rho,
            xi0=self.xi0,
            seed=self.seed,
        )
        if contract.exercise == "american":
            result = LSMCPricer(
                n_paths=self.n_paths,
                n_steps=self.n_steps,
                basis="laguerre",
                degree=3,
                seed=self.seed,
            ).price_from_paths(contract, market.rate, paths)
            result.metadata.update(self._metadata(variance_paths))
            return result

        discounted = math.exp(-market.rate * contract.maturity) * payoff(paths[:, -1], contract)
        return PricingResult(
            price=float(np.mean(discounted)),
            std_error=float(np.std(discounted, ddof=1) / math.sqrt(self.n_paths)),
            metadata=self._metadata(variance_paths),
        )

    def _metadata(self, variance_paths: np.ndarray) -> dict[str, float]:
        return {
            "hurst": self.hurst,
            "eta": self.eta,
            "rho": self.rho,
            "mean_terminal_variance": float(np.mean(variance_paths[:, -1])),
        }

