from __future__ import annotations

import math

import numpy as np

from hybrid_american_pricer.models.base import PricingResult
from hybrid_american_pricer.models.basis import design_matrix
from hybrid_american_pricer.models.path_simulation import simulate_black_scholes_paths
from hybrid_american_pricer.options.instruments import MarketState, OptionContract
from hybrid_american_pricer.options.payoffs import payoff
from hybrid_american_pricer.utils.timing import timed


class LSMCPricer:
    """Longstaff-Schwartz least-squares Monte Carlo for American options."""

    def __init__(
        self,
        n_paths: int = 20000,
        n_steps: int = 50,
        basis: str = "laguerre",
        degree: int = 3,
        seed: int | None = 42,
    ) -> None:
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.basis = basis
        self.degree = degree
        self.seed = seed

    @timed
    def price(self, contract: OptionContract, market: MarketState) -> PricingResult:
        paths = simulate_black_scholes_paths(
            market, contract.maturity, self.n_paths, self.n_steps, seed=self.seed
        )
        return self.price_from_paths(contract, market.rate, paths)

    def price_from_paths(
        self, contract: OptionContract, rate: float, paths: np.ndarray
    ) -> PricingResult:
        dt = contract.maturity / (paths.shape[1] - 1)
        discount = math.exp(-rate * dt)
        cashflows = payoff(paths[:, -1], contract).astype(float)
        exercise_times = np.full(paths.shape[0], paths.shape[1] - 1, dtype=int)
        boundaries: list[tuple[int, float]] = []

        for step in range(paths.shape[1] - 2, 0, -1):
            immediate = payoff(paths[:, step], contract).astype(float)
            in_money = immediate > 0.0
            cashflows *= discount
            if np.count_nonzero(in_money) <= self.degree + 1:
                continue

            x = paths[in_money, step]
            y = cashflows[in_money]
            basis_matrix = design_matrix(x, self.basis, self.degree)
            coeffs, *_ = np.linalg.lstsq(basis_matrix, y, rcond=None)
            continuation = basis_matrix @ coeffs
            should_exercise = immediate[in_money] > continuation

            indices = np.flatnonzero(in_money)
            exercise_indices = indices[should_exercise]
            cashflows[exercise_indices] = immediate[exercise_indices]
            exercise_times[exercise_indices] = step

            if exercise_indices.size:
                boundary = (
                    float(np.max(paths[exercise_indices, step]))
                    if contract.kind == "put"
                    else float(np.min(paths[exercise_indices, step]))
                )
                boundaries.append((step, boundary))

        discounted_cashflows = cashflows * discount
        return PricingResult(
            price=float(np.mean(discounted_cashflows)),
            std_error=float(np.std(discounted_cashflows, ddof=1) / math.sqrt(paths.shape[0])),
            metadata={
                "basis": self.basis,
                "degree": self.degree,
                "exercise_times": exercise_times,
                "exercise_boundaries": list(reversed(boundaries)),
            },
        )

