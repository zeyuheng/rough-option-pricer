from __future__ import annotations

import pandas as pd

from hybrid_american_pricer.models import (
    BinomialTreePricer,
    BlackScholesPricer,
    LSMCPricer,
    MonteCarloPricer,
    RoughBergomiPricer,
)
from hybrid_american_pricer.options import MarketState, OptionContract


def run_single_case() -> pd.DataFrame:
    contract = OptionContract(strike=100.0, maturity=1.0, kind="put", exercise="american")
    market = MarketState(spot=100.0, rate=0.05, volatility=0.2)
    pricers = {
        "black_scholes": BlackScholesPricer(),
        "binomial_tree": BinomialTreePricer(steps=500),
        "monte_carlo": MonteCarloPricer(n_paths=10000, n_steps=50),
        "lsmc": LSMCPricer(n_paths=10000, n_steps=50),
        "rough_bergomi": RoughBergomiPricer(n_paths=5000, n_steps=40),
    }
    rows = []
    for name, pricer in pricers.items():
        result = pricer.price(contract, market)
        rows.append(
            {
                "method": name,
                "price": result.price,
                "std_error": result.std_error,
                "runtime_seconds": result.runtime_seconds,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    print(run_single_case().to_string(index=False))


if __name__ == "__main__":
    main()

