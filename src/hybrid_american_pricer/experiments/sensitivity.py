from __future__ import annotations

import pandas as pd

from hybrid_american_pricer.models import RoughBergomiPricer
from hybrid_american_pricer.options import MarketState, OptionContract


def hurst_sensitivity() -> pd.DataFrame:
    contract = OptionContract(strike=100.0, maturity=1.0, kind="put", exercise="american")
    market = MarketState(spot=100.0, rate=0.05, volatility=0.2)
    rows = []
    for hurst in [0.05, 0.10, 0.20, 0.30, 0.45]:
        result = RoughBergomiPricer(n_paths=3000, n_steps=40, hurst=hurst).price(contract, market)
        rows.append({"hurst": hurst, "price": result.price, "std_error": result.std_error})
    return pd.DataFrame(rows)


def main() -> None:
    print(hurst_sensitivity().to_string(index=False))


if __name__ == "__main__":
    main()

