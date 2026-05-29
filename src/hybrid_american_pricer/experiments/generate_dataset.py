from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hybrid_american_pricer.meta.features import build_feature_row
from hybrid_american_pricer.models import BinomialTreePricer, BlackScholesPricer, LSMCPricer
from hybrid_american_pricer.options import MarketState, OptionContract


def generate_dataset(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    rows = []
    bs = BlackScholesPricer()
    tree = BinomialTreePricer(steps=300)
    lsmc = LSMCPricer(n_paths=4000, n_steps=40, seed=seed)
    for i in range(n_samples):
        spot = 50.0 + (i * 17 % 101)
        strike = 80.0 + (i * 13 % 41)
        maturity = 0.25 + (i % 8) * 0.25
        rate = 0.01 + (i % 8) * 0.01
        volatility = 0.10 + (i % 11) * 0.04
        hurst = 0.05 + (i % 9) * 0.05
        eta = 0.5 + (i % 6) * 0.5
        rho = -0.9 + (i % 10) * 0.1
        contract = OptionContract(strike=strike, maturity=maturity)
        market = MarketState(spot=spot, rate=rate, volatility=volatility)
        base_prices = {
            "black_scholes": bs.price(contract, market).price,
            "tree": tree.price(contract, market).price,
            "lsmc": lsmc.price(contract, market).price,
        }
        row = build_feature_row(contract, market, base_prices, {"hurst": hurst, "eta": eta, "rho": rho})
        row["target_price"] = base_prices["tree"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("data/processed/pricing_dataset.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_dataset(args.n_samples).to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

