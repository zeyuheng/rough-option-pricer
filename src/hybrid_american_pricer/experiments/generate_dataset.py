from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from hybrid_american_pricer.meta.features import build_feature_row
from hybrid_american_pricer.models import (
    BinomialTreePricer,
    BlackScholesPricer,
    LSMCPricer,
    MonteCarloPricer,
    RoughBergomiPricer,
)
from hybrid_american_pricer.options import MarketState, OptionContract


PARAMETER_RANGES: dict[str, tuple[float, float]] = {
    "spot": (50.0, 150.0),
    "strike": (80.0, 120.0),
    "maturity": (0.25, 2.0),
    "rate": (0.01, 0.08),
    "dividend": (0.00, 0.05),
    "volatility": (0.10, 0.60),
    "hurst": (0.05, 0.45),
    "eta": (0.50, 3.00),
    "rho": (-0.90, 0.00),
}


def sample_parameter_grid(n_samples: int, seed: int = 42) -> pd.DataFrame:
    """Randomly sample the stage-4 synthetic option parameter space."""

    rng = np.random.default_rng(seed)
    samples = {
        name: rng.uniform(low, high, size=n_samples)
        for name, (low, high) in PARAMETER_RANGES.items()
    }
    samples["case_id"] = [f"synthetic_{i:05d}" for i in range(n_samples)]
    return pd.DataFrame(samples)


def black_scholes_put_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    volatility: float,
) -> float:
    contract = OptionContract(strike=strike, maturity=maturity, kind="put", exercise="european")
    market = MarketState(spot=spot, rate=rate, dividend=dividend, volatility=volatility)
    return BlackScholesPricer().price(contract, market).price


def implied_volatility_from_put_price(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
) -> tuple[float, float]:
    """Return Black-Scholes-equivalent put IV and a clipping flag.

    American and rough-vol prices can sit outside the European Black-Scholes
    range, so the price is clipped to a valid European range before inversion.
    """

    discounted_strike = strike * np.exp(-rate * maturity)
    discounted_spot = spot * np.exp(-dividend * maturity)
    lower = max(discounted_strike - discounted_spot, 0.0)
    upper = discounted_strike
    if upper <= lower + 1e-8:
        return np.nan, 1.0

    clipped_price = float(np.clip(price, lower + 1e-8, upper - 1e-8))
    was_clipped = float(abs(clipped_price - price) > 1e-8)

    def objective(volatility: float) -> float:
        return (
            black_scholes_put_price(spot, strike, maturity, rate, dividend, volatility)
            - clipped_price
        )

    return float(brentq(objective, 1e-4, 5.0, maxiter=100)), was_clipped


def synthetic_market_quote(
    fair_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Create a simple synthetic bid/ask quote around a fair value."""

    log_moneyness = np.log(spot / strike)
    relative_spread = (
        0.008
        + 0.015 * abs(log_moneyness)
        + 0.006 * float(maturity <= 0.5)
        + rng.uniform(0.0, 0.012)
    )
    quote_noise = rng.normal(0.0, 0.20 * max(relative_spread * max(fair_price, 1.0), 0.01))
    mid_price = max(fair_price + quote_noise, 1e-4)
    spread = max(0.01, relative_spread * max(mid_price, 1.0))
    bid = max(mid_price - 0.5 * spread, 0.0)
    ask = mid_price + 0.5 * spread
    return {
        "bid": bid,
        "ask": ask,
        "mid_price": 0.5 * (bid + ask),
        "spread": ask - bid,
        "relative_spread": (ask - bid) / max(0.5 * (bid + ask), 1e-8),
    }


def generate_dataset(
    n_samples: int = 120,
    seed: int = 42,
    base_tree_steps: int = 250,
    mc_paths: int = 4000,
    lsmc_paths: int = 3000,
    rough_paths: int = 2000,
    target_rough_paths: int = 6000,
    n_steps: int = 40,
    target_n_steps: int = 60,
) -> pd.DataFrame:
    """Generate meta-model features and target labels from synthetic parameters."""

    sampled = sample_parameter_grid(n_samples=n_samples, seed=seed)
    quote_rng = np.random.default_rng(seed + 10_000)
    rows = []
    bs = BlackScholesPricer()
    base_tree = BinomialTreePricer(steps=base_tree_steps)

    for row in sampled.itertuples(index=False):
        contract = OptionContract(strike=row.strike, maturity=row.maturity, kind="put", exercise="american")
        european_contract = OptionContract(
            strike=row.strike,
            maturity=row.maturity,
            kind="put",
            exercise="european",
        )
        market = MarketState(
            spot=row.spot,
            rate=row.rate,
            dividend=row.dividend,
            volatility=row.volatility,
        )

        rough_params = {"hurst": row.hurst, "eta": row.eta, "rho": row.rho}
        base_prices = {
            "black_scholes": bs.price(european_contract, market).price,
            "tree": base_tree.price(contract, market).price,
            "monte_carlo": MonteCarloPricer(
                n_paths=mc_paths,
                n_steps=n_steps,
                seed=seed,
            ).price(european_contract, market).price,
            "lsmc": LSMCPricer(
                n_paths=lsmc_paths,
                n_steps=n_steps,
                seed=seed,
            ).price(contract, market).price,
            "rough_bergomi_mc": RoughBergomiPricer(
                n_paths=rough_paths,
                n_steps=n_steps,
                seed=seed,
                **rough_params,
            ).price(european_contract, market).price,
            "rough_bergomi_lsmc": RoughBergomiPricer(
                n_paths=rough_paths,
                n_steps=n_steps,
                seed=seed,
                **rough_params,
            ).price(contract, market).price,
        }

        target_result = RoughBergomiPricer(
            n_paths=target_rough_paths,
            n_steps=target_n_steps,
            seed=seed + 999,
            **rough_params,
        ).price(contract, market)
        quote = synthetic_market_quote(
            fair_price=target_result.price,
            spot=row.spot,
            strike=row.strike,
            maturity=row.maturity,
            rng=quote_rng,
        )
        market_iv, market_iv_clipped = implied_volatility_from_put_price(
            quote["mid_price"],
            row.spot,
            row.strike,
            row.maturity,
            row.rate,
            row.dividend,
        )
        model_iv, model_iv_clipped = implied_volatility_from_put_price(
            target_result.price,
            row.spot,
            row.strike,
            row.maturity,
            row.rate,
            row.dividend,
        )
        feature_row = build_feature_row(contract, market, base_prices, rough_params)
        feature_row.update(
            {
                "case_id": row.case_id,
                "target_method": f"rough_bergomi_lsmc_{target_rough_paths}x{target_n_steps}",
                "target_price": target_result.price,
                "target_runtime_seconds": target_result.runtime_seconds,
                "target_std_error": target_result.std_error,
                "target_minus_low_path_rough_lsmc": (
                    target_result.price - base_prices["rough_bergomi_lsmc"]
                ),
                "tree_vs_rough_target_gap": target_result.price - base_prices["tree"],
                "market_iv": market_iv,
                "model_iv": model_iv,
                "iv_error": market_iv - model_iv,
                "market_iv_clipped": market_iv_clipped,
                "model_iv_clipped": model_iv_clipped,
                **quote,
            }
        )
        rows.append(feature_row)
    return pd.DataFrame(rows)


def split_dataset(
    data: pd.DataFrame,
    train_frac: float = 0.70,
    valid_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle and split data into train, validation, and test sets."""

    if not 0 < train_frac < 1 or not 0 < valid_frac < 1:
        raise ValueError("train_frac and valid_frac must be in (0, 1)")
    if train_frac + valid_frac >= 1:
        raise ValueError("train_frac + valid_frac must be less than 1")

    shuffled = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    train_end = int(n * train_frac)
    valid_end = train_end + int(n * valid_frac)
    return (
        shuffled.iloc[:train_end].reset_index(drop=True),
        shuffled.iloc[train_end:valid_end].reset_index(drop=True),
        shuffled.iloc[valid_end:].reset_index(drop=True),
    )


def plot_dataset_distributions(data: pd.DataFrame, output_path: Path) -> None:
    """Plot key parameter and target distributions for data sanity checks."""

    columns = [
        "spot",
        "strike",
        "maturity",
        "rate",
        "dividend",
        "volatility",
        "hurst",
        "eta",
        "rho",
        "target_price",
    ]
    fig, axes = plt.subplots(2, 5, figsize=(13, 6.8))
    for ax, column in zip(axes.ravel(), columns, strict=True):
        ax.hist(data[column], bins=24, color="#2563eb", alpha=0.82)
        ax.set_title(column)
        ax.grid(alpha=0.18)
    fig.suptitle("Synthetic pricing dataset distributions", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_dataset_outputs(
    data: pd.DataFrame,
    output_dir: Path = Path("data/processed"),
    figure_path: Path = Path("reports/figures/pricing_dataset_distributions.png"),
    seed: int = 42,
) -> dict[str, Path]:
    """Write full dataset, splits, and distribution plot."""

    output_dir.mkdir(parents=True, exist_ok=True)
    train, valid, test = split_dataset(data, seed=seed)
    paths = {
        "full": output_dir / "pricing_dataset.csv",
        "train": output_dir / "train.csv",
        "valid": output_dir / "valid.csv",
        "test": output_dir / "test.csv",
        "figure": figure_path,
    }
    data.to_csv(paths["full"], index=False)
    train.to_csv(paths["train"], index=False)
    valid.to_csv(paths["valid"], index=False)
    test.to_csv(paths["test"], index=False)
    plot_dataset_distributions(data, figure_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stage-4 meta-pricer training data.")
    parser.add_argument("--n-samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("reports/figures/pricing_dataset_distributions.png"),
    )
    parser.add_argument("--target-rough-paths", type=int, default=6000)
    parser.add_argument("--base-tree-steps", type=int, default=250)
    parser.add_argument("--mc-paths", type=int, default=4000)
    parser.add_argument("--lsmc-paths", type=int, default=3000)
    parser.add_argument("--rough-paths", type=int, default=2000)
    parser.add_argument("--n-steps", type=int, default=40)
    parser.add_argument("--target-n-steps", type=int, default=60)
    args = parser.parse_args()
    data = generate_dataset(
        n_samples=args.n_samples,
        seed=args.seed,
        base_tree_steps=args.base_tree_steps,
        mc_paths=args.mc_paths,
        lsmc_paths=args.lsmc_paths,
        rough_paths=args.rough_paths,
        target_rough_paths=args.target_rough_paths,
        n_steps=args.n_steps,
        target_n_steps=args.target_n_steps,
    )
    paths = write_dataset_outputs(data, args.output_dir, args.figure, seed=args.seed)
    print(f"generated {len(data)} rows with {len(data.columns)} columns")
    for name, path in paths.items():
        print(f"wrote {name}: {path}")


if __name__ == "__main__":
    main()
