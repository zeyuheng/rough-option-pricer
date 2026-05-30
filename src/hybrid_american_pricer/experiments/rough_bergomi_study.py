from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hybrid_american_pricer.models import (
    BinomialTreePricer,
    BlackScholesPricer,
    LSMCPricer,
    RoughBergomiPricer,
)
from hybrid_american_pricer.models.path_simulation import (
    simulate_black_scholes_paths,
    simulate_rough_bergomi_paths,
)
from hybrid_american_pricer.options import MarketState, OptionContract


def default_market() -> MarketState:
    return MarketState(spot=100.0, rate=0.05, volatility=0.20)


def european_put() -> OptionContract:
    return OptionContract(strike=100.0, maturity=1.0, kind="put", exercise="european")


def american_put() -> OptionContract:
    return OptionContract(strike=100.0, maturity=1.0, kind="put", exercise="american")


def roughness_score(variance_paths: np.ndarray) -> float:
    """Path roughness proxy: mean absolute log-variance increment."""

    log_variance = np.log(np.maximum(variance_paths, 1e-12))
    return float(np.mean(np.abs(np.diff(log_variance, axis=1))))


def sample_path_table(
    market: MarketState | None = None,
    hurst: float = 0.10,
    eta: float = 1.80,
    rho: float = -0.70,
    xi0: float | None = None,
    n_paths: int = 12,
    n_steps: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a tidy table with rough Bergomi sample paths for plotting."""

    market = market or default_market()
    asset_paths, variance_paths = simulate_rough_bergomi_paths(
        market=market,
        maturity=1.0,
        n_paths=n_paths,
        n_steps=n_steps,
        hurst=hurst,
        eta=eta,
        rho=rho,
        xi0=xi0,
        seed=seed,
    )
    times = np.linspace(0.0, 1.0, n_steps + 1)
    rows = []
    for path_id in range(n_paths):
        for step, time in enumerate(times):
            rows.append(
                {
                    "path_id": path_id,
                    "time": time,
                    "asset_price": asset_paths[path_id, step],
                    "variance": variance_paths[path_id, step],
                    "volatility": np.sqrt(variance_paths[path_id, step]),
                    "hurst": hurst,
                    "eta": eta,
                    "rho": rho,
                }
            )
    return pd.DataFrame(rows)


def run_rough_price_table(
    market: MarketState | None = None,
    n_paths: int = 15000,
    n_steps: int = 75,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare constant-volatility benchmarks with rough Bergomi-style pricing."""

    market = market or default_market()
    eput = european_put()
    aput = american_put()
    bs_european = BlackScholesPricer().price(eput, market)
    tree_american = BinomialTreePricer(steps=1000).price(aput, market)
    lsmc_american = LSMCPricer(n_paths=n_paths, n_steps=n_steps, seed=seed).price(aput, market)

    rough_params = [
        {"hurst": 0.05, "eta": 1.80, "rho": -0.70},
        {"hurst": 0.10, "eta": 1.80, "rho": -0.70},
        {"hurst": 0.30, "eta": 1.80, "rho": -0.70},
    ]
    rows = [
        {
            "model": "black_scholes_constant_vol",
            "exercise": "european",
            "hurst": np.nan,
            "eta": np.nan,
            "rho": np.nan,
            "price": bs_european.price,
            "std_error": bs_european.std_error,
            "runtime_seconds": bs_european.runtime_seconds,
            "comparison_reference": "black_scholes_constant_vol",
            "difference_vs_reference": 0.0,
        },
        {
            "model": "binomial_tree_constant_vol",
            "exercise": "american",
            "hurst": np.nan,
            "eta": np.nan,
            "rho": np.nan,
            "price": tree_american.price,
            "std_error": tree_american.std_error,
            "runtime_seconds": tree_american.runtime_seconds,
            "comparison_reference": "binomial_tree_constant_vol",
            "difference_vs_reference": 0.0,
        },
        {
            "model": "lsmc_constant_vol",
            "exercise": "american",
            "hurst": np.nan,
            "eta": np.nan,
            "rho": np.nan,
            "price": lsmc_american.price,
            "std_error": lsmc_american.std_error,
            "runtime_seconds": lsmc_american.runtime_seconds,
            "comparison_reference": "binomial_tree_constant_vol",
            "difference_vs_reference": lsmc_american.price - tree_american.price,
        },
    ]
    for params in rough_params:
        rough_european = RoughBergomiPricer(
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            **params,
        ).price(eput, market)
        rough_american = RoughBergomiPricer(
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            **params,
        ).price(aput, market)
        rows.extend(
            [
                {
                    "model": "rough_bergomi_mc",
                    "exercise": "european",
                    **params,
                    "price": rough_european.price,
                    "std_error": rough_european.std_error,
                    "runtime_seconds": rough_european.runtime_seconds,
                    "comparison_reference": "black_scholes_constant_vol",
                    "difference_vs_reference": rough_european.price - bs_european.price,
                },
                {
                    "model": "rough_bergomi_lsmc",
                    "exercise": "american",
                    **params,
                    "price": rough_american.price,
                    "std_error": rough_american.std_error,
                    "runtime_seconds": rough_american.runtime_seconds,
                    "comparison_reference": "binomial_tree_constant_vol",
                    "difference_vs_reference": rough_american.price - tree_american.price,
                },
            ]
        )
    return pd.DataFrame(rows)


def run_hurst_sensitivity(
    market: MarketState | None = None,
    hurst_values: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.45),
    eta: float = 1.80,
    rho: float = -0.70,
    n_paths: int = 10000,
    n_steps: int = 75,
    seed: int = 42,
) -> pd.DataFrame:
    """Measure roughness and price sensitivity across Hurst exponents."""

    market = market or default_market()
    eput = european_put()
    aput = american_put()
    rows = []
    for hurst in hurst_values:
        asset_paths, variance_paths = simulate_rough_bergomi_paths(
            market=market,
            maturity=eput.maturity,
            n_paths=n_paths,
            n_steps=n_steps,
            hurst=hurst,
            eta=eta,
            rho=rho,
            seed=seed,
        )
        rough_european = RoughBergomiPricer(
            n_paths=n_paths,
            n_steps=n_steps,
            hurst=hurst,
            eta=eta,
            rho=rho,
            seed=seed,
        ).price(eput, market)
        rough_american = LSMCPricer(n_paths=n_paths, n_steps=n_steps, seed=seed).price_from_paths(
            aput, market.rate, asset_paths
        )
        rows.append(
            {
                "hurst": hurst,
                "eta": eta,
                "rho": rho,
                "roughness_score": roughness_score(variance_paths),
                "mean_terminal_variance": float(np.mean(variance_paths[:, -1])),
                "european_rough_price": rough_european.price,
                "european_std_error": rough_european.std_error,
                "american_rough_lsmc_price": rough_american.price,
                "american_std_error": rough_american.std_error,
            }
        )
    return pd.DataFrame(rows)


def plot_sample_paths(path_data: pd.DataFrame, volatility_output: Path, asset_output: Path) -> None:
    """Plot rough volatility and asset paths for visual inspection."""

    for column, ylabel, title, output in [
        ("volatility", "Instantaneous volatility", "rough Bergomi-style volatility paths", volatility_output),
        ("asset_price", "Asset price", "rough Bergomi-style asset paths", asset_output),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for path_id, group in path_data.groupby("path_id"):
            alpha = 0.85 if path_id < 4 else 0.30
            ax.plot(group["time"], group[column], linewidth=1.2, alpha=alpha)
        ax.set_xlabel("Time")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        plt.close(fig)


def plot_hurst_sensitivity(sensitivity: pd.DataFrame, output_path: Path) -> None:
    """Plot price and roughness changes as H varies."""

    fig, ax_price = plt.subplots(figsize=(8, 4.8))
    ax_rough = ax_price.twinx()
    ordered = sensitivity.sort_values("hurst")
    price_line = ax_price.plot(
        ordered["hurst"],
        ordered["american_rough_lsmc_price"],
        marker="o",
        color="#2563eb",
        label="American rough LSMC price",
    )
    rough_line = ax_rough.plot(
        ordered["hurst"],
        ordered["roughness_score"],
        marker="s",
        color="#dc2626",
        label="Roughness score",
    )
    ax_price.set_xlabel("Hurst exponent H")
    ax_price.set_ylabel("American option price")
    ax_rough.set_ylabel("Mean absolute log-variance increment")
    ax_price.set_title("Hurst sensitivity: roughness and price")
    lines = price_line + rough_line
    ax_price.legend(lines, [line.get_label() for line in lines], loc="best")
    ax_price.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_stage3(price_table: pd.DataFrame, sensitivity: pd.DataFrame) -> dict[str, float | bool]:
    """Compact answers to the stage-3 completion questions."""

    roughness_corr = sensitivity["hurst"].corr(sensitivity["roughness_score"])
    rough_rows = price_table[price_table["model"].str.contains("rough_bergomi")]
    max_abs_price_shift = float(rough_rows["difference_vs_reference"].abs().max())
    american_rows = price_table[
        (price_table["model"] == "rough_bergomi_lsmc") & (price_table["exercise"] == "american")
    ]
    return {
        "hurst_vs_roughness_correlation": float(roughness_corr),
        "max_abs_rough_price_shift": max_abs_price_shift,
        "rough_american_lsmc_available": bool(len(american_rows) > 0),
        "roughness_at_min_hurst": float(
            sensitivity.loc[sensitivity["hurst"].idxmin(), "roughness_score"]
        ),
        "roughness_at_max_hurst": float(
            sensitivity.loc[sensitivity["hurst"].idxmax(), "roughness_score"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage 3 rough Bergomi-style study.")
    parser.add_argument(
        "--path-output",
        type=Path,
        default=Path("reports/tables/rough_bergomi_sample_paths.csv"),
    )
    parser.add_argument(
        "--price-output",
        type=Path,
        default=Path("reports/tables/rough_bergomi_price_table.csv"),
    )
    parser.add_argument(
        "--sensitivity-output",
        type=Path,
        default=Path("reports/tables/hurst_sensitivity.csv"),
    )
    parser.add_argument(
        "--volatility-plot",
        type=Path,
        default=Path("reports/figures/rough_bergomi_volatility_paths.png"),
    )
    parser.add_argument(
        "--asset-plot",
        type=Path,
        default=Path("reports/figures/rough_bergomi_asset_paths.png"),
    )
    parser.add_argument(
        "--hurst-plot",
        type=Path,
        default=Path("reports/figures/hurst_sensitivity.png"),
    )
    args = parser.parse_args()

    paths = sample_path_table()
    price_table = run_rough_price_table()
    sensitivity = run_hurst_sensitivity()

    args.path_output.parent.mkdir(parents=True, exist_ok=True)
    paths.to_csv(args.path_output, index=False)
    price_table.to_csv(args.price_output, index=False)
    sensitivity.to_csv(args.sensitivity_output, index=False)
    plot_sample_paths(paths, args.volatility_plot, args.asset_plot)
    plot_hurst_sensitivity(sensitivity, args.hurst_plot)

    print("rough Bergomi-style price table:")
    print(price_table.to_string(index=False))
    print("\nHurst sensitivity:")
    print(sensitivity.to_string(index=False))
    print("\nStage 3 summary:")
    print(summarize_stage3(price_table, sensitivity))
    print(f"\nwrote {args.path_output}")
    print(f"wrote {args.price_output}")
    print(f"wrote {args.sensitivity_output}")
    print(f"wrote {args.volatility_plot}")
    print(f"wrote {args.asset_plot}")
    print(f"wrote {args.hurst_plot}")


if __name__ == "__main__":
    main()
