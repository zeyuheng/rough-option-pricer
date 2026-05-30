from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from hybrid_american_pricer.models import (
    BinomialTreePricer,
    BlackScholesPricer,
    MonteCarloPricer,
)
from hybrid_american_pricer.options import MarketState, OptionContract


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    contract: OptionContract
    market: MarketState


def default_cases() -> list[BenchmarkCase]:
    """Small but representative benchmark grid for stage 1."""

    return [
        BenchmarkCase(
            "atm_put_1y",
            OptionContract(strike=100.0, maturity=1.0, kind="put", exercise="european"),
            MarketState(spot=100.0, rate=0.05, volatility=0.20),
        ),
        BenchmarkCase(
            "itm_put_6m",
            OptionContract(strike=110.0, maturity=0.5, kind="put", exercise="european"),
            MarketState(spot=95.0, rate=0.03, volatility=0.25),
        ),
        BenchmarkCase(
            "otm_call_2y",
            OptionContract(strike=110.0, maturity=2.0, kind="call", exercise="european"),
            MarketState(spot=100.0, rate=0.04, volatility=0.30),
        ),
        BenchmarkCase(
            "high_vol_put_1y",
            OptionContract(strike=100.0, maturity=1.0, kind="put", exercise="european"),
            MarketState(spot=105.0, rate=0.02, volatility=0.45),
        ),
    ]


def _relative_error(price: float, reference: float) -> float:
    return abs(price - reference) / max(abs(reference), 1e-12)


def run_european_benchmarks(cases: list[BenchmarkCase]) -> pd.DataFrame:
    """Compare Black-Scholes, European binomial tree, and plain Monte Carlo."""

    bs = BlackScholesPricer()
    tree = BinomialTreePricer(steps=1000)
    mc = MonteCarloPricer(n_paths=30000, n_steps=80, seed=42)
    rows = []

    for case in cases:
        bs_result = bs.price(case.contract, case.market)
        tree_result = tree.price(case.contract, case.market)
        mc_result = mc.price(case.contract, case.market)
        reference_price = bs_result.price

        for method, result, comparison in [
            ("black_scholes", bs_result, "reference"),
            ("binomial_tree", tree_result, "Black-Scholes vs Binomial Tree"),
            ("monte_carlo", mc_result, "Monte Carlo vs Black-Scholes"),
        ]:
            rows.append(
                {
                    "case_id": case.case_id,
                    "comparison": comparison,
                    "option_kind": case.contract.kind,
                    "exercise": case.contract.exercise,
                    "spot": case.market.spot,
                    "strike": case.contract.strike,
                    "maturity": case.contract.maturity,
                    "rate": case.market.rate,
                    "volatility": case.market.volatility,
                    "method": method,
                    "reference_method": "black_scholes",
                    "reference_price": reference_price,
                    "price": result.price,
                    "absolute_error": abs(result.price - reference_price),
                    "relative_error": _relative_error(result.price, reference_price),
                    "std_error": result.std_error,
                    "runtime_seconds": result.runtime_seconds,
                }
            )

    return pd.DataFrame(rows)


def run_american_vs_european_put() -> pd.DataFrame:
    """Compare American and European put values under the same binomial tree."""

    cases = [
        BenchmarkCase(
            "american_put_atm_1y",
            OptionContract(strike=100.0, maturity=1.0, kind="put", exercise="american"),
            MarketState(spot=100.0, rate=0.05, volatility=0.20),
        ),
        BenchmarkCase(
            "american_put_itm_1y",
            OptionContract(strike=110.0, maturity=1.0, kind="put", exercise="american"),
            MarketState(spot=95.0, rate=0.05, volatility=0.25),
        ),
    ]
    tree = BinomialTreePricer(steps=1000)
    rows = []
    for case in cases:
        american = tree.price(case.contract, case.market)
        european_contract = OptionContract(
            strike=case.contract.strike,
            maturity=case.contract.maturity,
            kind=case.contract.kind,
            exercise="european",
        )
        european = tree.price(european_contract, case.market)
        rows.extend(
            [
                {
                    "case_id": case.case_id,
                    "comparison": "American Put vs European Put",
                    "option_kind": "put",
                    "exercise": "european",
                    "spot": case.market.spot,
                    "strike": case.contract.strike,
                    "maturity": case.contract.maturity,
                    "rate": case.market.rate,
                    "volatility": case.market.volatility,
                    "method": "binomial_tree_european",
                    "reference_method": "binomial_tree_european",
                    "reference_price": european.price,
                    "price": european.price,
                    "absolute_error": 0.0,
                    "relative_error": 0.0,
                    "std_error": european.std_error,
                    "runtime_seconds": european.runtime_seconds,
                    "early_exercise_premium": 0.0,
                },
                {
                    "case_id": case.case_id,
                    "comparison": "American Put vs European Put",
                    "option_kind": "put",
                    "exercise": "american",
                    "spot": case.market.spot,
                    "strike": case.contract.strike,
                    "maturity": case.contract.maturity,
                    "rate": case.market.rate,
                    "volatility": case.market.volatility,
                    "method": "binomial_tree_american",
                    "reference_method": "binomial_tree_european",
                    "reference_price": european.price,
                    "price": american.price,
                    "absolute_error": american.price - european.price,
                    "relative_error": _relative_error(american.price, european.price),
                    "std_error": american.std_error,
                    "runtime_seconds": american.runtime_seconds,
                    "early_exercise_premium": american.price - european.price,
                },
            ]
        )

    return pd.DataFrame(rows)


def run_stage1_benchmarks() -> pd.DataFrame:
    european = run_european_benchmarks(default_cases())
    american = run_american_vs_european_put()
    return pd.concat([european, american], ignore_index=True).fillna({"early_exercise_premium": 0.0})


def write_runtime_plot(results: pd.DataFrame, output_path: Path) -> None:
    summary = (
        results.groupby("method", as_index=False)["runtime_seconds"]
        .mean()
        .sort_values("runtime_seconds")
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(summary["method"], summary["runtime_seconds"], color="#3b82f6")
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean runtime (seconds)")
    ax.set_title("Benchmark pricing runtime by method")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage 1 classical pricing benchmarks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tables/benchmark_results.csv"),
        help="CSV path for benchmark results.",
    )
    parser.add_argument(
        "--runtime-plot",
        type=Path,
        default=Path("reports/figures/benchmark_runtime.png"),
        help="PNG path for the runtime comparison plot.",
    )
    args = parser.parse_args()

    results = run_stage1_benchmarks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    write_runtime_plot(results, args.runtime_plot)

    display_columns = [
        "case_id",
        "comparison",
        "method",
        "price",
        "absolute_error",
        "relative_error",
        "runtime_seconds",
    ]
    print(results[display_columns].to_string(index=False))
    print(f"\nwrote {args.output}")
    print(f"wrote {args.runtime_plot}")


if __name__ == "__main__":
    main()
