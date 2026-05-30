from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from hybrid_american_pricer.models import BinomialTreePricer, LSMCPricer
from hybrid_american_pricer.options import MarketState, OptionContract


def default_contract() -> OptionContract:
    """Canonical American put case used for the LSMC stage-2 study."""

    return OptionContract(strike=100.0, maturity=1.0, kind="put", exercise="american")


def default_market() -> MarketState:
    return MarketState(spot=100.0, rate=0.05, volatility=0.20)


def tree_reference_price(
    contract: OptionContract,
    market: MarketState,
    steps: int = 1500,
) -> float:
    """High-step binomial tree reference for American put comparisons."""

    return BinomialTreePricer(steps=steps).price(contract, market).price


def run_lsmc_convergence(
    contract: OptionContract | None = None,
    market: MarketState | None = None,
    path_counts: tuple[int, ...] = (1000, 3000, 8000, 15000),
    time_steps: tuple[int, ...] = (25, 50, 75),
    basis: str = "laguerre",
    degree: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare LSMC error across path counts and exercise time grids."""

    contract = contract or default_contract()
    market = market or default_market()
    reference = tree_reference_price(contract, market)
    rows = []

    for n_steps in time_steps:
        for n_paths in path_counts:
            result = LSMCPricer(
                n_paths=n_paths,
                n_steps=n_steps,
                basis=basis,
                degree=degree,
                seed=seed,
            ).price(contract, market)
            absolute_error = abs(result.price - reference)
            rows.append(
                {
                    "n_paths": n_paths,
                    "n_steps": n_steps,
                    "basis": basis,
                    "degree": degree,
                    "reference_method": "binomial_tree_american",
                    "reference_price": reference,
                    "lsmc_price": result.price,
                    "absolute_error": absolute_error,
                    "relative_error": absolute_error / reference,
                "std_error": result.std_error,
                "runtime_seconds": result.runtime_seconds,
                "mean_exercise_step": float(result.metadata["exercise_times"].mean()),
                "mean_regression_condition": result.metadata["mean_regression_condition"],
                "max_regression_condition": result.metadata["max_regression_condition"],
            }
        )

    return pd.DataFrame(rows)


def run_basis_comparison(
    contract: OptionContract | None = None,
    market: MarketState | None = None,
    bases: tuple[str, ...] = ("polynomial", "laguerre", "hermite"),
    n_paths: int = 15000,
    n_steps: int = 50,
    degree: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare LSMC regression basis families against the tree reference."""

    contract = contract or default_contract()
    market = market or default_market()
    reference = tree_reference_price(contract, market)
    rows = []

    for basis in bases:
        result = LSMCPricer(
            n_paths=n_paths,
            n_steps=n_steps,
            basis=basis,
            degree=degree,
            seed=seed,
        ).price(contract, market)
        absolute_error = abs(result.price - reference)
        rows.append(
            {
                "basis": basis,
                "degree": degree,
                "n_paths": n_paths,
                "n_steps": n_steps,
                "reference_method": "binomial_tree_american",
                "reference_price": reference,
                "lsmc_price": result.price,
                "absolute_error": absolute_error,
                "relative_error": absolute_error / reference,
                "std_error": result.std_error,
                "runtime_seconds": result.runtime_seconds,
                "n_boundary_points": len(result.metadata["exercise_boundaries"]),
                "mean_regression_condition": result.metadata["mean_regression_condition"],
                "max_regression_condition": result.metadata["max_regression_condition"],
            }
        )

    return pd.DataFrame(rows).sort_values(["absolute_error", "mean_regression_condition"])


def estimate_exercise_boundary(
    contract: OptionContract | None = None,
    market: MarketState | None = None,
    n_paths: int = 20000,
    n_steps: int = 75,
    basis: str = "laguerre",
    degree: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Extract the estimated early exercise boundary from one high-quality run."""

    contract = contract or default_contract()
    market = market or default_market()
    result = LSMCPricer(
        n_paths=n_paths,
        n_steps=n_steps,
        basis=basis,
        degree=degree,
        seed=seed,
    ).price(contract, market)
    boundary = pd.DataFrame(result.metadata["exercise_boundaries"], columns=["step", "boundary"])
    if boundary.empty:
        return boundary
    boundary["time"] = boundary["step"] / n_steps * contract.maturity
    boundary["lsmc_price"] = result.price
    boundary["std_error"] = result.std_error
    return boundary


def plot_convergence(convergence: pd.DataFrame, output_path: Path) -> None:
    """Plot relative error against path count for each exercise time grid."""

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for n_steps, group in convergence.groupby("n_steps"):
        ordered = group.sort_values("n_paths")
        ax.plot(
            ordered["n_paths"],
            ordered["relative_error"],
            marker="o",
            label=f"{n_steps} steps",
        )
    ax.set_xlabel("Number of Monte Carlo paths")
    ax.set_ylabel("Relative error vs binomial tree")
    ax.set_title("LSMC convergence against American binomial tree")
    ax.legend(title="Exercise grid")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_exercise_boundary(boundary: pd.DataFrame, output_path: Path) -> None:
    """Plot estimated American put early exercise boundary over time."""

    if boundary.empty:
        raise ValueError("no exercise boundary points were generated")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ordered = boundary.sort_values("time")
    ax.plot(ordered["time"], ordered["boundary"], marker="o", linewidth=1.8)
    ax.set_xlabel("Time to maturity elapsed")
    ax.set_ylabel("Asset price boundary")
    ax.set_title("Estimated LSMC early exercise boundary")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_stage2(convergence: pd.DataFrame, basis_comparison: pd.DataFrame) -> dict[str, float | str]:
    """Small machine-readable summary for the main research questions."""

    best_convergence = convergence.sort_values("absolute_error").iloc[0]
    best_basis = basis_comparison.sort_values("absolute_error").iloc[0]
    most_stable_basis = basis_comparison.sort_values("mean_regression_condition").iloc[0]
    largest_path_grid = convergence[convergence["n_steps"] == convergence["n_steps"].max()]
    path_error_correlation = largest_path_grid["n_paths"].corr(largest_path_grid["relative_error"])
    return {
        "best_convergence_n_paths": int(best_convergence["n_paths"]),
        "best_convergence_n_steps": int(best_convergence["n_steps"]),
        "best_convergence_relative_error": float(best_convergence["relative_error"]),
        "best_basis": str(best_basis["basis"]),
        "best_basis_relative_error": float(best_basis["relative_error"]),
        "most_stable_basis": str(most_stable_basis["basis"]),
        "most_stable_basis_mean_condition": float(
            most_stable_basis["mean_regression_condition"]
        ),
        "path_count_error_correlation_at_max_steps": float(path_error_correlation),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage 2 LSMC American option study.")
    parser.add_argument(
        "--convergence-output",
        type=Path,
        default=Path("reports/tables/lsmc_convergence.csv"),
    )
    parser.add_argument(
        "--basis-output",
        type=Path,
        default=Path("reports/tables/lsmc_basis_comparison.csv"),
    )
    parser.add_argument(
        "--boundary-output",
        type=Path,
        default=Path("reports/tables/lsmc_exercise_boundary.csv"),
    )
    parser.add_argument(
        "--convergence-plot",
        type=Path,
        default=Path("reports/figures/lsmc_convergence.png"),
    )
    parser.add_argument(
        "--boundary-plot",
        type=Path,
        default=Path("reports/figures/lsmc_exercise_boundary.png"),
    )
    args = parser.parse_args()

    convergence = run_lsmc_convergence()
    basis_comparison = run_basis_comparison()
    boundary = estimate_exercise_boundary()

    args.convergence_output.parent.mkdir(parents=True, exist_ok=True)
    convergence.to_csv(args.convergence_output, index=False)
    basis_comparison.to_csv(args.basis_output, index=False)
    boundary.to_csv(args.boundary_output, index=False)
    plot_convergence(convergence, args.convergence_plot)
    plot_exercise_boundary(boundary, args.boundary_plot)

    print("LSMC convergence:")
    print(convergence.to_string(index=False))
    print("\nLSMC basis comparison:")
    print(basis_comparison.to_string(index=False))
    print("\nStage 2 summary:")
    print(summarize_stage2(convergence, basis_comparison))
    print(f"\nwrote {args.convergence_output}")
    print(f"wrote {args.basis_output}")
    print(f"wrote {args.boundary_output}")
    print(f"wrote {args.convergence_plot}")
    print(f"wrote {args.boundary_plot}")


if __name__ == "__main__":
    main()
