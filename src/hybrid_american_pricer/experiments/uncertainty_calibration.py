from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from hybrid_american_pricer.experiments.train_meta_model import (
    RESIDUAL_ANCHOR_COLUMN,
    PreparedData,
    load_prepared_data,
)
from hybrid_american_pricer.meta.bnn import MCDropoutRegressor
from hybrid_american_pricer.meta.calibration import (
    conformal_interval,
    coverage,
    mean_interval_width,
)
from hybrid_american_pricer.utils.metrics import rmse


@dataclass(frozen=True)
class DropoutIntervalModel:
    model: MCDropoutRegressor
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    residual_blend_alpha: float


def train_dropout_interval_model(
    data: PreparedData,
    epochs: int = 400,
    mc_samples: int = 200,
    seed: int = 42,
) -> DropoutIntervalModel:
    """Train MC-dropout residual model and tune residual blend on validation RMSE."""

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(data.x_train)
    train_residual = data.y_train.to_numpy() - data.x_train[RESIDUAL_ANCHOR_COLUMN].to_numpy()
    y_train = y_scaler.fit_transform(train_residual.reshape(-1, 1)).ravel()
    model = MCDropoutRegressor(
        input_dim=x_train.shape[1],
        hidden_sizes=(64, 32),
        dropout=0.10,
        learning_rate=2e-3,
        seed=seed,
    )
    model.fit(x_train, y_train, epochs=epochs, batch_size=32)

    valid_samples = residual_distribution(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        x=data.x_valid,
        mc_samples=mc_samples,
    )
    valid_mean_residual = valid_samples.mean(axis=0)
    valid_anchor = data.x_valid[RESIDUAL_ANCHOR_COLUMN].to_numpy()
    alphas = np.linspace(0.0, 1.0, 21)
    best_alpha = min(
        alphas,
        key=lambda alpha: rmse(
            data.y_valid.to_numpy(),
            np.maximum(valid_anchor + alpha * valid_mean_residual, 0.0),
        ),
    )
    return DropoutIntervalModel(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        residual_blend_alpha=float(best_alpha),
    )


def residual_distribution(
    model: MCDropoutRegressor,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    x: pd.DataFrame,
    mc_samples: int,
) -> np.ndarray:
    """Return residual samples in original price units."""

    x_scaled = x_scaler.transform(x)
    scaled_samples = model.predict_distribution(x_scaled, n_samples=mc_samples)
    flat = scaled_samples.reshape(-1, 1)
    unscaled = y_scaler.inverse_transform(flat).reshape(scaled_samples.shape)
    return unscaled


def price_distribution(
    interval_model: DropoutIntervalModel,
    x: pd.DataFrame,
    mc_samples: int = 200,
) -> np.ndarray:
    """Convert dropout residual samples into price samples."""

    residual_samples = residual_distribution(
        model=interval_model.model,
        x_scaler=interval_model.x_scaler,
        y_scaler=interval_model.y_scaler,
        x=x,
        mc_samples=mc_samples,
    )
    anchor = x[RESIDUAL_ANCHOR_COLUMN].to_numpy()
    prices = anchor[None, :] + interval_model.residual_blend_alpha * residual_samples
    return np.maximum(prices, 0.0)


def interval_from_samples(
    samples: np.ndarray,
    nominal: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean prediction and central interval from predictive samples."""

    alpha = 1.0 - nominal
    mean = samples.mean(axis=0)
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    return mean, lower, upper


def interval_metrics(
    model: str,
    nominal: float,
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    qhat: float = 0.0,
) -> dict[str, float | str]:
    return {
        "model": model,
        "nominal": nominal,
        "empirical_coverage": coverage(y_true, lower, upper),
        "interval_width": mean_interval_width(lower, upper),
        "qhat": qhat,
    }


def run_uncertainty_calibration(
    data_dir: Path = Path("data/processed"),
    output_dir: Path = Path("reports/tables"),
    figure_dir: Path = Path("reports/figures"),
    epochs: int = 400,
    mc_samples: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train MC-dropout intervals and conformal calibration, then write artifacts."""

    data = load_prepared_data(data_dir)
    interval_model = train_dropout_interval_model(
        data=data,
        epochs=epochs,
        mc_samples=mc_samples,
    )
    valid_samples = price_distribution(interval_model, data.x_valid, mc_samples=mc_samples)
    test_samples = price_distribution(interval_model, data.x_test, mc_samples=mc_samples)

    valid_mean, valid_lower, valid_upper = interval_from_samples(valid_samples, nominal=0.95)
    test_mean, test_lower, test_upper = interval_from_samples(test_samples, nominal=0.95)
    calibrated_lower, calibrated_upper, qhat = conformal_interval(
        data.y_valid.to_numpy(),
        valid_lower,
        valid_upper,
        test_lower,
        test_upper,
        alpha=0.05,
    )

    rows = [
        interval_metrics(
            "mc_dropout_raw",
            0.95,
            data.y_test.to_numpy(),
            test_lower,
            test_upper,
        ),
        interval_metrics(
            "mc_dropout_conformal",
            0.95,
            data.y_test.to_numpy(),
            calibrated_lower,
            calibrated_upper,
            qhat=qhat,
        ),
    ]
    uncertainty_results = pd.DataFrame(rows)

    predictions = pd.DataFrame(
        {
            "case_id": pd.read_csv(data_dir / "test.csv")["case_id"],
            "y_true": data.y_test.to_numpy(),
            "y_pred": test_mean,
            "raw_lower_95": test_lower,
            "raw_upper_95": test_upper,
            "conformal_lower_95": calibrated_lower,
            "conformal_upper_95": calibrated_upper,
            "residual_blend_alpha": interval_model.residual_blend_alpha,
        }
    )
    calibration_curve = calibration_curve_table(
        data.y_valid.to_numpy(),
        valid_samples,
        data.y_test.to_numpy(),
        test_samples,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_results.to_csv(output_dir / "uncertainty_results.csv", index=False)
    predictions.to_csv(output_dir / "uncertainty_predictions.csv", index=False)
    calibration_curve.to_csv(output_dir / "uncertainty_calibration_curve.csv", index=False)
    plot_calibration_curve(calibration_curve, figure_dir / "uncertainty_calibration_curve.png")
    plot_interval_widths(uncertainty_results, figure_dir / "uncertainty_interval_widths.png")
    plot_calibrated_intervals(
        predictions,
        figure_dir / "uncertainty_calibrated_intervals.png",
    )
    return uncertainty_results, predictions, calibration_curve


def calibration_curve_table(
    y_valid: np.ndarray,
    valid_samples: np.ndarray,
    y_test: np.ndarray,
    test_samples: np.ndarray,
    nominals: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95),
) -> pd.DataFrame:
    """Build raw and conformal coverage table across nominal interval levels."""

    rows = []
    for nominal in nominals:
        alpha = 1.0 - nominal
        _, valid_lower, valid_upper = interval_from_samples(valid_samples, nominal)
        _, test_lower, test_upper = interval_from_samples(test_samples, nominal)
        calibrated_lower, calibrated_upper, qhat = conformal_interval(
            y_valid,
            valid_lower,
            valid_upper,
            test_lower,
            test_upper,
            alpha=alpha,
        )
        rows.append(
            interval_metrics("mc_dropout_raw", nominal, y_test, test_lower, test_upper)
        )
        rows.append(
            interval_metrics(
                "mc_dropout_conformal",
                nominal,
                y_test,
                calibrated_lower,
                calibrated_upper,
                qhat=qhat,
            )
        )
    return pd.DataFrame(rows)


def plot_calibration_curve(curve: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for model, group in curve.groupby("model"):
        ordered = group.sort_values("nominal")
        ax.plot(
            ordered["nominal"],
            ordered["empirical_coverage"],
            marker="o",
            label=model,
        )
    ax.plot([0.45, 1.0], [0.45, 1.0], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Prediction interval calibration")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_interval_widths(results: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(results["model"], results["interval_width"], color=["#2563eb", "#dc2626"])
    ax.set_ylabel("Mean interval width")
    ax.set_title("95% interval width before and after conformal calibration")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_calibrated_intervals(predictions: pd.DataFrame, output_path: Path) -> None:
    ordered = predictions.sort_values("y_true").reset_index(drop=True)
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, ordered["y_true"], color="black", marker="o", linewidth=1.3, label="target")
    ax.plot(x, ordered["y_pred"], color="#2563eb", linewidth=1.2, label="prediction")
    ax.fill_between(
        x,
        ordered["raw_lower_95"],
        ordered["raw_upper_95"],
        color="#2563eb",
        alpha=0.18,
        label="raw 95%",
    )
    ax.fill_between(
        x,
        ordered["conformal_lower_95"],
        ordered["conformal_upper_95"],
        color="#dc2626",
        alpha=0.16,
        label="conformal 95%",
    )
    ax.set_xlabel("Test sample sorted by target price")
    ax.set_ylabel("Option price")
    ax.set_title("MC Dropout intervals before and after conformal calibration")
    ax.legend()
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage 6 uncertainty calibration.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--mc-samples", type=int, default=200)
    args = parser.parse_args()

    results, _, curve = run_uncertainty_calibration(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        epochs=args.epochs,
        mc_samples=args.mc_samples,
    )
    print("Uncertainty results:")
    print(results.to_string(index=False))
    print("\nCalibration curve:")
    print(curve.to_string(index=False))
    print(f"\nwrote {args.output_dir / 'uncertainty_results.csv'}")
    print(f"wrote {args.output_dir / 'uncertainty_predictions.csv'}")
    print(f"wrote {args.output_dir / 'uncertainty_calibration_curve.csv'}")
    print(f"wrote {args.figure_dir / 'uncertainty_calibration_curve.png'}")
    print(f"wrote {args.figure_dir / 'uncertainty_interval_widths.png'}")
    print(f"wrote {args.figure_dir / 'uncertainty_calibrated_intervals.png'}")


if __name__ == "__main__":
    main()

