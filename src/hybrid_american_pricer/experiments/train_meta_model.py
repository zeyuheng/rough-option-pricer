from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hybrid_american_pricer.meta.bnn import MCDropoutRegressor
from hybrid_american_pricer.utils.metrics import mae, mape, rmse


TARGET_COLUMN = "target_price"
NON_FEATURE_COLUMNS = {
    "case_id",
    "target_method",
    "target_price",
    "target_runtime_seconds",
    "target_std_error",
    "tree_discretization_gap",
    "target_minus_low_path_rough_lsmc",
    "tree_vs_rough_target_gap",
    "bid",
    "ask",
    "mid_price",
    "spread",
    "relative_spread",
    "market_iv",
    "model_iv",
    "iv_error",
    "market_iv_clipped",
    "model_iv_clipped",
}
RESIDUAL_ANCHOR_COLUMN = "price_rough_bergomi_lsmc"
BASE_PRICER_COLUMNS = [
    "price_black_scholes",
    "price_tree",
    "price_monte_carlo",
    "price_lsmc",
    "price_rough_bergomi_mc",
    "price_rough_bergomi_lsmc",
]


@dataclass(frozen=True)
class PreparedData:
    feature_columns: list[str]
    x_train: pd.DataFrame
    y_train: pd.Series
    x_valid: pd.DataFrame
    y_valid: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


def load_prepared_data(data_dir: Path = Path("data/processed")) -> PreparedData:
    """Load stage-4 splits and keep numeric meta-model features only."""

    train = pd.read_csv(data_dir / "train.csv")
    valid = pd.read_csv(data_dir / "valid.csv")
    test = pd.read_csv(data_dir / "test.csv")
    feature_columns = [
        column
        for column in train.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(train[column])
    ]
    return PreparedData(
        feature_columns=feature_columns,
        x_train=train[feature_columns],
        y_train=train[TARGET_COLUMN],
        x_valid=valid[feature_columns],
        y_valid=valid[TARGET_COLUMN],
        x_test=test[feature_columns],
        y_test=test[TARGET_COLUMN],
    )


def evaluate_predictions(
    model_name: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mape_floor: float = 1.0,
) -> dict[str, float | str]:
    """Return standard point-prediction metrics."""

    return {
        "model": model_name,
        "split": split,
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred, eps=mape_floor),
        "mape_floor": mape_floor,
    }


def base_pricer_baselines(data: PreparedData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate single-pricer columns as naive baselines."""

    rows = []
    predictions = []
    for column in BASE_PRICER_COLUMNS:
        if column not in data.feature_columns:
            continue
        for split, x, y in [
            ("valid", data.x_valid, data.y_valid),
            ("test", data.x_test, data.y_test),
        ]:
            pred = x[column].to_numpy()
            rows.append(evaluate_predictions(column, split, y.to_numpy(), pred))
            predictions.append(
                pd.DataFrame(
                    {
                        "model": column,
                        "split": split,
                        "y_true": y.to_numpy(),
                        "y_pred": pred,
                        "residual": pred - y.to_numpy(),
                    }
                )
            )
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def train_sklearn_models(data: PreparedData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train classical meta-model baselines."""

    models = {
        "linear_regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=3,
            random_state=42,
        ),
    }
    rows = []
    predictions = []
    importances = []

    for name, model in models.items():
        model.fit(data.x_train, data.y_train)
        for split, x, y in [
            ("valid", data.x_valid, data.y_valid),
            ("test", data.x_test, data.y_test),
        ]:
            pred = model.predict(x)
            rows.append(evaluate_predictions(name, split, y.to_numpy(), pred))
            predictions.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "split": split,
                        "y_true": y.to_numpy(),
                        "y_pred": pred,
                        "residual": pred - y.to_numpy(),
                    }
                )
            )

        raw_model = model.named_steps["model"] if isinstance(model, Pipeline) else model
        if hasattr(raw_model, "feature_importances_"):
            for feature, importance in zip(
                data.feature_columns,
                raw_model.feature_importances_,
                strict=True,
            ):
                importances.append(
                    {"model": name, "feature": feature, "importance": float(importance)}
                )
        elif hasattr(raw_model, "coef_"):
            for feature, coefficient in zip(data.feature_columns, raw_model.coef_, strict=True):
                importances.append(
                    {
                        "model": name,
                        "feature": feature,
                        "importance": float(abs(coefficient)),
                        "coefficient": float(coefficient),
                    }
                )

    return (
        pd.DataFrame(rows),
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(importances).sort_values(["model", "importance"], ascending=[True, False]),
    )


def train_mc_dropout(
    data: PreparedData,
    epochs: int = 400,
    mc_samples: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train the NumPy MC-dropout neural meta-model as a residual corrector."""

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
        seed=42,
    )
    model.fit(x_train, y_train, epochs=epochs, batch_size=32)

    valid_scaled = x_scaler.transform(data.x_valid)
    valid_residual_scaled = model.predict(valid_scaled, n_samples=mc_samples)
    valid_residual = y_scaler.inverse_transform(valid_residual_scaled.reshape(-1, 1)).ravel()
    valid_anchor = data.x_valid[RESIDUAL_ANCHOR_COLUMN].to_numpy()
    alphas = np.linspace(0.0, 1.0, 21)
    best_alpha = min(
        alphas,
        key=lambda alpha: rmse(
            data.y_valid.to_numpy(),
            np.maximum(valid_anchor + alpha * valid_residual, 0.0),
        ),
    )

    rows = []
    predictions = []
    for split, x, y in [
        ("valid", data.x_valid, data.y_valid),
        ("test", data.x_test, data.y_test),
    ]:
        x_scaled = x_scaler.transform(x)
        pred_scaled = model.predict(x_scaled, n_samples=mc_samples)
        residual = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        pred = np.maximum(x[RESIDUAL_ANCHOR_COLUMN].to_numpy() + best_alpha * residual, 0.0)
        row = evaluate_predictions("mc_dropout_nn", split, y.to_numpy(), pred)
        row["residual_blend_alpha"] = float(best_alpha)
        rows.append(row)
        predictions.append(
            pd.DataFrame(
                {
                    "model": "mc_dropout_nn",
                    "split": split,
                    "y_true": y.to_numpy(),
                    "y_pred": pred,
                    "residual": pred - y.to_numpy(),
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def answer_stage5_questions(
    results: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> dict[str, float | str | bool]:
    """Summarize whether meta-models beat single-pricer baselines."""

    test = results[results["split"] == "test"].copy()
    single_pricers = test[test["model"].str.startswith("price_")]
    meta_models = test[~test["model"].str.startswith("price_")]
    best_single = single_pricers.sort_values("mape").iloc[0]
    best_meta = meta_models.sort_values("mape").iloc[0]
    top_feature = (
        feature_importance[feature_importance["model"].isin(["random_forest", "gradient_boosting"])]
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .iloc[0]
    )
    return {
        "best_single_pricer": str(best_single["model"]),
        "best_single_pricer_test_mape": float(best_single["mape"]),
        "best_meta_model": str(best_meta["model"]),
        "best_meta_model_test_mape": float(best_meta["mape"]),
        "meta_model_beats_best_single_pricer": bool(best_meta["mape"] < best_single["mape"]),
        "target_mape_11_1_percent_met": bool(best_meta["mape"] <= 11.1),
        "top_feature": str(top_feature["feature"]),
    }


def plot_prediction_vs_true(predictions: pd.DataFrame, output_path: Path) -> None:
    test = predictions[predictions["split"] == "test"]
    models = sorted(test["model"].unique())
    fig, axes = plt.subplots(2, int(np.ceil(len(models) / 2)), figsize=(13, 8), squeeze=False)
    min_price = min(test["y_true"].min(), test["y_pred"].min())
    max_price = max(test["y_true"].max(), test["y_pred"].max())
    for ax, model in zip(axes.ravel(), models, strict=False):
        subset = test[test["model"] == model]
        ax.scatter(subset["y_true"], subset["y_pred"], s=20, alpha=0.75)
        ax.plot([min_price, max_price], [min_price, max_price], color="black", linewidth=1)
        ax.set_title(model)
        ax.set_xlabel("True target price")
        ax.set_ylabel("Predicted price")
        ax.grid(alpha=0.2)
    for ax in axes.ravel()[len(models) :]:
        ax.axis("off")
    fig.suptitle("Meta-model predictions vs true prices", y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(predictions: pd.DataFrame, output_path: Path) -> None:
    test = predictions[predictions["split"] == "test"]
    fig, ax = plt.subplots(figsize=(9, 5))
    model_order = (
        test.groupby("model")["residual"]
        .apply(lambda x: float(np.mean(np.abs(x))))
        .sort_values()
        .index.tolist()
    )
    data = [test[test["model"] == model]["residual"].to_numpy() for model in model_order]
    ax.boxplot(data, tick_labels=model_order, showfliers=False)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Prediction residual")
    ax.set_title("Test residual distribution by model")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_meta_model_training(
    data_dir: Path = Path("data/processed"),
    output_dir: Path = Path("reports/tables"),
    figure_dir: Path = Path("reports/figures"),
    epochs: int = 400,
    mc_samples: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float | str | bool]]:
    data = load_prepared_data(data_dir)
    baseline_results, baseline_predictions = base_pricer_baselines(data)
    sklearn_results, sklearn_predictions, feature_importance = train_sklearn_models(data)
    dropout_results, dropout_predictions = train_mc_dropout(data, epochs=epochs, mc_samples=mc_samples)

    results = pd.concat([baseline_results, sklearn_results, dropout_results], ignore_index=True)
    predictions = pd.concat(
        [baseline_predictions, sklearn_predictions, dropout_predictions],
        ignore_index=True,
    )
    summary = answer_stage5_questions(results, feature_importance)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "meta_model_results.csv", index=False)
    predictions.to_csv(output_dir / "meta_model_predictions.csv", index=False)
    feature_importance.to_csv(output_dir / "meta_model_feature_importance.csv", index=False)
    plot_prediction_vs_true(predictions, figure_dir / "meta_model_prediction_vs_true.png")
    plot_residuals(predictions, figure_dir / "meta_model_residuals.png")
    return results, predictions, feature_importance, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stage-5 hybrid meta-pricing models.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--figure-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--mc-samples", type=int, default=100)
    args = parser.parse_args()

    results, _, feature_importance, summary = run_meta_model_training(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        epochs=args.epochs,
        mc_samples=args.mc_samples,
    )
    print("Meta-model results:")
    print(results.sort_values(["split", "mape"]).to_string(index=False))
    print("\nTop feature importances:")
    print(feature_importance.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False).head(10).to_string(index=False))
    print("\nStage 5 summary:")
    print(summary)
    print(f"\nwrote {args.output_dir / 'meta_model_results.csv'}")
    print(f"wrote {args.output_dir / 'meta_model_predictions.csv'}")
    print(f"wrote {args.output_dir / 'meta_model_feature_importance.csv'}")
    print(f"wrote {args.figure_dir / 'meta_model_prediction_vs_true.png'}")
    print(f"wrote {args.figure_dir / 'meta_model_residuals.png'}")


if __name__ == "__main__":
    main()
