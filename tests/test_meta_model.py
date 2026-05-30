from pathlib import Path

import numpy as np

from hybrid_american_pricer.experiments.generate_dataset import generate_dataset, write_dataset_outputs
from hybrid_american_pricer.experiments.train_meta_model import (
    NON_FEATURE_COLUMNS,
    load_prepared_data,
    run_meta_model_training,
)
from hybrid_american_pricer.meta.bnn import MCDropoutRegressor


def test_mc_dropout_regressor_predicts_distribution_shape() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 3))
    y = 0.5 * x[:, 0] - x[:, 1]
    model = MCDropoutRegressor(input_dim=3, hidden_sizes=(8,), dropout=0.1, seed=4)
    model.fit(x, y, epochs=5, batch_size=10)
    samples = model.predict_distribution(x[:4], n_samples=7)
    assert samples.shape == (7, 4)


def test_prepared_data_excludes_target_leakage(tmp_path: Path) -> None:
    data = generate_dataset(
        n_samples=12,
        seed=12,
        base_tree_steps=25,
        mc_paths=80,
        lsmc_paths=80,
        rough_paths=80,
        target_rough_paths=100,
        n_steps=6,
        target_n_steps=6,
    )
    write_dataset_outputs(data, output_dir=tmp_path / "processed", figure_path=tmp_path / "fig.png")
    prepared = load_prepared_data(tmp_path / "processed")
    assert NON_FEATURE_COLUMNS.isdisjoint(prepared.feature_columns)
    assert "price_tree" in prepared.feature_columns


def test_run_meta_model_training_writes_outputs(tmp_path: Path) -> None:
    data = generate_dataset(
        n_samples=18,
        seed=13,
        base_tree_steps=25,
        mc_paths=80,
        lsmc_paths=80,
        rough_paths=80,
        target_rough_paths=100,
        n_steps=6,
        target_n_steps=6,
    )
    data_dir = tmp_path / "processed"
    write_dataset_outputs(data, output_dir=data_dir, figure_path=tmp_path / "dataset.png")
    results, predictions, feature_importance, summary = run_meta_model_training(
        data_dir=data_dir,
        output_dir=tmp_path / "tables",
        figure_dir=tmp_path / "figures",
        epochs=5,
        mc_samples=5,
    )
    assert {"linear_regression", "random_forest", "gradient_boosting", "mc_dropout_nn"}.issubset(
        set(results["model"])
    )
    assert not predictions.empty
    assert not feature_importance.empty
    assert "best_meta_model" in summary
    assert (tmp_path / "tables" / "meta_model_results.csv").exists()
    assert (tmp_path / "figures" / "meta_model_prediction_vs_true.png").exists()
