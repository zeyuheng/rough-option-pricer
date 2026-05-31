from pathlib import Path

import numpy as np

from hybrid_american_pricer.experiments.generate_dataset import generate_dataset, write_dataset_outputs
from hybrid_american_pricer.experiments.uncertainty_calibration import (
    calibration_curve_table,
    interval_from_samples,
    run_uncertainty_calibration,
)
from hybrid_american_pricer.meta.calibration import conformal_interval, coverage


def test_conformal_interval_handles_small_calibration_sets() -> None:
    y_cal = np.array([1.0, 2.0])
    lower_cal = np.array([0.9, 1.9])
    upper_cal = np.array([1.1, 2.1])
    lower, upper, qhat = conformal_interval(
        y_cal,
        lower_cal,
        upper_cal,
        np.array([0.8, 1.8]),
        np.array([1.2, 2.2]),
        alpha=0.05,
    )
    assert qhat >= 0
    assert (upper >= lower).all()


def test_interval_from_samples_has_expected_shape() -> None:
    samples = np.array([[1.0, 2.0], [1.2, 2.2], [0.8, 1.8]])
    mean, lower, upper = interval_from_samples(samples, nominal=0.90)
    assert mean.shape == lower.shape == upper.shape == (2,)
    assert coverage(np.array([1.0, 2.0]), lower, upper) == 1.0


def test_calibration_curve_table_contains_raw_and_conformal() -> None:
    rng = np.random.default_rng(3)
    valid_samples = rng.normal(size=(20, 5))
    test_samples = rng.normal(size=(20, 6))
    curve = calibration_curve_table(
        np.zeros(5),
        valid_samples,
        np.zeros(6),
        test_samples,
        nominals=(0.5, 0.9),
    )
    assert {"mc_dropout_raw", "mc_dropout_conformal"} == set(curve["model"])
    assert {0.5, 0.9} == set(curve["nominal"])


def test_run_uncertainty_calibration_writes_outputs(tmp_path: Path) -> None:
    data = generate_dataset(
        n_samples=20,
        seed=20,
        base_tree_steps=20,
        mc_paths=60,
        lsmc_paths=60,
        rough_paths=60,
        target_rough_paths=80,
        n_steps=5,
        target_n_steps=5,
    )
    data_dir = tmp_path / "processed"
    write_dataset_outputs(data, output_dir=data_dir, figure_path=tmp_path / "dataset.png")
    results, predictions, curve = run_uncertainty_calibration(
        data_dir=data_dir,
        output_dir=tmp_path / "tables",
        figure_dir=tmp_path / "figures",
        epochs=5,
        mc_samples=10,
    )
    assert {"mc_dropout_raw", "mc_dropout_conformal"} == set(results["model"])
    assert not predictions.empty
    assert not curve.empty
    assert (tmp_path / "tables" / "uncertainty_results.csv").exists()
    assert (tmp_path / "figures" / "uncertainty_calibration_curve.png").exists()

