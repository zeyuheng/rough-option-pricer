from __future__ import annotations

import numpy as np


def conformal_interval(
    y_calibration: np.ndarray,
    lower_calibration: np.ndarray,
    upper_calibration: np.ndarray,
    lower_test: np.ndarray,
    upper_test: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply split conformal correction to prediction intervals."""

    misses = np.maximum(lower_calibration - y_calibration, y_calibration - upper_calibration)
    qhat = float(np.quantile(misses, np.ceil((len(misses) + 1) * (1 - alpha)) / len(misses)))
    return lower_test - qhat, upper_test + qhat, qhat


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((lower <= y_true) & (y_true <= upper)))


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))

