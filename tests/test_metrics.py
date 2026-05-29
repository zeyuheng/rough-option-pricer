import numpy as np

from hybrid_american_pricer.utils.metrics import mape, summarize_errors


def test_metrics_are_finite() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    assert mape(y_true, y_pred) > 0
    assert set(summarize_errors(y_true, y_pred)) == {"mae", "rmse", "mape"}

