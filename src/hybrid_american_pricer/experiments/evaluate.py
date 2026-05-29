from __future__ import annotations

import numpy as np

from hybrid_american_pricer.meta.calibration import coverage, mean_interval_width
from hybrid_american_pricer.utils.metrics import summarize_errors


def main() -> None:
    y_true = np.array([10.0, 12.0, 9.0])
    y_pred = np.array([10.5, 11.2, 9.4])
    lower = y_pred - 1.0
    upper = y_pred + 1.0
    print(summarize_errors(y_true, y_pred))
    print({"coverage": coverage(y_true, lower, upper), "width": mean_interval_width(lower, upper)})


if __name__ == "__main__":
    main()

