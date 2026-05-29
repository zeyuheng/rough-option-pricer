from __future__ import annotations

import numpy as np

from hybrid_american_pricer.options.instruments import OptionContract


def payoff(spot: np.ndarray | float, contract: OptionContract) -> np.ndarray | float:
    """Return vanilla call or put payoff."""

    values = np.asarray(spot)
    if contract.kind == "call":
        return np.maximum(values - contract.strike, 0.0)
    if contract.kind == "put":
        return np.maximum(contract.strike - values, 0.0)
    raise ValueError(f"unsupported option kind: {contract.kind}")

