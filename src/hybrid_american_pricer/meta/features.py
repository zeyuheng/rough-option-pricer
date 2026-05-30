from __future__ import annotations

import math
from collections.abc import Mapping

from hybrid_american_pricer.options.instruments import MarketState, OptionContract


def maturity_bucket(maturity: float) -> str:
    """Bucket time-to-maturity into coarse market regimes."""

    if maturity <= 0.5:
        return "short"
    if maturity <= 1.25:
        return "medium"
    return "long"


def build_feature_row(
    contract: OptionContract,
    market: MarketState,
    base_prices: Mapping[str, float],
    rough_params: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Build one meta-model feature row from contract, market, and base pricers."""

    rough_params = rough_params or {}
    bucket = maturity_bucket(contract.maturity)
    moneyness = market.spot / contract.strike
    row = {
        "spot": market.spot,
        "strike": contract.strike,
        "maturity": contract.maturity,
        "rate": market.rate,
        "dividend": market.dividend,
        "volatility": market.volatility,
        "moneyness": moneyness,
        "log_moneyness": math.log(moneyness),
        "maturity_bucket_short": 1.0 if bucket == "short" else 0.0,
        "maturity_bucket_medium": 1.0 if bucket == "medium" else 0.0,
        "maturity_bucket_long": 1.0 if bucket == "long" else 0.0,
        "is_put": 1.0 if contract.kind == "put" else 0.0,
        "is_american": 1.0 if contract.exercise == "american" else 0.0,
        "hurst": rough_params.get("hurst", 0.1),
        "eta": rough_params.get("eta", 1.8),
        "rho": rough_params.get("rho", -0.7),
    }
    for name, value in base_prices.items():
        row[f"price_{name}"] = float(value)
    if "tree" in base_prices and "black_scholes" in base_prices:
        row["early_exercise_premium"] = base_prices["tree"] - base_prices["black_scholes"]
    return row
