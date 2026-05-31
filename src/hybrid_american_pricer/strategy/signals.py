from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hybrid_american_pricer.strategy.fair_value import FairValueConfig, FairValueEngine, FairValueModelName


@dataclass(frozen=True)
class SignalConfig:
    min_abs_edge: float = 0.05
    min_relative_edge: float = 0.03
    max_relative_spread: float = 0.18
    min_volume: float = 20.0
    min_open_interest: float = 100.0
    min_mid_price: float = 0.10
    min_days_to_expiry: int = 7
    fair_value_model: FairValueModelName = "binomial"
    tree_steps: int = 300
    use_rough_price: bool = False
    rough_paths: int = 1500
    rough_steps: int = 40
    hurst: float = 0.1
    eta: float = 1.8
    rho: float = -0.7


def _liquidity_pass(row: pd.Series, config: SignalConfig) -> bool:
    return bool(
        row["relative_spread"] <= config.max_relative_spread
        and row["volume"] >= config.min_volume
        and row["open_interest"] >= config.min_open_interest
        and row["mid_price"] >= config.min_mid_price
        and row["days_to_expiry"] >= config.min_days_to_expiry
    )


def _signal(row: pd.Series, config: SignalConfig) -> str:
    if not row["liquidity_pass"]:
        return "no_trade_liquidity"
    if (
        row["buy_edge"] >= config.min_abs_edge
        and row["relative_buy_edge"] >= config.min_relative_edge
    ):
        return "buy_underpriced"
    if (
        row["sell_edge"] >= config.min_abs_edge
        and row["relative_sell_edge"] >= config.min_relative_edge
    ):
        return "sell_overpriced"
    return "hold"


def _confidence(row: pd.Series, config: SignalConfig) -> float:
    best_edge = max(float(row["buy_edge"]), float(row["sell_edge"]), 0.0)
    liquidity_haircut = min(float(row["relative_spread"]) / max(config.max_relative_spread, 1e-12), 1.0)
    raw_score = best_edge / max(float(row["mid_price"]), 1.0)
    return float(np.clip(raw_score * (1.0 - 0.5 * liquidity_haircut), 0.0, 1.0))


def build_mispricing_signals(
    option_chain: pd.DataFrame,
    config: SignalConfig | None = None,
) -> pd.DataFrame:
    """Reprice a standardized option chain and convert model-vs-market gaps into signals."""

    config = config or SignalConfig()
    model_name = "rough" if config.use_rough_price else config.fair_value_model
    fair_value_engine = FairValueEngine(
        FairValueConfig(
            model=model_name,
            tree_steps=config.tree_steps,
            rough_paths=config.rough_paths,
            rough_steps=config.rough_steps,
            hurst=config.hurst,
            eta=config.eta,
            rho=config.rho,
        )
    )
    rows = []

    for _, row in option_chain.iterrows():
        fair_value_fields = fair_value_engine.price_row(row)
        fair_value = float(fair_value_fields["fair_value"])

        enriched = row.to_dict()
        enriched.update(
            {
                **fair_value_fields,
                "model_minus_mid": fair_value - float(row["mid_price"]),
                "buy_edge": fair_value - float(row["ask"]),
                "sell_edge": float(row["bid"]) - fair_value,
                "relative_buy_edge": (fair_value - float(row["ask"])) / max(float(row["mid_price"]), 1e-12),
                "relative_sell_edge": (float(row["bid"]) - fair_value) / max(float(row["mid_price"]), 1e-12),
            }
        )
        rows.append(enriched)

    signals = pd.DataFrame(rows)
    signals["liquidity_pass"] = signals.apply(_liquidity_pass, axis=1, config=config)
    signals["signal"] = signals.apply(_signal, axis=1, config=config)
    signals["signal_confidence"] = signals.apply(_confidence, axis=1, config=config)
    return signals.sort_values("signal_confidence", ascending=False).reset_index(drop=True)
