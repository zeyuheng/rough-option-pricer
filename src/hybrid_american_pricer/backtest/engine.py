from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from hybrid_american_pricer.strategy.signals import SignalConfig, build_mispricing_signals


CONTRACT_KEYS = ["ticker", "expiration", "strike", "option_kind"]


@dataclass(frozen=True)
class BacktestConfig:
    name: str = "baseline"
    holding_days: int = 5
    max_trades_per_day: int = 5
    min_moneyness: float = 0.95
    max_moneyness: float = 1.05
    min_days_to_expiry: int = 21
    max_days_to_expiry: int = 45
    option_kind: str = "both"
    min_signal_confidence: float = 0.0
    confidence_top_fraction: float = 1.0
    trend_filter: bool = False
    max_market_iv: float | None = None
    take_profit: float | None = None
    stop_loss: float | None = None
    exit_policy: Literal["fixed_horizon", "stop_or_horizon"] = "fixed_horizon"
    allowed_tickers: tuple[str, ...] = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA")
    option_multiplier: float = 100.0
    initial_capital: float = 100_000.0
    signal_config: SignalConfig = SignalConfig(
        min_abs_edge=0.05,
        min_relative_edge=0.03,
        max_relative_spread=0.15,
        min_volume=100.0,
        min_open_interest=500.0,
        min_mid_price=0.10,
        min_days_to_expiry=21,
        tree_steps=250,
    )


def _to_datetime_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column]).dt.normalize()


def _candidate_mask(signals: pd.DataFrame, config: BacktestConfig) -> pd.Series:
    mask = (
        (signals["signal"] == "buy_underpriced")
        & (signals["liquidity_pass"])
        & (signals["ticker"].isin(config.allowed_tickers))
        & (signals["moneyness"].between(config.min_moneyness, config.max_moneyness))
        & (signals["days_to_expiry"].between(config.min_days_to_expiry, config.max_days_to_expiry))
        & (signals["signal_confidence"] >= config.min_signal_confidence)
    )
    if config.option_kind in {"call", "put"}:
        mask = mask & (signals["option_kind"] == config.option_kind)
    if config.max_market_iv is not None:
        mask = mask & (signals["market_iv"] <= config.max_market_iv)
    if config.trend_filter and "underlying_trend" in signals.columns:
        call_trend = (signals["option_kind"] == "call") & (signals["underlying_trend"] > 0.0)
        put_trend = (signals["option_kind"] == "put") & (signals["underlying_trend"] < 0.0)
        mask = mask & (call_trend | put_trend)
    return mask


def _apply_top_confidence(candidates: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    if candidates.empty or config.confidence_top_fraction >= 1.0:
        return candidates
    keep = max(1, int(len(candidates) * config.confidence_top_fraction))
    return candidates.sort_values("signal_confidence", ascending=False).head(keep)


def _add_underlying_trend(history: pd.DataFrame) -> pd.DataFrame:
    snapshots = (
        history.groupby(["ticker", "quote_date_ts"], as_index=False)["underlying_price"]
        .first()
        .sort_values(["ticker", "quote_date_ts"])
    )
    snapshots["underlying_trend"] = snapshots.groupby("ticker")["underlying_price"].pct_change()
    return history.merge(
        snapshots[["ticker", "quote_date_ts", "underlying_trend"]],
        on=["ticker", "quote_date_ts"],
        how="left",
    ).fillna({"underlying_trend": 0.0})


def _find_exit_snapshot(
    history: pd.DataFrame,
    entry: pd.Series,
    available_dates: list[pd.Timestamp],
    config: BacktestConfig,
) -> pd.Series | None:
    entry_date = pd.Timestamp(entry["quote_date_ts"])
    expiration = pd.Timestamp(entry["expiration"])
    target_exit = min(entry_date + pd.Timedelta(days=config.holding_days), expiration)
    candidate_dates = [date for date in available_dates if entry_date < date <= target_exit]
    if not candidate_dates:
        candidate_dates = [date for date in available_dates if date > entry_date and date < expiration]
    if not candidate_dates:
        return None
    for exit_date in candidate_dates:
        same_contract = history[
            (history["quote_date_ts"] == exit_date)
            & (history["ticker"] == entry["ticker"])
            & (history["expiration"] == entry["expiration"])
            & (history["strike"] == entry["strike"])
            & (history["option_kind"] == entry["option_kind"])
        ]
        if same_contract.empty:
            continue
        row = same_contract.iloc[0]
        if config.exit_policy == "stop_or_horizon":
            current_return = (float(row["bid"]) - float(entry["ask"])) / max(float(entry["ask"]), 1e-12)
            if config.take_profit is not None and current_return >= config.take_profit:
                return row
            if config.stop_loss is not None and current_return <= -abs(config.stop_loss):
                return row
        if exit_date == candidate_dates[-1]:
            return row
    return None


def run_long_mispricing_backtest(
    history: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Backtest long-only buys of underpriced options using ask entry and bid exit."""

    config = config or BacktestConfig()
    if history.empty:
        return pd.DataFrame()

    market_history = history.copy()
    market_history["quote_date_ts"] = _to_datetime_series(market_history, "quote_date")
    market_history["expiration"] = _to_datetime_series(market_history, "expiration")
    market_history = _add_underlying_trend(market_history)
    available_dates = sorted(market_history["quote_date_ts"].drop_duplicates().tolist())
    trades: list[dict[str, object]] = []

    for quote_date, snapshot in market_history.groupby("quote_date_ts", sort=True):
        signals = build_mispricing_signals(snapshot.drop(columns=["quote_date_ts"]), config.signal_config)
        signals["quote_date_ts"] = quote_date
        signals["expiration"] = _to_datetime_series(signals, "expiration")
        candidates = _apply_top_confidence(signals[_candidate_mask(signals, config)], config).head(
            config.max_trades_per_day
        )

        for _, entry in candidates.iterrows():
            exit_row = _find_exit_snapshot(market_history, entry, available_dates, config)
            if exit_row is None:
                continue

            entry_price = float(entry["ask"])
            exit_price = float(exit_row["bid"])
            pnl_per_contract = exit_price - entry_price
            trade_return = pnl_per_contract / max(entry_price, 1e-12)
            trades.append(
                {
                    "entry_date": pd.Timestamp(entry["quote_date_ts"]).date().isoformat(),
                    "exit_date": pd.Timestamp(exit_row["quote_date_ts"]).date().isoformat(),
                    "ticker": entry["ticker"],
                    "expiration": pd.Timestamp(entry["expiration"]).date().isoformat(),
                    "strike": float(entry["strike"]),
                    "option_kind": entry["option_kind"],
                    "entry_ask": entry_price,
                    "exit_bid": exit_price,
                    "entry_mid": float(entry["mid_price"]),
                    "exit_mid": float(exit_row["mid_price"]),
                    "fair_value_entry": float(entry["fair_value"]),
                    "buy_edge_entry": float(entry["buy_edge"]),
                    "relative_buy_edge_entry": float(entry["relative_buy_edge"]),
                    "signal_confidence": float(entry["signal_confidence"]),
                    "holding_days": int(
                        (pd.Timestamp(exit_row["quote_date_ts"]) - pd.Timestamp(entry["quote_date_ts"])).days
                    ),
                    "pnl_per_contract": pnl_per_contract,
                    "pnl_dollars": pnl_per_contract * config.option_multiplier,
                    "return": trade_return,
                    "strategy_name": config.name,
                    "fair_value_source": entry["fair_value_source"],
                    "underlying_trend_entry": float(entry.get("underlying_trend", 0.0)),
                    "market_iv_entry": float(entry["market_iv"]),
                }
            )

    return pd.DataFrame(trades)
