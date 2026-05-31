from __future__ import annotations

import math

import numpy as np
import pandas as pd


def build_equity_curve(trades: pd.DataFrame, initial_capital: float = 100_000.0) -> pd.DataFrame:
    """Build a simple realized-PnL equity curve indexed by trade exit date."""

    if trades.empty:
        return pd.DataFrame(columns=["date", "daily_pnl", "equity", "drawdown"])
    pnl_by_date = (
        trades.assign(exit_date=pd.to_datetime(trades["exit_date"]))
        .groupby("exit_date", as_index=False)["pnl_dollars"]
        .sum()
        .rename(columns={"exit_date": "date", "pnl_dollars": "daily_pnl"})
        .sort_values("date")
    )
    pnl_by_date["equity"] = initial_capital + pnl_by_date["daily_pnl"].cumsum()
    running_max = pnl_by_date["equity"].cummax()
    pnl_by_date["drawdown"] = pnl_by_date["equity"] / running_max - 1.0
    pnl_by_date["date"] = pnl_by_date["date"].dt.date.astype(str)
    return pnl_by_date


def summarize_trades(
    trades: pd.DataFrame,
    initial_capital: float = 100_000.0,
) -> pd.DataFrame:
    """Return one-row performance summary for the long mispricing strategy."""

    if trades.empty:
        return pd.DataFrame(
            [
                {
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "mean_return": 0.0,
                    "median_return": 0.0,
                    "total_pnl": 0.0,
                    "return_on_capital": 0.0,
                    "profit_factor": 0.0,
                    "sharpe_per_trade": 0.0,
                    "max_drawdown": 0.0,
                    "average_holding_days": 0.0,
                }
            ]
        )

    returns = trades["return"].astype(float)
    pnl = trades["pnl_dollars"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    equity = build_equity_curve(trades, initial_capital)
    sharpe = 0.0
    if returns.std(ddof=1) > 0:
        sharpe = math.sqrt(len(returns)) * float(returns.mean() / returns.std(ddof=1))
    profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else np.inf
    return pd.DataFrame(
        [
            {
                "total_trades": int(len(trades)),
                "win_rate": float((pnl > 0).mean()),
                "mean_return": float(returns.mean()),
                "median_return": float(returns.median()),
                "total_pnl": float(pnl.sum()),
                "return_on_capital": float(pnl.sum() / initial_capital),
                "profit_factor": profit_factor,
                "sharpe_per_trade": float(sharpe),
                "max_drawdown": float(equity["drawdown"].min()) if not equity.empty else 0.0,
                "average_holding_days": float(trades["holding_days"].mean()),
            }
        ]
    )
