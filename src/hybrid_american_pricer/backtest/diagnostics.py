from __future__ import annotations

import pandas as pd

from hybrid_american_pricer.backtest.performance import summarize_trades


def _bucketize(trades: pd.DataFrame) -> pd.DataFrame:
    bucketed = trades.copy()
    bucketed["entry_dte"] = (
        pd.to_datetime(bucketed["expiration"]) - pd.to_datetime(bucketed["entry_date"])
    ).dt.days
    bucketed["dte_bucket"] = pd.cut(
        bucketed["entry_dte"],
        bins=[0, 30, 45, 60, 10_000],
        labels=["0-30", "31-45", "46-60", "60+"],
        include_lowest=True,
    )
    bucketed["confidence_bucket"] = pd.qcut(
        bucketed["signal_confidence"].rank(method="first"),
        q=min(4, len(bucketed)),
        labels=False,
        duplicates="drop",
    )
    bucketed["edge_bucket"] = pd.qcut(
        bucketed["relative_buy_edge_entry"].rank(method="first"),
        q=min(4, len(bucketed)),
        labels=False,
        duplicates="drop",
    )
    bucketed["entry_price_bucket"] = pd.cut(
        bucketed["entry_ask"],
        bins=[0, 1, 3, 7, 10_000],
        labels=["0-1", "1-3", "3-7", "7+"],
        include_lowest=True,
    )
    return bucketed


def summarize_by_group(trades: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=[group_column, "total_trades", "win_rate", "mean_return", "total_pnl"])
    rows = []
    for value, group in trades.groupby(group_column, dropna=False):
        summary = summarize_trades(group).iloc[0].to_dict()
        summary[group_column] = value
        rows.append(summary)
    columns = [group_column] + [column for column in rows[0] if column != group_column]
    return pd.DataFrame(rows)[columns]


def build_backtest_diagnostics(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create grouped diagnostics that explain where strategy PnL comes from."""

    if trades.empty:
        empty = pd.DataFrame()
        return {
            "by_option_kind": empty,
            "by_dte_bucket": empty,
            "by_confidence_bucket": empty,
            "by_edge_bucket": empty,
            "by_entry_price_bucket": empty,
        }
    bucketed = _bucketize(trades)
    return {
        "by_option_kind": summarize_by_group(bucketed, "option_kind"),
        "by_dte_bucket": summarize_by_group(bucketed, "dte_bucket"),
        "by_confidence_bucket": summarize_by_group(bucketed, "confidence_bucket"),
        "by_edge_bucket": summarize_by_group(bucketed, "edge_bucket"),
        "by_entry_price_bucket": summarize_by_group(bucketed, "entry_price_bucket"),
    }
