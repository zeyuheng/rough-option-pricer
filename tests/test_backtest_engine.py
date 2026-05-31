from __future__ import annotations

from datetime import date

import pandas as pd

from hybrid_american_pricer.backtest import (
    BacktestConfig,
    build_equity_curve,
    run_long_mispricing_backtest,
    summarize_trades,
)
from hybrid_american_pricer.data.option_chain import OptionChainConfig, generate_sample_option_chain_history
from hybrid_american_pricer.experiments.backtest_mispricing_strategy import run_backtest_experiment
from hybrid_american_pricer.strategy.fair_value import FairValueConfig, FairValueEngine
from hybrid_american_pricer.strategy.signals import SignalConfig


def test_long_mispricing_backtest_creates_trades() -> None:
    history = generate_sample_option_chain_history(
        OptionChainConfig(
            ticker="AAPL",
            quote_date=date(2026, 5, 29),
            underlying_price=195.0,
        ),
        n_quote_dates=8,
        quote_spacing_days=3,
    )
    config = BacktestConfig(
        holding_days=5,
        max_trades_per_day=3,
        signal_config=SignalConfig(
            min_abs_edge=0.01,
            min_relative_edge=0.005,
            max_relative_spread=0.20,
            min_volume=20,
            min_open_interest=100,
            min_days_to_expiry=10,
            tree_steps=75,
        ),
    )

    trades = run_long_mispricing_backtest(history, config)

    assert not trades.empty
    assert {"entry_date", "exit_date", "entry_ask", "exit_bid", "pnl_dollars", "return"}.issubset(
        trades.columns
    )
    assert (trades["entry_ask"] > 0).all()


def test_backtest_performance_tables() -> None:
    trades, summary, equity_curve = run_backtest_experiment(
        source="sample",
        ticker="AAPL",
        start=date(2026, 5, 29),
        underlying_price=195.0,
        holding_days=5,
        max_trades_per_day=3,
    )

    assert not trades.empty
    assert not summary.empty
    assert not equity_curve.empty
    assert summary.loc[0, "total_trades"] == len(trades)
    assert {"date", "daily_pnl", "equity", "drawdown"}.issubset(equity_curve.columns)


def test_empty_backtest_summary_is_well_formed() -> None:
    summary = summarize_trades(pd.DataFrame())

    assert summary.loc[0, "total_trades"] == 0


def test_fair_value_engine_supports_multiple_models() -> None:
    history = generate_sample_option_chain_history(
        OptionChainConfig(ticker="AAPL", quote_date=date(2026, 5, 29), underlying_price=195.0),
        n_quote_dates=1,
    )
    row = history.iloc[0]

    for model in ["black_scholes", "binomial", "monte_carlo", "lsmc", "rough", "ensemble_mean", "hybrid_proxy"]:
        engine = FairValueEngine(
            FairValueConfig(
                model=model,
                tree_steps=20,
                mc_paths=100,
                mc_steps=8,
                lsmc_paths=100,
                lsmc_steps=8,
                rough_paths=100,
                rough_steps=8,
            )
        )
        price = engine.price_row(row)["fair_value"]
        assert price >= 0.0
