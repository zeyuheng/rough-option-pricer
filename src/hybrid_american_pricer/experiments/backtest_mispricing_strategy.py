from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from hybrid_american_pricer.backtest import (
    BacktestConfig,
    build_backtest_diagnostics,
    build_equity_curve,
    run_long_mispricing_backtest,
    summarize_trades,
)
from hybrid_american_pricer.data.option_chain import OptionChainConfig
from hybrid_american_pricer.data.providers import (
    CsvHistoricalOptionChainProvider,
    OnclickHistoricalOptionChainProvider,
    OratsHistoricalOptionChainProvider,
    SampleHistoricalOptionChainProvider,
)
from hybrid_american_pricer.strategy.signals import SignalConfig


FAIR_VALUE_MODELS = [
    "black_scholes",
    "binomial",
    "monte_carlo",
    "lsmc",
    "rough",
    "ensemble_mean",
    "hybrid_proxy",
    "meta_model",
]


def strategy_configs(
    holding_days: int = 5,
    max_trades_per_day: int = 5,
    fair_value_model: str = "binomial",
) -> list[BacktestConfig]:
    base_signal = SignalConfig(
        min_abs_edge=0.05,
        min_relative_edge=0.03,
        max_relative_spread=0.10,
        min_volume=100.0,
        min_open_interest=500.0,
        min_mid_price=0.10,
        min_days_to_expiry=21,
        fair_value_model=fair_value_model,
        tree_steps=250,
        rough_paths=600,
        rough_steps=30,
    )
    strict_signal = SignalConfig(
        min_abs_edge=0.08,
        min_relative_edge=0.08,
        max_relative_spread=0.08,
        min_volume=150.0,
        min_open_interest=700.0,
        min_mid_price=0.20,
        min_days_to_expiry=21,
        fair_value_model=fair_value_model,
        tree_steps=250,
        rough_paths=600,
        rough_steps=30,
    )
    return [
        BacktestConfig(
            name="baseline",
            holding_days=holding_days,
            max_trades_per_day=max_trades_per_day,
            min_moneyness=0.90,
            max_moneyness=1.10,
            max_days_to_expiry=60,
            signal_config=base_signal,
        ),
        BacktestConfig(
            name="call_only",
            holding_days=holding_days,
            max_trades_per_day=max_trades_per_day,
            option_kind="call",
            signal_config=base_signal,
        ),
        BacktestConfig(
            name="put_only",
            holding_days=holding_days,
            max_trades_per_day=max_trades_per_day,
            option_kind="put",
            signal_config=base_signal,
        ),
        BacktestConfig(
            name="atm_tight",
            holding_days=holding_days,
            max_trades_per_day=max_trades_per_day,
            min_moneyness=0.95,
            max_moneyness=1.05,
            signal_config=base_signal,
        ),
        BacktestConfig(
            name="tight_spread_high_edge",
            holding_days=holding_days,
            max_trades_per_day=max_trades_per_day,
            min_signal_confidence=0.06,
            signal_config=strict_signal,
        ),
        BacktestConfig(
            name="shorter_dte",
            holding_days=holding_days,
            max_trades_per_day=max_trades_per_day,
            min_days_to_expiry=21,
            max_days_to_expiry=45,
            signal_config=base_signal,
        ),
        BacktestConfig(
            name="hybrid_mispricing_regime",
            holding_days=holding_days,
            max_trades_per_day=min(max_trades_per_day, 3),
            min_moneyness=0.95,
            max_moneyness=1.05,
            min_days_to_expiry=21,
            max_days_to_expiry=45,
            min_signal_confidence=0.06,
            confidence_top_fraction=0.20,
            trend_filter=True,
            max_market_iv=0.35,
            take_profit=0.30,
            stop_loss=0.30,
            exit_policy="stop_or_horizon",
            signal_config=strict_signal,
        ),
    ]


def run_backtest_experiment(
    source: str = "sample",
    ticker: str = "AAPL",
    input_path: Path | None = None,
    start: date = date(2026, 5, 29),
    end: date | None = None,
    underlying_price: float = 195.0,
    rate: float = 0.045,
    dividend: float = 0.006,
    holding_days: int = 5,
    max_trades_per_day: int = 5,
    fair_value_model: str = "binomial",
) -> tuple:
    market_config = OptionChainConfig(
        ticker=ticker,
        quote_date=start,
        underlying_price=underlying_price,
        rate=rate,
        dividend=dividend,
    )
    if source == "sample":
        provider = SampleHistoricalOptionChainProvider(market_config)
    elif source == "csv":
        if input_path is None:
            raise ValueError("--input is required when --source csv")
        provider = CsvHistoricalOptionChainProvider(input_path, market_config)
    elif source == "orats":
        provider = OratsHistoricalOptionChainProvider(market_config)
    elif source == "onclick":
        provider = OnclickHistoricalOptionChainProvider(market_config)
    else:
        raise ValueError(f"unsupported source: {source}")

    history = provider.fetch_history(ticker, start=start, end=end)
    backtest_config = strategy_configs(holding_days, max_trades_per_day, fair_value_model)[0]
    trades = run_long_mispricing_backtest(history, backtest_config)
    summary = summarize_trades(trades, backtest_config.initial_capital)
    equity_curve = build_equity_curve(trades, backtest_config.initial_capital)
    return trades, summary, equity_curve


def run_strategy_comparison(
    history,
    holding_days: int = 5,
    max_trades_per_day: int = 5,
    fair_value_model: str = "binomial",
) -> tuple:
    models = FAIR_VALUE_MODELS if fair_value_model == "all" else [fair_value_model]
    trade_frames = []
    summary_frames = []
    equity_frames = []
    for model_name in models:
        for config in strategy_configs(holding_days, max_trades_per_day, model_name):
            trades = run_long_mispricing_backtest(history, config)
            summary = summarize_trades(trades, config.initial_capital)
            summary.insert(0, "strategy_name", config.name)
            summary.insert(1, "fair_value_model", model_name)
            equity = build_equity_curve(trades, config.initial_capital)
            if not trades.empty:
                trade_frames.append(trades)
            summary_frames.append(summary)
            if not equity.empty:
                equity.insert(0, "fair_value_model", model_name)
                equity.insert(1, "strategy_name", config.name)
                equity_frames.append(equity)
    import pandas as pd

    return (
        pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(),
        pd.concat(summary_frames, ignore_index=True),
        pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest long underpriced option signals.")
    parser.add_argument("--source", choices=["sample", "csv", "orats", "onclick"], default="sample")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--orats-token", default=None)
    parser.add_argument("--orats-token-env", default="ORATS_TOKEN")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 5, 29))
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--underlying-price", type=float, default=195.0)
    parser.add_argument("--rate", type=float, default=0.045)
    parser.add_argument("--dividend", type=float, default=0.006)
    parser.add_argument("--holding-days", type=int, default=5)
    parser.add_argument("--max-trades-per-day", type=int, default=5)
    parser.add_argument(
        "--fair-value-model",
        choices=FAIR_VALUE_MODELS + ["all"],
        default="binomial",
    )
    parser.add_argument("--trades-output", type=Path, default=Path("reports/tables/trades.csv"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/tables/backtest_summary.csv"),
    )
    parser.add_argument(
        "--equity-output",
        type=Path,
        default=Path("reports/tables/equity_curve.csv"),
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=Path("reports/tables"),
    )
    args = parser.parse_args()

    market_config = OptionChainConfig(
        ticker=args.ticker,
        quote_date=args.start,
        underlying_price=args.underlying_price,
        rate=args.rate,
        dividend=args.dividend,
    )
    if args.source == "sample":
        provider = SampleHistoricalOptionChainProvider(market_config)
    elif args.source == "csv":
        if args.input is None:
            raise ValueError("--input is required when --source csv")
        provider = CsvHistoricalOptionChainProvider(args.input, market_config)
    elif args.source == "orats":
        provider = OratsHistoricalOptionChainProvider(
            market_config,
            token=args.orats_token,
            token_env=args.orats_token_env,
            dte=(21, 60),
        )
    elif args.source == "onclick":
        provider = OnclickHistoricalOptionChainProvider(market_config)
    else:
        raise ValueError(f"unsupported source: {args.source}")
    history = provider.fetch_history(args.ticker, start=args.start, end=args.end)
    trades, summary, equity_curve = run_strategy_comparison(
        history,
        holding_days=args.holding_days,
        max_trades_per_day=args.max_trades_per_day,
        fair_value_model=args.fair_value_model,
    )
    for output, frame in [
        (args.trades_output, trades),
        (args.summary_output, summary),
        (args.equity_output, equity_curve),
    ]:
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
    baseline_trades = trades[trades["strategy_name"] == "baseline"] if not trades.empty else trades
    for name, frame in build_backtest_diagnostics(baseline_trades).items():
        output = args.diagnostics_dir / f"backtest_diagnostics_{name}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)

    print(summary.to_string(index=False))
    print(f"\nwrote {args.trades_output}")
    print(f"wrote {args.summary_output}")
    print(f"wrote {args.equity_output}")


if __name__ == "__main__":
    main()
