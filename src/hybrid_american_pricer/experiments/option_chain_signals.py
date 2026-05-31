from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from hybrid_american_pricer.data.option_chain import (
    OptionChainConfig,
)
from hybrid_american_pricer.data.providers import (
    CsvLiveOptionChainProvider,
    SampleLiveOptionChainProvider,
    YFinanceLiveOptionChainProvider,
)
from hybrid_american_pricer.strategy.signals import SignalConfig, build_mispricing_signals


def summarize_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scanner output into a compact strategy diagnostics table."""

    summary = (
        signals.groupby("signal", as_index=False)
        .agg(
            contracts=("signal", "size"),
            mean_mid_price=("mid_price", "mean"),
            mean_relative_spread=("relative_spread", "mean"),
            mean_buy_edge=("buy_edge", "mean"),
            mean_sell_edge=("sell_edge", "mean"),
            mean_confidence=("signal_confidence", "mean"),
        )
        .sort_values("contracts", ascending=False)
    )
    return summary


def run_option_chain_signal_scan(
    source: str = "sample",
    ticker: str = "AAPL",
    input_path: Path | None = None,
    quote_date: date = date(2026, 5, 29),
    underlying_price: float = 195.0,
    rate: float = 0.045,
    dividend: float = 0.006,
    use_rough_price: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load an option chain, reprice it, and return row-level signals plus summary."""

    market_config = OptionChainConfig(
        ticker=ticker,
        quote_date=quote_date,
        underlying_price=underlying_price,
        rate=rate,
        dividend=dividend,
    )
    if source == "sample":
        chain = SampleLiveOptionChainProvider(market_config).fetch_chain(ticker, quote_date)
    elif source == "csv":
        if input_path is None:
            raise ValueError("--input is required when --source csv")
        chain = CsvLiveOptionChainProvider(input_path, market_config).fetch_chain(ticker, quote_date)
    elif source == "yfinance":
        chain = YFinanceLiveOptionChainProvider(market_config).fetch_chain(ticker, quote_date)
    else:
        raise ValueError(f"unsupported source: {source}")

    signal_config = SignalConfig(use_rough_price=use_rough_price)
    signals = build_mispricing_signals(chain, config=signal_config)
    summary = summarize_signals(signals)
    return signals, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan an equity option chain for model mispricing.")
    parser.add_argument("--source", choices=["sample", "csv", "yfinance"], default="sample")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--quote-date", type=date.fromisoformat, default=date(2026, 5, 29))
    parser.add_argument("--underlying-price", type=float, default=195.0)
    parser.add_argument("--rate", type=float, default=0.045)
    parser.add_argument("--dividend", type=float, default=0.006)
    parser.add_argument("--use-rough-price", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/option_chain_signals.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/tables/option_chain_signal_summary.csv"),
    )
    args = parser.parse_args()

    signals, summary = run_option_chain_signal_scan(
        source=args.source,
        ticker=args.ticker,
        input_path=args.input,
        quote_date=args.quote_date,
        underlying_price=args.underlying_price,
        rate=args.rate,
        dividend=args.dividend,
        use_rough_price=args.use_rough_price,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(args.output, index=False)
    summary.to_csv(args.summary_output, index=False)

    display_cols = [
        "ticker",
        "expiration",
        "option_kind",
        "strike",
        "bid",
        "ask",
        "mid_price",
        "fair_value",
        "buy_edge",
        "sell_edge",
        "relative_spread",
        "signal",
        "signal_confidence",
    ]
    print(signals[display_cols].head(20).to_string(index=False))
    print(f"\nwrote {args.output}")
    print(f"wrote {args.summary_output}")


if __name__ == "__main__":
    main()
