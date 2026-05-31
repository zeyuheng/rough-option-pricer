from __future__ import annotations

from datetime import date

import pandas as pd

from hybrid_american_pricer.data.option_chain import OptionChainConfig, generate_sample_option_chain
from hybrid_american_pricer.data.providers import CsvHistoricalOptionChainProvider
from hybrid_american_pricer.data.providers import onclick_options_to_option_chain, orats_strikes_to_option_chain
from hybrid_american_pricer.data.schema import validate_option_chain_schema
from hybrid_american_pricer.data.storage import OptionChainStore
from hybrid_american_pricer.experiments.option_chain_signals import run_option_chain_signal_scan
from hybrid_american_pricer.strategy.signals import SignalConfig, build_mispricing_signals


def test_sample_option_chain_has_strategy_schema() -> None:
    chain = generate_sample_option_chain(
        OptionChainConfig(
            ticker="SPY",
            quote_date=date(2026, 5, 29),
            underlying_price=500.0,
        )
    )

    required = {
        "ticker",
        "expiration",
        "strike",
        "option_kind",
        "bid",
        "ask",
        "mid_price",
        "relative_spread",
        "market_iv",
        "maturity",
        "moneyness",
    }
    assert required.issubset(chain.columns)
    assert set(chain["option_kind"]) == {"call", "put"}
    assert (chain["ask"] >= chain["bid"]).all()
    assert validate_option_chain_schema(chain).valid


def test_mispricing_signals_include_trade_edges() -> None:
    chain = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "quote_date": "2026-05-29",
                "expiration": "2026-06-19",
                "days_to_expiry": 21,
                "strike": 195.0,
                "option_kind": "put",
                "exercise": "american",
                "bid": 1.0,
                "ask": 1.1,
                "mid_price": 1.05,
                "spread": 0.1,
                "relative_spread": 0.095,
                "volume": 500,
                "open_interest": 1000,
                "market_iv": 0.25,
                "underlying_price": 195.0,
                "rate": 0.04,
                "dividend": 0.0,
                "maturity": 21 / 365,
                "moneyness": 1.0,
                "log_moneyness": 0.0,
                "maturity_bucket": "short",
            }
        ]
    )
    signals = build_mispricing_signals(
        chain,
        SignalConfig(min_abs_edge=0.01, min_relative_edge=0.001, tree_steps=50),
    )

    assert {"fair_value", "buy_edge", "sell_edge", "signal", "liquidity_pass"}.issubset(
        signals.columns
    )
    assert signals.loc[0, "liquidity_pass"]
    assert signals.loc[0, "signal"] in {"buy_underpriced", "sell_overpriced", "hold"}


def test_option_chain_signal_experiment_writes_tables(tmp_path) -> None:
    signals, summary = run_option_chain_signal_scan(
        source="sample",
        ticker="AAPL",
        quote_date=date(2026, 5, 29),
        underlying_price=195.0,
    )

    assert not signals.empty
    assert not summary.empty
    assert "signal" in summary.columns


def test_csv_historical_provider_standardizes_multiple_quote_dates(tmp_path) -> None:
    path = tmp_path / "history.csv"
    pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "quote_date": "2026-05-29",
                "expiration": "2026-06-19",
                "strike": 195.0,
                "option_type": "call",
                "bid": 3.1,
                "ask": 3.3,
                "market_iv": 0.22,
                "volume": 100,
                "open_interest": 800,
                "underlying_price": 195.0,
            },
            {
                "ticker": "AAPL",
                "quote_date": "2026-05-30",
                "expiration": "2026-06-19",
                "strike": 195.0,
                "option_type": "put",
                "bid": 2.4,
                "ask": 2.6,
                "market_iv": 0.24,
                "volume": 120,
                "open_interest": 850,
                "underlying_price": 197.0,
            },
        ]
    ).to_csv(path, index=False)
    provider = CsvHistoricalOptionChainProvider(
        path=path,
        fallback_config=OptionChainConfig(ticker="AAPL", quote_date=date(2026, 5, 29)),
    )

    history = provider.fetch_history("AAPL")

    assert validate_option_chain_schema(history).valid
    assert list(history["quote_date"]) == ["2026-05-29", "2026-05-30"]
    assert list(history["option_kind"]) == ["call", "put"]


def test_option_chain_store_round_trips_snapshot(tmp_path) -> None:
    chain = generate_sample_option_chain(
        OptionChainConfig(
            ticker="MSFT",
            quote_date=date(2026, 5, 29),
            underlying_price=420.0,
        )
    ).head(4)
    store = OptionChainStore(root=tmp_path)

    path = store.save_snapshot(chain)
    loaded = store.load_snapshot("MSFT", "2026-05-29")

    assert path.exists()
    assert validate_option_chain_schema(loaded).valid
    assert len(loaded) == 4


def test_orats_strikes_are_converted_to_standard_schema() -> None:
    raw = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "tradeDate": "2024-01-03",
                "expirDate": "2024-02-16",
                "dte": 44,
                "strike": 190.0,
                "stockPrice": 185.0,
                "callVolume": 100,
                "callOpenInterest": 1200,
                "putVolume": 80,
                "putOpenInterest": 900,
                "callBidPrice": 5.1,
                "callAskPrice": 5.3,
                "putBidPrice": 8.2,
                "putAskPrice": 8.5,
                "callValue": 5.2,
                "putValue": 8.35,
                "callMidIv": 0.22,
                "putMidIv": 0.24,
                "smvVol": 0.23,
            }
        ]
    )

    standardized = orats_strikes_to_option_chain(
        raw,
        OptionChainConfig(ticker="AAPL", quote_date=date(2024, 1, 3), underlying_price=185.0),
    )

    assert validate_option_chain_schema(standardized).valid
    assert len(standardized) == 2
    assert set(standardized["option_kind"]) == {"call", "put"}
    assert set(standardized["quote_date"]) == {"2024-01-03"}


def test_onclick_options_are_converted_to_standard_schema() -> None:
    raw = pd.DataFrame(
        [
            {
                "symbol": "AAPL240216C00190000",
                "expiration": "2024-02-16",
                "strike": 190.0,
                "type": "call",
                "last": 5.2,
                "bid": 5.1,
                "ask": 5.3,
                "volume": 100,
                "open_interest": 1200,
                "implied_volatility": 0.22,
            },
            {
                "symbol": "AAPL240216P00190000",
                "expiration": "2024-02-16",
                "strike": 190.0,
                "type": "put",
                "last": 8.35,
                "bid": 8.2,
                "ask": 8.5,
                "volume": 80,
                "open_interest": 900,
                "implied_volatility": 0.24,
            },
        ]
    )

    standardized = onclick_options_to_option_chain(
        raw,
        OptionChainConfig(
            ticker="AAPL",
            quote_date=date(2024, 1, 3),
            underlying_price=185.0,
        ),
    )

    assert validate_option_chain_schema(standardized).valid
    assert len(standardized) == 2
    assert set(standardized["option_kind"]) == {"call", "put"}
    assert standardized["market_iv"].notna().all()
