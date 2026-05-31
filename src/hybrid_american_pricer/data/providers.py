from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
import os
from pathlib import Path
from typing import Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from io import StringIO

import pandas as pd

from hybrid_american_pricer.data.option_chain import (
    OptionChainConfig,
    fetch_yfinance_option_chain,
    generate_sample_option_chain,
    generate_sample_option_chain_history,
    load_option_chain_csv,
    standardize_option_chain,
)
from hybrid_american_pricer.data.schema import enforce_option_chain_schema


class LiveOptionChainProvider(Protocol):
    """Provider interface for one quote-date option-chain snapshots."""

    def fetch_chain(self, ticker: str, quote_date: date | None = None) -> pd.DataFrame:
        ...


class HistoricalOptionChainProvider(Protocol):
    """Provider interface for backtestable historical option-chain snapshots."""

    def fetch_history(
        self,
        ticker: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        ...


@dataclass(frozen=True)
class SampleHistoricalOptionChainProvider:
    config: OptionChainConfig = OptionChainConfig()
    n_quote_dates: int = 12
    quote_spacing_days: int = 3

    def fetch_history(
        self,
        ticker: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        config = OptionChainConfig(
            ticker=ticker,
            quote_date=start or self.config.quote_date,
            underlying_price=self.config.underlying_price,
            rate=self.config.rate,
            dividend=self.config.dividend,
            exercise=self.config.exercise,
        )
        history = generate_sample_option_chain_history(
            config,
            n_quote_dates=self.n_quote_dates,
            quote_spacing_days=self.quote_spacing_days,
        )
        if end is not None and not history.empty:
            history = history[pd.to_datetime(history["quote_date"]).dt.date <= end]
        return enforce_option_chain_schema(history.reset_index(drop=True))


@dataclass(frozen=True)
class SampleLiveOptionChainProvider:
    config: OptionChainConfig = OptionChainConfig()

    def fetch_chain(self, ticker: str, quote_date: date | None = None) -> pd.DataFrame:
        config = OptionChainConfig(
            ticker=ticker,
            quote_date=quote_date or self.config.quote_date,
            underlying_price=self.config.underlying_price,
            rate=self.config.rate,
            dividend=self.config.dividend,
            exercise=self.config.exercise,
        )
        return enforce_option_chain_schema(generate_sample_option_chain(config))


@dataclass(frozen=True)
class CsvLiveOptionChainProvider:
    path: Path
    config: OptionChainConfig

    def fetch_chain(self, ticker: str, quote_date: date | None = None) -> pd.DataFrame:
        config = OptionChainConfig(
            ticker=ticker,
            quote_date=quote_date or self.config.quote_date,
            underlying_price=self.config.underlying_price,
            rate=self.config.rate,
            dividend=self.config.dividend,
            exercise=self.config.exercise,
        )
        return enforce_option_chain_schema(load_option_chain_csv(self.path, config))


@dataclass(frozen=True)
class YFinanceLiveOptionChainProvider:
    config: OptionChainConfig | None = None
    max_expirations: int = 2

    def fetch_chain(self, ticker: str, quote_date: date | None = None) -> pd.DataFrame:
        config = self.config
        if config is not None and quote_date is not None:
            config = OptionChainConfig(
                ticker=ticker,
                quote_date=quote_date,
                underlying_price=config.underlying_price,
                rate=config.rate,
                dividend=config.dividend,
                exercise=config.exercise,
            )
        return enforce_option_chain_schema(
            fetch_yfinance_option_chain(ticker, config=config, max_expirations=self.max_expirations)
        )


@dataclass(frozen=True)
class CsvHistoricalOptionChainProvider:
    """Historical provider for CSV exports with one or many quote_date snapshots."""

    path: Path
    fallback_config: OptionChainConfig

    def fetch_history(
        self,
        ticker: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        raw = pd.read_csv(self.path)
        frames = []
        if "quote_date" not in raw.columns:
            config = OptionChainConfig(
                ticker=ticker,
                quote_date=self.fallback_config.quote_date,
                underlying_price=self.fallback_config.underlying_price,
                rate=self.fallback_config.rate,
                dividend=self.fallback_config.dividend,
                exercise=self.fallback_config.exercise,
            )
            frames.append(standardize_option_chain(raw, config))
        else:
            for quote_date_value, snapshot in raw.groupby("quote_date"):
                snapshot_date = pd.to_datetime(quote_date_value).date()
                if start is not None and snapshot_date < start:
                    continue
                if end is not None and snapshot_date > end:
                    continue
                if "ticker" in snapshot.columns:
                    snapshot = snapshot[snapshot["ticker"].astype(str).str.upper() == ticker.upper()]
                if snapshot.empty:
                    continue
                spot = self._snapshot_value(snapshot, "underlying_price", self.fallback_config.underlying_price)
                rate = self._snapshot_value(snapshot, "rate", self.fallback_config.rate)
                dividend = self._snapshot_value(snapshot, "dividend", self.fallback_config.dividend)
                config = OptionChainConfig(
                    ticker=ticker,
                    quote_date=snapshot_date,
                    underlying_price=spot,
                    rate=rate,
                    dividend=dividend,
                    exercise=self.fallback_config.exercise,
                )
                frames.append(standardize_option_chain(snapshot, config))

        if not frames:
            return pd.DataFrame(columns=[])
        history = pd.concat(frames, ignore_index=True)
        history = history.sort_values(["quote_date", "expiration", "option_kind", "strike"])
        return enforce_option_chain_schema(history.reset_index(drop=True))

    @staticmethod
    def _snapshot_value(snapshot: pd.DataFrame, column: str, default: float) -> float:
        if column in snapshot.columns and snapshot[column].notna().any():
            return float(snapshot[column].dropna().iloc[0])
        return float(default)


def orats_strikes_to_option_chain(raw: pd.DataFrame, fallback_config: OptionChainConfig) -> pd.DataFrame:
    """Convert ORATS hist/strikes wide call/put rows into the common option-chain schema."""

    frames = []
    if raw.empty:
        return pd.DataFrame()
    for trade_date_value, snapshot in raw.groupby("tradeDate"):
        quote_date = pd.to_datetime(trade_date_value).date()
        stock_price = float(snapshot["stockPrice"].dropna().iloc[0])
        rows = []
        for _, row in snapshot.iterrows():
            for kind, prefix in [("call", "call"), ("put", "put")]:
                bid = row.get(f"{prefix}BidPrice")
                ask = row.get(f"{prefix}AskPrice")
                if pd.isna(bid) or pd.isna(ask):
                    continue
                rows.append(
                    {
                        "expiration": row["expirDate"],
                        "strike": row["strike"],
                        "option_type": kind,
                        "bid": bid,
                        "ask": ask,
                        "last": row.get(f"{prefix}Value", pd.NA),
                        "volume": row.get(f"{prefix}Volume", 0.0),
                        "open_interest": row.get(f"{prefix}OpenInterest", 0.0),
                        "market_iv": row.get(f"{prefix}MidIv", row.get("smvVol", pd.NA)),
                    }
                )
        if rows:
            config = OptionChainConfig(
                ticker=str(snapshot["ticker"].iloc[0]).upper(),
                quote_date=quote_date,
                underlying_price=stock_price,
                rate=fallback_config.rate,
                dividend=fallback_config.dividend,
                exercise=fallback_config.exercise,
            )
            frames.append(standardize_option_chain(pd.DataFrame(rows), config))
    if not frames:
        return pd.DataFrame()
    return enforce_option_chain_schema(pd.concat(frames, ignore_index=True))


@dataclass(frozen=True)
class OratsHistoricalOptionChainProvider:
    """Historical EOD option-chain provider backed by ORATS datav2 hist/strikes."""

    fallback_config: OptionChainConfig
    token: str | None = None
    token_env: str = "ORATS_TOKEN"
    base_url: str = "https://api.orats.io/datav2/hist/strikes"
    dte: tuple[int, int] | None = (21, 60)
    timeout_seconds: int = 30

    def fetch_history(
        self,
        ticker: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        start_date = start or self.fallback_config.quote_date
        end_date = end or start_date
        frames = []
        for trade_date in pd.bdate_range(start_date, end_date):
            raw = self.fetch_snapshot(ticker, trade_date.date())
            if not raw.empty:
                frames.append(orats_strikes_to_option_chain(raw, self.fallback_config))
        if not frames:
            return pd.DataFrame()
        history = pd.concat(frames, ignore_index=True)
        return enforce_option_chain_schema(
            history.sort_values(["quote_date", "expiration", "option_kind", "strike"]).reset_index(drop=True)
        )

    def fetch_snapshot(self, ticker: str, trade_date: date) -> pd.DataFrame:
        token = self.token or os.getenv(self.token_env)
        if not token:
            raise ValueError(f"ORATS token not found. Set {self.token_env} or pass token=...")
        params: dict[str, str] = {
            "token": token,
            "ticker": ticker.upper(),
            "tradeDate": trade_date.isoformat(),
            "fields": ",".join(
                [
                    "ticker",
                    "tradeDate",
                    "expirDate",
                    "dte",
                    "strike",
                    "stockPrice",
                    "callVolume",
                    "callOpenInterest",
                    "putVolume",
                    "putOpenInterest",
                    "callBidPrice",
                    "callAskPrice",
                    "putBidPrice",
                    "putAskPrice",
                    "callValue",
                    "putValue",
                    "callMidIv",
                    "putMidIv",
                    "smvVol",
                ]
            ),
        }
        if self.dte is not None:
            params["dte"] = f"{self.dte[0]},{self.dte[1]}"
        url = f"{self.base_url}?{urlencode(params)}"
        with urlopen(url, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return pd.DataFrame(payload.get("data", []))


def onclick_options_to_option_chain(raw: pd.DataFrame, config: OptionChainConfig) -> pd.DataFrame:
    """Convert OnclickMedia /options rows into the common option-chain schema."""

    if raw.empty:
        return pd.DataFrame()
    rename_map = {
        "type": "option_type",
        "implied_volatility": "market_iv",
        "date": "quote_date",
    }
    normalized = raw.rename(columns={key: value for key, value in rename_map.items() if key in raw.columns})
    return standardize_option_chain(normalized, config)


@dataclass(frozen=True)
class OnclickHistoricalOptionChainProvider:
    """Free historical EOD option-chain provider backed by OnclickMedia."""

    fallback_config: OptionChainConfig
    base_url: str = "https://api.onclickmedia.com/options/"
    stock_url: str = "https://api.onclickmedia.com/stock-data/"
    data: str = "all"
    dte: tuple[int, int] | None = (21, 60)
    moneyness: tuple[float, float] | None = (0.90, 1.10)
    timeout_seconds: int = 30
    user_agent: str = "Mozilla/5.0 (compatible; HybridAmericanPricer/0.1)"

    def fetch_history(
        self,
        ticker: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        start_date = start or self.fallback_config.quote_date
        end_date = end or start_date
        frames = []
        for trade_date in pd.bdate_range(start_date, end_date):
            raw = self.fetch_snapshot(ticker, trade_date.date())
            if raw.empty:
                continue
            spot = self._underlying_price(raw, ticker, trade_date.date())
            config = OptionChainConfig(
                ticker=ticker,
                quote_date=trade_date.date(),
                underlying_price=spot,
                rate=self.fallback_config.rate,
                dividend=self.fallback_config.dividend,
                exercise=self.fallback_config.exercise,
            )
            standardized = onclick_options_to_option_chain(raw, config)
            if self.dte is not None:
                standardized = standardized[
                    standardized["days_to_expiry"].between(self.dte[0], self.dte[1])
                ]
            if self.moneyness is not None:
                standardized = standardized[
                    standardized["moneyness"].between(self.moneyness[0], self.moneyness[1])
                ]
            if not standardized.empty:
                frames.append(standardized)
        if not frames:
            return pd.DataFrame()
        history = pd.concat(frames, ignore_index=True)
        return enforce_option_chain_schema(
            history.sort_values(["quote_date", "expiration", "option_kind", "strike"]).reset_index(drop=True)
        )

    def fetch_snapshot(self, ticker: str, trade_date: date) -> pd.DataFrame:
        params = {
            "ticker": ticker.upper(),
            "date": trade_date.isoformat(),
            "data": self.data,
            "output": "csv",
        }
        url = f"{self.base_url}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            text = response.read().decode("utf-8")
        if not text.strip() or text.strip().startswith("{"):
            return pd.DataFrame()
        frame = pd.read_csv(StringIO(text))
        if "error" in {column.lower() for column in frame.columns}:
            return pd.DataFrame()
        return frame

    def _underlying_price(self, raw: pd.DataFrame, ticker: str, trade_date: date) -> float:
        for column in ("underlying_price", "underlying", "stock_price", "stockPrice"):
            if column in raw.columns and raw[column].notna().any():
                return float(raw[column].dropna().iloc[0])
        try:
            params = {
                "ticker": ticker.upper(),
                "from": trade_date.isoformat(),
                "to": trade_date.isoformat(),
                "output": "csv",
            }
            url = f"{self.stock_url}?{urlencode(params)}"
            request = Request(url, headers={"User-Agent": self.user_agent})
            with urlopen(request, timeout=self.timeout_seconds) as response:
                text = response.read().decode("utf-8")
            stock = pd.read_csv(StringIO(text))
            for column in ("close", "Close", "price", "mid", "mark"):
                if column in stock.columns and stock[column].notna().any():
                    return float(stock[column].dropna().iloc[-1])
        except Exception:
            pass
        return self.fallback_config.underlying_price
