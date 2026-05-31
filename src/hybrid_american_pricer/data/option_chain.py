from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from hybrid_american_pricer.models.black_scholes import BlackScholesPricer
from hybrid_american_pricer.options.instruments import MarketState, OptionContract
from hybrid_american_pricer.meta.features import maturity_bucket


YEAR_DAYS = 365.0


@dataclass(frozen=True)
class OptionChainConfig:
    ticker: str = "AAPL"
    quote_date: date = date(2026, 5, 29)
    underlying_price: float = 195.0
    rate: float = 0.045
    dividend: float = 0.006
    exercise: str = "american"


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def _first_present(row: pd.Series, candidates: tuple[str, ...], default: object = np.nan) -> object:
    for name in candidates:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def _option_kind(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"c", "call", "calls"}:
        return "call"
    if text in {"p", "put", "puts"}:
        return "put"
    raise ValueError(f"unsupported option type: {value}")


def _maturity_years(quote_date: date, expiration: date) -> float:
    return max((expiration - quote_date).days / YEAR_DAYS, 1.0 / YEAR_DAYS)


def standardize_option_chain(
    raw: pd.DataFrame,
    config: OptionChainConfig,
) -> pd.DataFrame:
    """Normalize vendor-specific option-chain columns into one strategy-ready schema."""

    rows: list[dict[str, object]] = []
    for _, row in raw.iterrows():
        expiration = _as_date(_first_present(row, ("expiration", "expiry", "expirationDate")))
        kind = _option_kind(_first_present(row, ("option_type", "type", "kind", "contract_type")))
        strike = float(_first_present(row, ("strike", "strikePrice")))
        bid = float(_first_present(row, ("bid",)))
        ask = float(_first_present(row, ("ask",)))
        last = float(_first_present(row, ("last", "lastPrice"), default=np.nan))
        market_iv = float(
            _first_present(
                row,
                ("market_iv", "impliedVolatility", "implied_volatility", "iv"),
                default=np.nan,
            )
        )
        volume = float(_first_present(row, ("volume",), default=0.0) or 0.0)
        open_interest = float(_first_present(row, ("open_interest", "openInterest"), default=0.0) or 0.0)

        mid = 0.5 * (bid + ask)
        spread = ask - bid
        maturity = _maturity_years(config.quote_date, expiration)
        moneyness = config.underlying_price / strike
        rows.append(
            {
                "ticker": config.ticker.upper(),
                "quote_date": config.quote_date.isoformat(),
                "expiration": expiration.isoformat(),
                "days_to_expiry": int((expiration - config.quote_date).days),
                "strike": strike,
                "option_kind": kind,
                "exercise": config.exercise,
                "bid": bid,
                "ask": ask,
                "mid_price": mid,
                "last": last,
                "spread": spread,
                "relative_spread": spread / max(mid, 1e-12),
                "volume": volume,
                "open_interest": open_interest,
                "market_iv": market_iv,
                "underlying_price": config.underlying_price,
                "rate": config.rate,
                "dividend": config.dividend,
                "maturity": maturity,
                "moneyness": moneyness,
                "log_moneyness": math.log(moneyness),
                "maturity_bucket": maturity_bucket(maturity),
            }
        )

    standardized = pd.DataFrame(rows)
    return standardized.sort_values(["expiration", "option_kind", "strike"]).reset_index(drop=True)


def load_option_chain_csv(path: str | Path, config: OptionChainConfig) -> pd.DataFrame:
    """Load a local option-chain CSV and return the normalized schema."""

    return standardize_option_chain(pd.read_csv(path), config=config)


def generate_sample_option_chain(config: OptionChainConfig) -> pd.DataFrame:
    """Create a deterministic, realistic-looking option chain for offline strategy tests."""

    rng = np.random.default_rng(7)
    expiries = [config.quote_date + timedelta(days=d) for d in (21, 45, 75, 120)]
    strike_grid = np.round(config.underlying_price * np.linspace(0.82, 1.18, 13) / 5.0) * 5.0
    bs = BlackScholesPricer()
    rows = []

    for expiration in expiries:
        maturity = _maturity_years(config.quote_date, expiration)
        tenor_bump = 0.015 * math.sqrt(maturity)
        for strike in strike_grid:
            log_moneyness = math.log(config.underlying_price / strike)
            smile = 0.20 + 0.10 * abs(log_moneyness) + tenor_bump
            for kind in ("call", "put"):
                contract = OptionContract(
                    strike=float(strike),
                    maturity=maturity,
                    kind=kind,
                    exercise="european",
                )
                market = MarketState(
                    spot=config.underlying_price,
                    rate=config.rate,
                    volatility=smile,
                    dividend=config.dividend,
                )
                theoretical = bs.price(contract, market).price
                quote_noise = rng.normal(0.0, 0.025 * max(theoretical, 1.0))
                mid = max(theoretical + quote_noise, 0.01)
                liquidity_penalty = min(abs(log_moneyness) * 0.20 + 0.015, 0.25)
                spread = max(0.02, mid * liquidity_penalty)
                rows.append(
                    {
                        "expiration": expiration.isoformat(),
                        "strike": float(strike),
                        "option_type": kind,
                        "bid": max(mid - 0.5 * spread, 0.01),
                        "ask": mid + 0.5 * spread,
                        "last": mid + rng.normal(0.0, 0.2 * spread),
                        "volume": int(max(0, rng.normal(350, 150) - 700 * abs(log_moneyness))),
                        "open_interest": int(max(10, rng.normal(1800, 600) - 2200 * abs(log_moneyness))),
                        "market_iv": smile + rng.normal(0.0, 0.01),
                    }
                )

    return standardize_option_chain(pd.DataFrame(rows), config=config)


def generate_sample_option_chain_history(
    config: OptionChainConfig,
    n_quote_dates: int = 12,
    quote_spacing_days: int = 3,
) -> pd.DataFrame:
    """Create deterministic historical snapshots with overlapping option contracts."""

    rng = np.random.default_rng(11)
    base_expiries = [config.quote_date + timedelta(days=d) for d in (30, 45, 60)]
    strike_grid = np.round(config.underlying_price * np.linspace(0.90, 1.10, 9) / 5.0) * 5.0
    frames = []

    for idx in range(n_quote_dates):
        quote_date = config.quote_date + timedelta(days=idx * quote_spacing_days)
        spot = config.underlying_price * math.exp(0.015 * math.sin(idx / 2.0) + 0.003 * idx)
        snapshot_config = OptionChainConfig(
            ticker=config.ticker,
            quote_date=quote_date,
            underlying_price=spot,
            rate=config.rate,
            dividend=config.dividend,
            exercise=config.exercise,
        )
        raw_rows = []
        for expiration in base_expiries:
            if expiration <= quote_date + timedelta(days=1):
                continue
            maturity = _maturity_years(quote_date, expiration)
            for strike in strike_grid:
                log_moneyness = math.log(spot / strike)
                smile = 0.20 + 0.12 * abs(log_moneyness) + 0.015 * math.sqrt(maturity)
                for kind in ("call", "put"):
                    contract = OptionContract(
                        strike=float(strike),
                        maturity=maturity,
                        kind=kind,
                        exercise="european",
                    )
                    market = MarketState(
                        spot=spot,
                        rate=config.rate,
                        volatility=smile,
                        dividend=config.dividend,
                    )
                    theoretical = BlackScholesPricer().price(contract, market).price
                    convergence_noise = 0.04 * math.sin(idx + strike / 11.0 + (0.7 if kind == "put" else 0.0))
                    mid = max(theoretical * (1.0 + convergence_noise) + rng.normal(0.0, 0.015), 0.01)
                    spread = max(0.02, mid * min(0.015 + 0.16 * abs(log_moneyness), 0.18))
                    raw_rows.append(
                        {
                            "expiration": expiration.isoformat(),
                            "strike": float(strike),
                            "option_type": kind,
                            "bid": max(mid - 0.5 * spread, 0.01),
                            "ask": mid + 0.5 * spread,
                            "last": mid + rng.normal(0.0, 0.15 * spread),
                            "volume": int(max(20, rng.normal(550, 120) - 900 * abs(log_moneyness))),
                            "open_interest": int(max(100, rng.normal(2400, 500) - 2400 * abs(log_moneyness))),
                            "market_iv": smile + rng.normal(0.0, 0.006),
                        }
                    )
        if raw_rows:
            frames.append(standardize_option_chain(pd.DataFrame(raw_rows), snapshot_config))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_yfinance_option_chain(
    ticker: str,
    config: OptionChainConfig | None = None,
    max_expirations: int = 2,
) -> pd.DataFrame:
    """Optionally fetch a delayed option chain with yfinance if the user installed it."""

    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:
        raise ImportError("Install yfinance to use the live Yahoo Finance provider.") from exc

    ticker_obj = yf.Ticker(ticker)
    expirations = list(ticker_obj.options[:max_expirations])
    if not expirations:
        raise ValueError(f"No option expirations found for {ticker}.")

    spot = float(ticker_obj.history(period="1d")["Close"].iloc[-1])
    base_config = config or OptionChainConfig(
        ticker=ticker,
        quote_date=date.today(),
        underlying_price=spot,
    )
    raw_frames = []
    for expiration in expirations:
        chain = ticker_obj.option_chain(expiration)
        calls = chain.calls.copy()
        calls["option_type"] = "call"
        puts = chain.puts.copy()
        puts["option_type"] = "put"
        frame = pd.concat([calls, puts], ignore_index=True)
        frame["expiration"] = expiration
        raw_frames.append(frame)
    return standardize_option_chain(pd.concat(raw_frames, ignore_index=True), base_config)
