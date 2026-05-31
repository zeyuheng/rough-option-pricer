from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


OPTION_CHAIN_COLUMNS = [
    "ticker",
    "quote_date",
    "expiration",
    "days_to_expiry",
    "strike",
    "option_kind",
    "exercise",
    "bid",
    "ask",
    "mid_price",
    "last",
    "spread",
    "relative_spread",
    "volume",
    "open_interest",
    "market_iv",
    "underlying_price",
    "rate",
    "dividend",
    "maturity",
    "moneyness",
    "log_moneyness",
    "maturity_bucket",
]


@dataclass(frozen=True)
class SchemaCheck:
    valid: bool
    missing_columns: tuple[str, ...]
    invalid_rows: int


def validate_option_chain_schema(frame: pd.DataFrame) -> SchemaCheck:
    """Check the common option-chain schema shared by live scans and backtests."""

    missing = tuple(column for column in OPTION_CHAIN_COLUMNS if column not in frame.columns)
    invalid_rows = 0
    if not missing and not frame.empty:
        invalid = (
            (frame["strike"] <= 0)
            | (frame["underlying_price"] <= 0)
            | (frame["maturity"] <= 0)
            | (frame["ask"] < frame["bid"])
            | ~frame["option_kind"].isin(["call", "put"])
        )
        invalid_rows = int(invalid.sum())
    return SchemaCheck(valid=not missing and invalid_rows == 0, missing_columns=missing, invalid_rows=invalid_rows)


def enforce_option_chain_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a schema-ordered copy and fail early if critical fields are invalid."""

    check = validate_option_chain_schema(frame)
    if not check.valid:
        pieces = []
        if check.missing_columns:
            pieces.append(f"missing columns: {', '.join(check.missing_columns)}")
        if check.invalid_rows:
            pieces.append(f"invalid rows: {check.invalid_rows}")
        raise ValueError("; ".join(pieces))
    extra_columns = [column for column in frame.columns if column not in OPTION_CHAIN_COLUMNS]
    return frame[OPTION_CHAIN_COLUMNS + extra_columns].copy()
