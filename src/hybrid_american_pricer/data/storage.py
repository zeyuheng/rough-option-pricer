from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hybrid_american_pricer.data.schema import enforce_option_chain_schema


@dataclass(frozen=True)
class OptionChainStore:
    """Tiny local store for normalized option-chain snapshots."""

    root: Path = Path("data/market")

    def snapshot_path(self, ticker: str, quote_date: str) -> Path:
        return self.root / ticker.upper() / f"{quote_date}.csv"

    def save_snapshot(self, frame: pd.DataFrame) -> Path:
        checked = enforce_option_chain_schema(frame)
        if checked.empty:
            raise ValueError("cannot save an empty option-chain snapshot")
        ticker = str(checked["ticker"].iloc[0]).upper()
        quote_date = str(checked["quote_date"].iloc[0])
        path = self.snapshot_path(ticker, quote_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        checked.to_csv(path, index=False)
        return path

    def load_snapshot(self, ticker: str, quote_date: str) -> pd.DataFrame:
        return enforce_option_chain_schema(pd.read_csv(self.snapshot_path(ticker, quote_date)))
