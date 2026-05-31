from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor

from hybrid_american_pricer.meta.features import build_feature_row
from hybrid_american_pricer.models import (
    BinomialTreePricer,
    BlackScholesPricer,
    LSMCPricer,
    MonteCarloPricer,
    RoughBergomiPricer,
)
from hybrid_american_pricer.options import MarketState, OptionContract


FairValueModelName = Literal[
    "black_scholes",
    "binomial",
    "lsmc",
    "monte_carlo",
    "rough",
    "ensemble_mean",
    "hybrid_proxy",
    "meta_model",
]


@dataclass(frozen=True)
class FairValueConfig:
    model: FairValueModelName = "binomial"
    tree_steps: int = 300
    rough_paths: int = 1500
    rough_steps: int = 40
    lsmc_paths: int = 2500
    lsmc_steps: int = 40
    mc_paths: int = 5000
    mc_steps: int = 40
    hurst: float = 0.1
    eta: float = 1.8
    rho: float = -0.7
    hybrid_tree_weight: float = 0.55
    hybrid_rough_weight: float = 0.35
    hybrid_bs_weight: float = 0.10
    meta_train_path: str = "data/processed/train.csv"


class FairValueEngine:
    """Compute fair values from interchangeable pricing models for strategy research."""

    def __init__(self, config: FairValueConfig | None = None) -> None:
        self.config = config or FairValueConfig()
        self.bs = BlackScholesPricer()
        self.tree = BinomialTreePricer(steps=self.config.tree_steps)
        self.mc = MonteCarloPricer(
            n_paths=self.config.mc_paths,
            n_steps=self.config.mc_steps,
            seed=103,
        )
        self.lsmc = LSMCPricer(
            n_paths=self.config.lsmc_paths,
            n_steps=self.config.lsmc_steps,
            basis="laguerre",
            degree=3,
            seed=107,
        )
        self.rough = RoughBergomiPricer(
            n_paths=self.config.rough_paths,
            n_steps=self.config.rough_steps,
            hurst=self.config.hurst,
            eta=self.config.eta,
            rho=self.config.rho,
            seed=101,
        )
        self._meta_model: GradientBoostingRegressor | None = None
        self._meta_features: list[str] | None = None

    def price_row(self, row: pd.Series) -> dict[str, float | str]:
        volatility = float(row["market_iv"])
        if not np.isfinite(volatility) or volatility <= 0:
            volatility = 0.25
        contract = OptionContract(
            strike=float(row["strike"]),
            maturity=float(row["maturity"]),
            kind=str(row["option_kind"]),
            exercise=str(row.get("exercise", "american")),
        )
        market = MarketState(
            spot=float(row["underlying_price"]),
            rate=float(row["rate"]),
            volatility=volatility,
            dividend=float(row.get("dividend", 0.0)),
        )
        european_contract = OptionContract(
            strike=contract.strike,
            maturity=contract.maturity,
            kind=contract.kind,
            exercise="european",
        )
        bs_price = self.bs.price(european_contract, market).price
        tree_price = self.tree.price(contract, market).price
        mc_price = np.nan
        lsmc_price = np.nan
        rough_price = np.nan

        if self.config.model in {"monte_carlo", "ensemble_mean", "meta_model"}:
            mc_price = self.mc.price(european_contract, market).price
        if self.config.model in {"lsmc", "ensemble_mean", "meta_model"}:
            lsmc_price = self.lsmc.price(contract, market).price
        if self.config.model in {"rough", "hybrid_proxy", "ensemble_mean", "meta_model"}:
            rough_price = self.rough.price(contract, market).price

        if self.config.model == "black_scholes":
            fair_value = bs_price
        elif self.config.model == "binomial":
            fair_value = tree_price
        elif self.config.model == "monte_carlo":
            fair_value = mc_price
        elif self.config.model == "lsmc":
            fair_value = lsmc_price
        elif self.config.model == "rough":
            fair_value = rough_price
        elif self.config.model == "ensemble_mean":
            fair_value = float(np.nanmean([bs_price, tree_price, mc_price, lsmc_price, rough_price]))
        elif self.config.model == "hybrid_proxy":
            fair_value = (
                self.config.hybrid_tree_weight * tree_price
                + self.config.hybrid_rough_weight * rough_price
                + self.config.hybrid_bs_weight * bs_price
            )
        elif self.config.model == "meta_model":
            fair_value = self._meta_price(row, contract, market, bs_price, tree_price, mc_price, lsmc_price, rough_price)
        else:
            raise ValueError(f"unsupported fair value model: {self.config.model}")

        return {
            "black_scholes_price": float(bs_price),
            "tree_price": float(tree_price),
            "monte_carlo_price": float(mc_price) if np.isfinite(mc_price) else np.nan,
            "lsmc_price": float(lsmc_price) if np.isfinite(lsmc_price) else np.nan,
            "rough_price": float(rough_price) if np.isfinite(rough_price) else np.nan,
            "fair_value": float(fair_value),
            "fair_value_source": self.config.model,
        }

    def _meta_price(
        self,
        row: pd.Series,
        contract: OptionContract,
        market: MarketState,
        bs_price: float,
        tree_price: float,
        mc_price: float,
        lsmc_price: float,
        rough_price: float,
    ) -> float:
        model, features = self._load_meta_model()
        base_prices = {
            "black_scholes": bs_price,
            "tree": tree_price,
            "monte_carlo": mc_price,
            "lsmc": lsmc_price,
            "rough_bergomi_mc": rough_price,
            "rough_bergomi_lsmc": rough_price,
        }
        feature_row = build_feature_row(
            contract,
            market,
            base_prices,
            rough_params={"hurst": self.config.hurst, "eta": self.config.eta, "rho": self.config.rho},
        )
        X = pd.DataFrame([feature_row])
        for feature in features:
            if feature not in X.columns:
                X[feature] = 0.0
        prediction = model.predict(X[features])[0]
        return float(max(prediction, 0.0))

    def _load_meta_model(self) -> tuple[GradientBoostingRegressor, list[str]]:
        if self._meta_model is not None and self._meta_features is not None:
            return self._meta_model, self._meta_features
        path = Path(self.config.meta_train_path)
        if not path.exists():
            raise FileNotFoundError(
                f"meta_model requires {self.config.meta_train_path}; run generate_dataset first"
            )
        data = pd.read_csv(path)
        excluded = {
            "case_id",
            "target_method",
            "target_price",
            "target_runtime_seconds",
            "target_std_error",
            "bid",
            "ask",
            "mid_price",
            "spread",
            "relative_spread",
            "market_iv",
            "model_iv",
            "iv_error",
            "market_iv_clipped",
            "model_iv_clipped",
        }
        features = [
            column
            for column in data.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(data[column])
        ]
        model = GradientBoostingRegressor(random_state=42)
        model.fit(data[features], data["target_price"])
        self._meta_model = model
        self._meta_features = features
        return model, features
