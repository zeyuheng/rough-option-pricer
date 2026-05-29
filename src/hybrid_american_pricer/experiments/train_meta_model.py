from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


def train_baseline_meta_model(data: pd.DataFrame) -> tuple[RandomForestRegressor, float]:
    y = data["target_price"]
    x = data.drop(columns=["target_price"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return model, float(mean_absolute_percentage_error(y_test, pred) * 100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/pricing_dataset.csv"))
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    _, mape = train_baseline_meta_model(data)
    print(f"baseline meta-model MAPE: {mape:.2f}%")


if __name__ == "__main__":
    main()

