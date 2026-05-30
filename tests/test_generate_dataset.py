from pathlib import Path

from hybrid_american_pricer.experiments.generate_dataset import (
    generate_dataset,
    split_dataset,
    write_dataset_outputs,
)


def test_generate_dataset_contains_stage4_required_columns() -> None:
    data = generate_dataset(
        n_samples=8,
        seed=7,
        base_tree_steps=40,
        mc_paths=200,
        lsmc_paths=200,
        rough_paths=150,
        target_rough_paths=250,
        n_steps=10,
        target_n_steps=12,
    )
    required = {
        "case_id",
        "spot",
        "strike",
        "maturity",
        "rate",
        "volatility",
        "hurst",
        "eta",
        "rho",
        "price_black_scholes",
        "price_tree",
        "price_monte_carlo",
        "price_lsmc",
        "price_rough_bergomi_mc",
        "price_rough_bergomi_lsmc",
        "bid",
        "ask",
        "mid_price",
        "spread",
        "relative_spread",
        "market_iv",
        "model_iv",
        "iv_error",
        "log_moneyness",
        "maturity_bucket_short",
        "maturity_bucket_medium",
        "maturity_bucket_long",
        "dividend",
        "target_price",
        "target_method",
    }
    assert required.issubset(data.columns)
    assert len(data) == 8
    assert (data["target_price"] >= 0).all()
    assert data["target_method"].str.startswith("rough_bergomi_lsmc").all()
    assert (data["ask"] >= data["bid"]).all()
    assert (data["relative_spread"] >= 0).all()
    assert data["case_id"].is_unique


def test_split_dataset_partitions_all_rows() -> None:
    data = generate_dataset(
        n_samples=10,
        seed=8,
        base_tree_steps=30,
        mc_paths=100,
        lsmc_paths=100,
        rough_paths=100,
        target_rough_paths=120,
        n_steps=8,
        target_n_steps=8,
    )
    train, valid, test = split_dataset(data, seed=8)
    assert len(train) + len(valid) + len(test) == len(data)
    assert len(train) == 7
    assert len(valid) == 1
    assert len(test) == 2


def test_write_dataset_outputs(tmp_path: Path) -> None:
    data = generate_dataset(
        n_samples=6,
        seed=9,
        base_tree_steps=20,
        mc_paths=80,
        lsmc_paths=80,
        rough_paths=80,
        target_rough_paths=100,
        n_steps=6,
        target_n_steps=6,
    )
    paths = write_dataset_outputs(
        data,
        output_dir=tmp_path / "processed",
        figure_path=tmp_path / "figures" / "distributions.png",
        seed=9,
    )
    for path in paths.values():
        assert path.exists()
