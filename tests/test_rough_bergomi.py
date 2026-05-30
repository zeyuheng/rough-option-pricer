import numpy as np

from hybrid_american_pricer.experiments.rough_bergomi_study import (
    roughness_score,
    run_hurst_sensitivity,
    run_rough_price_table,
    summarize_stage3,
)
from hybrid_american_pricer.models import RoughBergomiPricer
from hybrid_american_pricer.models.path_simulation import simulate_rough_bergomi_paths
from hybrid_american_pricer.options import MarketState, OptionContract


def test_rough_bergomi_paths_are_martingale_corrected() -> None:
    market = MarketState(spot=100, rate=0.05, volatility=0.2)
    paths, variance = simulate_rough_bergomi_paths(
        market, maturity=1.0, n_paths=1500, n_steps=30, hurst=0.1, seed=7
    )
    expected_forward = market.spot * np.exp(market.rate)
    assert paths.shape == variance.shape == (1500, 31)
    assert abs(paths[:, -1].mean() - expected_forward) / expected_forward < 0.02
    assert (variance > 0).all()


def test_smaller_hurst_has_larger_roughness_score() -> None:
    market = MarketState(spot=100, rate=0.05, volatility=0.2)
    _, low_h_variance = simulate_rough_bergomi_paths(
        market, maturity=1.0, n_paths=1000, n_steps=40, hurst=0.05, seed=11
    )
    _, high_h_variance = simulate_rough_bergomi_paths(
        market, maturity=1.0, n_paths=1000, n_steps=40, hurst=0.45, seed=11
    )
    assert roughness_score(low_h_variance) > roughness_score(high_h_variance)


def test_rough_bergomi_pricer_supports_american_lsmc() -> None:
    result = RoughBergomiPricer(n_paths=1000, n_steps=20, hurst=0.1, seed=5).price(
        OptionContract(strike=100, maturity=1, kind="put", exercise="american"),
        MarketState(spot=100, rate=0.05, volatility=0.2),
    )
    assert result.price > 0
    assert result.metadata["hurst"] == 0.1
    assert "exercise_boundaries" in result.metadata


def test_stage3_summary_has_required_signals() -> None:
    price_table = run_rough_price_table(n_paths=800, n_steps=20, seed=3)
    sensitivity = run_hurst_sensitivity(n_paths=800, n_steps=20, seed=3)
    summary = summarize_stage3(price_table, sensitivity)
    assert summary["rough_american_lsmc_available"] is True
    assert summary["hurst_vs_roughness_correlation"] < 0
    assert summary["max_abs_rough_price_shift"] > 0

