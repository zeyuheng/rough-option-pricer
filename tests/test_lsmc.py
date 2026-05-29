from hybrid_american_pricer.models import LSMCPricer
from hybrid_american_pricer.options import MarketState, OptionContract


def test_lsmc_returns_positive_price() -> None:
    result = LSMCPricer(n_paths=1000, n_steps=20, seed=7).price(
        OptionContract(strike=100, maturity=1, kind="put", exercise="american"),
        MarketState(spot=100, rate=0.05, volatility=0.2),
    )
    assert result.price > 0
    assert "exercise_boundaries" in result.metadata

