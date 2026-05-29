from hybrid_american_pricer.models import BlackScholesPricer
from hybrid_american_pricer.options import MarketState, OptionContract


def test_black_scholes_put_price_is_positive() -> None:
    result = BlackScholesPricer().price(
        OptionContract(strike=100, maturity=1, kind="put", exercise="european"),
        MarketState(spot=100, rate=0.05, volatility=0.2),
    )
    assert result.price > 0

