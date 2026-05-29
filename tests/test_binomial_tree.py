from hybrid_american_pricer.models import BinomialTreePricer
from hybrid_american_pricer.options import MarketState, OptionContract


def test_american_put_is_at_least_european_put_proxy() -> None:
    market = MarketState(spot=100, rate=0.05, volatility=0.2)
    american = BinomialTreePricer(steps=200).price(
        OptionContract(strike=100, maturity=1, kind="put", exercise="american"), market
    )
    european = BinomialTreePricer(steps=200).price(
        OptionContract(strike=100, maturity=1, kind="put", exercise="european"), market
    )
    assert american.price >= european.price

