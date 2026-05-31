# Historical Option Chain Data Schema

Use this CSV format when running a real historical backtest:

```text
quote_date,ticker,expiration,strike,option_type,bid,ask,market_iv,volume,open_interest,underlying_price,rate,dividend
2024-01-03,SPY,2024-02-16,470,call,6.10,6.25,0.182,1200,18000,468.79,0.045,0.014
2024-01-03,SPY,2024-02-16,470,put,7.35,7.50,0.191,980,16500,468.79,0.045,0.014
```

Required columns:

```text
quote_date
ticker
expiration
strike
option_type
bid
ask
market_iv
volume
open_interest
underlying_price
```

Optional columns:

```text
rate
dividend
last
```

Recommended first real-data vendors:

```text
ORATS EOD history
ThetaData EOD option chains
Polygon historical option quotes/chains
Databento OPRA CBBO or OHLCV for more advanced intraday research
```

Run a CSV market mispricing validation:

```bash
python -m hybrid_american_pricer.experiments.backtest_mispricing_strategy --source csv --input data/raw/spy_history.csv --ticker SPY --underlying-price 470
```

Run an ORATS historical EOD mispricing validation:

```bash
$env:ORATS_TOKEN="your-secret-token"
python -m hybrid_american_pricer.experiments.backtest_mispricing_strategy --source orats --ticker SPY --start 2024-01-03 --end 2024-01-31 --fair-value-model all
```

The ORATS provider uses the official `datav2/hist/strikes` endpoint and requests
21-60 DTE contracts by default. It converts ORATS fields such as
`tradeDate`, `expirDate`, `stockPrice`, `callBidPrice`, `callAskPrice`,
`putBidPrice`, `putAskPrice`, `callMidIv`, and `putMidIv` into the project's
standard option-chain schema.

Run a free OnclickMedia historical EOD mispricing validation:

```bash
python -m hybrid_american_pricer.experiments.backtest_mispricing_strategy --source onclick --ticker SPY --start 2025-01-03 --end 2025-01-10 --fair-value-model binomial
```

The OnclickMedia provider uses the free `https://api.onclickmedia.com/options/`
endpoint. No API key is required. The provider requests CSV data and converts
fields such as `expiration`, `strike`, `type`, `last`, `bid`, `ask`, `volume`,
`open_interest`, and `implied_volatility` into the same project schema.

The backtest writes:

```text
reports/tables/trades.csv
reports/tables/backtest_summary.csv
reports/tables/equity_curve.csv
reports/tables/backtest_diagnostics_*.csv
```
