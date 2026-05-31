# Rough American Hybrid Pricer

American option pricing research framework with Longstaff-Schwartz Monte Carlo,
rough Bergomi-style stochastic volatility simulation, benchmark pricers, and a
hybrid meta-pricer with uncertainty intervals. The market-data and backtest
modules are included as application validation for model-driven mispricing
detection, not as a standalone trading system.

## Goals

- Model American option exercise as an optimal stopping problem.
- Compare Black-Scholes, binomial tree, Monte Carlo, LSMC, and rough Bergomi-style Monte Carlo.
- Train a hybrid meta-model that fuses base pricer outputs and market features.
- Evaluate point error, uncertainty coverage, interval width, calibration, and runtime.
- Produce interpretable artifacts such as exercise boundaries and Hurst sensitivity plots.
- Validate pricing outputs on real option chains through market mispricing detection.

## Quick Start

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"
pytest
python -m hybrid_american_pricer.experiments.run_benchmarks
```

## Layout

```text
configs/                         Experiment configuration
src/hybrid_american_pricer/
  options/                       Contracts and payoff functions
  models/                        Black-Scholes, tree, MC, LSMC, rough Bergomi
  meta/                          Features, MC-dropout model, calibration
  data/                          Option-chain loaders and standardization
  strategy/                      Fair-value and mispricing signal construction
  backtest/                      Application validation for mispricing signals
  experiments/                   CLI experiment entry points
  visualization/                 Plotting helpers
  utils/                         Metrics and timing
tests/                           Smoke and correctness tests
reports/                         Generated figures and tables
```

## Core Commands

```bash
python -m hybrid_american_pricer.experiments.generate_dataset
python -m hybrid_american_pricer.experiments.run_benchmarks
python -m hybrid_american_pricer.experiments.lsmc_study
python -m hybrid_american_pricer.experiments.rough_bergomi_study
python -m hybrid_american_pricer.experiments.train_meta_model
python -m hybrid_american_pricer.experiments.evaluate
python -m hybrid_american_pricer.experiments.uncertainty_calibration
python -m hybrid_american_pricer.experiments.option_chain_signals
python -m hybrid_american_pricer.experiments.backtest_mispricing_strategy
python -m hybrid_american_pricer.experiments.sensitivity
```

The dataset command writes `pricing_dataset.csv`, `train.csv`, `valid.csv`,
`test.csv`, and a parameter distribution figure for the hybrid meta-model stage.

The option-chain scanner writes `data/processed/option_chain_signals.csv` and
`reports/tables/option_chain_signal_summary.csv`. By default it uses a
deterministic sample chain, so the trading-signal pipeline can run offline. Use
`--source csv --input path/to/chain.csv` for local market data, or install
`yfinance` and use `--source yfinance` for delayed Yahoo Finance chains.

## Market Data Layer

Live scans and future historical backtests share one normalized option-chain
schema:

```text
ticker, quote_date, expiration, strike, option_kind, bid, ask, mid_price,
market_iv, volume, open_interest, underlying_price, rate, dividend
```

Provider classes live under `hybrid_american_pricer.data.providers`:

- `SampleLiveOptionChainProvider`: deterministic offline live snapshot.
- `CsvLiveOptionChainProvider`: current snapshot from a local CSV export.
- `YFinanceLiveOptionChainProvider`: delayed Yahoo Finance chain if `yfinance`
  is installed.
- `CsvHistoricalOptionChainProvider`: one or many quote-date snapshots for
  backtesting research.

`OptionChainStore` can save and reload normalized snapshots under `data/market/`.

## Market Mispricing Application

This module is an application demo for the hybrid American option pricer. It
uses model fair values to detect gaps versus market bid/ask quotes:

```text
fair_value > ask  -> potentially underpriced
fair_value < bid  -> potentially overpriced
```

The simple long-only backtest is used to check whether these signals have
historical economic meaning after bid-ask costs. It is not the primary project
objective and should not be interpreted as a production trading strategy.

Run the offline application validation:

```bash
python -m hybrid_american_pricer.experiments.backtest_mispricing_strategy
```

It writes `reports/tables/trades.csv`, `reports/tables/backtest_summary.csv`,
and `reports/tables/equity_curve.csv`. Use `--source csv --input path/to/history.csv`
or `--source onclick` when historical option-chain snapshots are available.

The backtest now runs multiple strategy variants by default:

```text
baseline
call_only
put_only
atm_tight
tight_spread_high_edge
shorter_dte
```

Choose a fair-value source:

```bash
python -m hybrid_american_pricer.experiments.backtest_mispricing_strategy --fair-value-model binomial
python -m hybrid_american_pricer.experiments.backtest_mispricing_strategy --fair-value-model lsmc
python -m hybrid_american_pricer.experiments.backtest_mispricing_strategy --fair-value-model rough
python -m hybrid_american_pricer.experiments.backtest_mispricing_strategy --fair-value-model hybrid_proxy
python -m hybrid_american_pricer.experiments.backtest_mispricing_strategy --fair-value-model meta_model
python -m hybrid_american_pricer.experiments.backtest_mispricing_strategy --fair-value-model all
```

Available fair-value sources are `black_scholes`, `binomial`, `monte_carlo`,
`lsmc`, `rough`, `ensemble_mean`, `hybrid_proxy`, and `meta_model`.
`meta_model` trains a Gradient Boosting meta-pricer from
`data/processed/train.csv` at runtime; it is useful for model comparison, but it
is still trained on the project's synthetic rough-volatility target until real
historical option labels are added.

The stricter application variant `hybrid_mispricing_regime` adds realistic
filters for the mispricing validation:

```text
21-45 DTE, 0.95-1.05 moneyness, spread <= 8%, relative edge >= 8%,
top 20% confidence, call-uptrend / put-downtrend filter, +30% take-profit,
and -30% stop-loss.
```

See `docs/historical_data_schema.md` for the CSV format.

Real historical option-chain data can be pulled through ORATS if you have an API
token:

```powershell
$env:ORATS_TOKEN="your-secret-token"
python -m hybrid_american_pricer.experiments.backtest_mispricing_strategy --source orats --ticker SPY --start 2024-01-03 --end 2024-01-31 --fair-value-model all
```

The project also supports free OnclickMedia historical option chains:

```powershell
python -m hybrid_american_pricer.experiments.backtest_mispricing_strategy --source onclick --ticker SPY --start 2025-01-03 --end 2025-01-10 --fair-value-model binomial
```

OnclickMedia requires no API key, but treat it as a free research source: verify
coverage, gaps, and quote quality before drawing trading conclusions.

See `docs/project_scope.md` for how the pricing core and application validation
fit together.
