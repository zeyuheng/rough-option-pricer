# Project Scope

## Core Objective

This project is a hybrid American option pricing framework. Its main objective
is to estimate and explain American option fair value by combining:

```text
Black-Scholes benchmarks
Binomial Tree American pricing
Longstaff-Schwartz Monte Carlo
rough Bergomi-style stochastic volatility
hybrid/meta-model pricing
uncertainty intervals and conformal calibration
```

The core research question is:

```text
Can multiple pricing engines be fused into an interpretable and uncertainty-aware
American option pricer?
```

## Main Research Stages

```text
Stage 1: Classical benchmark pricers
Stage 2: LSMC American optimal stopping
Stage 3: rough Bergomi-style stochastic volatility
Stage 4: pricing dataset and target construction
Stage 5: hybrid meta-model / MC Dropout model
Stage 6: uncertainty intervals and calibration
Stage 7: real option-chain data interface
```

These stages are the main deliverables.

## Application Validation

The market-data and backtest code is an application validation module:

```text
Stage 8: Market mispricing detection demo
```

Its role is to connect the pricer to real option chains and compare:

```text
model fair value
market bid / ask / mid
```

The simple long-only backtest is included to test whether model-generated
mispricing signals have historical economic meaning after bid-ask costs. It is
not presented as a production trading strategy or a claim of live profitability.

## Recommended Presentation

When presenting the project, treat the trading-related outputs as secondary:

```text
Primary: pricing accuracy, model comparison, exercise behavior, uncertainty.
Secondary: real-market mispricing signal and application validation.
```

This keeps the project centered on the hybrid American option pricer while still
showing that the model can be connected to real market data.
