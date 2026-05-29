# Rough American Hybrid Pricer

American option pricing research framework with Longstaff-Schwartz Monte Carlo,
rough Bergomi-style stochastic volatility simulation, benchmark pricers, and a
hybrid meta-pricer with uncertainty intervals.

## Goals

- Model American option exercise as an optimal stopping problem.
- Compare Black-Scholes, binomial tree, Monte Carlo, LSMC, and rough Bergomi-style Monte Carlo.
- Train a hybrid meta-model that fuses base pricer outputs and market features.
- Evaluate point error, uncertainty coverage, interval width, calibration, and runtime.
- Produce interpretable artifacts such as exercise boundaries and Hurst sensitivity plots.

## Quick Start

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev,ml]"
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
python -m hybrid_american_pricer.experiments.train_meta_model
python -m hybrid_american_pricer.experiments.evaluate
python -m hybrid_american_pricer.experiments.sensitivity
```

