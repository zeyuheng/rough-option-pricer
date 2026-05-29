from __future__ import annotations

import math

import numpy as np

from hybrid_american_pricer.options.instruments import MarketState


def simulate_black_scholes_paths(
    market: MarketState,
    maturity: float,
    n_paths: int,
    n_steps: int,
    seed: int | None = None,
    antithetic: bool = True,
) -> np.ndarray:
    """Simulate geometric Brownian motion paths shaped as (n_paths, n_steps + 1)."""

    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    effective_paths = n_paths // 2 if antithetic else n_paths
    shocks = rng.normal(size=(effective_paths, n_steps))
    if antithetic:
        shocks = np.vstack([shocks, -shocks])
    shocks = shocks[:n_paths]

    drift = (market.rate - market.dividend - 0.5 * market.volatility**2) * dt
    diffusion = market.volatility * math.sqrt(dt) * shocks
    log_paths = np.cumsum(drift + diffusion, axis=1)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = market.spot
    paths[:, 1:] = market.spot * np.exp(log_paths)
    return paths


def fractional_gaussian_noise(
    n_paths: int,
    n_steps: int,
    hurst: float,
    maturity: float,
    seed: int | None = None,
) -> np.ndarray:
    """Generate approximate fractional Gaussian noise using covariance Cholesky."""

    if not 0.0 < hurst < 1.0:
        raise ValueError("hurst must be between 0 and 1")
    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    idx = np.arange(n_steps)
    cov = 0.5 * (
        np.abs(idx[:, None] - idx[None, :] + 1) ** (2 * hurst)
        - 2 * np.abs(idx[:, None] - idx[None, :]) ** (2 * hurst)
        + np.abs(idx[:, None] - idx[None, :] - 1) ** (2 * hurst)
    )
    cov *= dt ** (2 * hurst)
    chol = np.linalg.cholesky(cov + 1e-12 * np.eye(n_steps))
    return rng.normal(size=(n_paths, n_steps)) @ chol.T


def simulate_rough_bergomi_paths(
    market: MarketState,
    maturity: float,
    n_paths: int,
    n_steps: int,
    hurst: float = 0.1,
    eta: float = 1.8,
    rho: float = -0.7,
    xi0: float | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate rough Bergomi-style asset and variance paths."""

    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be in [-1, 1]")
    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    xi0 = market.volatility**2 if xi0 is None else xi0

    fgn = fractional_gaussian_noise(n_paths, n_steps, hurst, maturity, seed=seed)
    rough_driver = np.cumsum(fgn, axis=1)
    times = np.linspace(dt, maturity, n_steps)
    variance = xi0 * np.exp(eta * rough_driver - 0.5 * eta**2 * times ** (2 * hurst))
    variance = np.maximum(variance, 1e-10)

    independent = rng.normal(size=(n_paths, n_steps))
    normalized_fgn = fgn / np.maximum(np.std(fgn, axis=0, keepdims=True), 1e-12)
    asset_shocks = rho * normalized_fgn + math.sqrt(max(1.0 - rho**2, 0.0)) * independent
    log_returns = (
        (market.rate - market.dividend - 0.5 * variance) * dt
        + np.sqrt(variance * dt) * asset_shocks
    )

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = market.spot
    paths[:, 1:] = market.spot * np.exp(np.cumsum(log_returns, axis=1))
    variance_paths = np.empty_like(paths)
    variance_paths[:, 0] = xi0
    variance_paths[:, 1:] = variance
    return paths, variance_paths

