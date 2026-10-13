"""Synthetic cointegrated pair with a drifting hedge ratio. For tests and offline demos."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_pair(
    n: int = 1500,
    seed: int = 7,
    target: str = "AAA",
    ref: str = "BBB",
    beta0: float = 1.2,
    beta_vol: float = 3e-4,
    alpha: float = 5.0,
    ou_theta: float = 0.10,
    ou_sigma: float = 0.6,
    freq: str = "B",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = 100 * np.exp(np.cumsum(rng.normal(2e-4, 0.01, n)))
    beta = beta0 + np.cumsum(rng.normal(0, beta_vol, n))        # hidden, slowly wandering hedge ratio
    s = np.zeros(n)
    for t in range(1, n):                                        # Ornstein-Uhlenbeck spread
        s[t] = s[t - 1] * (1 - ou_theta) + ou_sigma * rng.normal()
    y = beta * x + alpha + s
    idx = pd.date_range("2019-01-01", periods=n, freq=freq, tz="UTC")
    jitter = lambda p: p * (1 + rng.normal(0, 5e-4, n))
    return pd.DataFrame(
        {f"{target}_open": jitter(y), f"{target}_close": y, f"{ref}_open": jitter(x), f"{ref}_close": x},
        index=idx,
    )
