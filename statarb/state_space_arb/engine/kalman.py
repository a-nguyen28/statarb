"""Kalman filter for the time-varying regression  y_t = beta_t * x_t + alpha_t + eps_t.

State s_t = [beta_t, alpha_t] follows a random walk:  s_t = s_{t-1} + w_t,  w ~ N(0, Q = q*I)
Observation:                                          y_t = H_t s_t + v_t,  H_t = [x_t, 1], v ~ N(0, R)

The recursion is inherently sequential, so it's a loop over plain numpy floats with the 2x2
matrix algebra written out by hand (much faster than calling matmul on tiny arrays).

Look-ahead safety
-----------------
`spread` is the *innovation*: y_t - H_t @ s_{t|t-1}, i.e. y_t minus the prediction made
BEFORE seeing y_t. `sigma` is an exponentially-weighted running std of PAST innovations
(excluding bar t itself), so z_t = spread_t / sigma_t is a clean out-of-sample score.

Why not use the model's own sqrt(S_t)? It depends on R, which is only as good as the warmup
estimate, and a short sample of an autocorrelated spread gives a noisy R (z-scores came out
~0.5 instead of ~1 on some seeds). The empirical std self-calibrates.
`beta` and `alpha` are the posterior estimates after seeing bar t (what you'd hedge with next bar).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config.settings import KalmanConfig


class MatrixSingularityError(ArithmeticError):
    """Covariance update failed (degenerate or non-finite inputs)."""


@dataclass(frozen=True)
class StepResult:
    beta: float
    alpha: float
    spread: float
    sigma: float


class KalmanTracker:
    """Streaming filter. One `update()` per bar; identical code path for backtest and live."""

    def __init__(
        self,
        beta0: float,
        alpha0: float,
        p_beta: float,
        p_alpha: float,
        q: float,
        r: float,
        p_cross: float = 0.0,
        sigma_halflife: float = 60.0,
    ) -> None:
        if r <= 0 or q < 0:
            raise ValueError("Need R > 0 and Q >= 0.")
        self.beta, self.alpha = float(beta0), float(alpha0)
        self.p00, self.p01, self.p11 = float(p_beta), float(p_cross), float(p_alpha)
        self.q, self.r = float(q), float(r)
        self._decay = 0.5 ** (1.0 / sigma_halflife)   # weight kept on old variance each bar
        self._var: float | None = None                # EW variance of innovations

    @classmethod
    def from_ols(cls, y: np.ndarray, x: np.ndarray, q: float, r: float | None = None,
                 sigma_halflife: float = 60.0) -> "KalmanTracker":
        """Seed state, prior covariance, and (optionally) R from an OLS fit on a warmup window."""
        y, x = np.asarray(y, float), np.asarray(x, float)
        n = len(y)
        if n < 3:
            raise ValueError("Need at least 3 points to seed the filter.")
        xm = x.mean()
        sxx = float(((x - xm) ** 2).sum())
        if sxx <= 1e-12 * max(1.0, xm**2) * n:
            raise MatrixSingularityError("Reference series is constant; hedge ratio is unidentifiable.")
        beta = float(((x - xm) * (y - y.mean())).sum() / sxx)
        alpha = float(y.mean() - beta * xm)
        s2 = float(((y - beta * x - alpha) ** 2).sum() / (n - 2))
        s2 = max(s2, 1e-12)
        return cls(
            beta0=beta,
            alpha0=alpha,
            p_beta=s2 / sxx,
            p_alpha=s2 * (1 / n + xm**2 / sxx),
            p_cross=-s2 * xm / sxx,
            q=q,
            r=r if r is not None else s2,
            sigma_halflife=sigma_halflife,
        )

    def update(self, y: float, x: float) -> StepResult:
        if not (np.isfinite(y) and np.isfinite(x)):
            raise MatrixSingularityError("Non-finite input to Kalman update.")
        q = self.q
        # Predict (F = I): P <- P + Q
        p00, p01, p11 = self.p00 + q, self.p01, self.p11 + q
        # Innovation
        e = y - (self.beta * x + self.alpha)
        hp0 = x * p00 + p01          # (H P)[0]
        hp1 = x * p01 + p11          # (H P)[1]
        s = x * hp0 + hp1 + self.r   # innovation variance S = H P H' + R
        if not np.isfinite(s) or s <= 0:
            raise MatrixSingularityError(f"Innovation variance is {s}; cannot invert.")
        k0, k1 = hp0 / s, hp1 / s
        # Update
        self.beta += k0 * e
        self.alpha += k1 * e
        self.p00 = p00 - k0 * hp0
        self.p01 = p01 - k0 * hp1
        self.p11 = p11 - k1 * hp1
        # sigma uses only innovations BEFORE this one; seeded with the model's S on the first bar
        sigma = float(np.sqrt(s if self._var is None else self._var))
        self._var = s if self._var is None else self._decay * self._var + (1 - self._decay) * e * e
        return StepResult(self.beta, self.alpha, e, max(sigma, 1e-12))


class KalmanFilterEngine:
    """Batch wrapper: DataFrame in, DataFrame of states out. Knows nothing about data providers."""

    def __init__(self, config: KalmanConfig | None = None) -> None:
        self.config = config or KalmanConfig()

    def compute_states(
        self,
        data: pd.DataFrame,
        target_col: str,
        ref_col: str,
        transition_covariance_q: float | None = None,
        measurement_covariance_r: float | None = None,
    ) -> pd.DataFrame:
        """Return DataFrame[beta, alpha, spread, sigma, zscore] indexed like `data`.

        The first `config.warmup` rows seed the filter (OLS) and are still emitted, but their
        spreads are NOT trustworthy (the seed fit saw them). Trim them before analysing.
        """
        cfg = self.config
        q = cfg.transition_covariance_q if transition_covariance_q is None else transition_covariance_q
        r = cfg.measurement_covariance_r if measurement_covariance_r is None else measurement_covariance_r
        y = data[target_col].to_numpy(float)
        x = data[ref_col].to_numpy(float)
        w = min(cfg.warmup, len(y))
        tracker = KalmanTracker.from_ols(y[:w], x[:w], q=q, r=r, sigma_halflife=cfg.sigma_halflife)

        n = len(y)
        out = np.empty((n, 4))
        for i in range(n):
            res = tracker.update(y[i], x[i])
            out[i] = (res.beta, res.alpha, res.spread, res.sigma)
        df = pd.DataFrame(out, columns=["beta", "alpha", "spread", "sigma"], index=data.index)
        df["zscore"] = df["spread"] / df["sigma"]
        return df
