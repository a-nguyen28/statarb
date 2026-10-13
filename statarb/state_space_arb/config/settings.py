"""All tunable parameters live here so nothing is hardcoded in the logic modules."""
from __future__ import annotations

from dataclasses import dataclass, field

PERIODS_PER_YEAR: dict[str, int] = {
    "1d": 252,
    "1h": 252 * 7,
    "30m": 252 * 13,
    "15m": 252 * 26,
    "5m": 252 * 78,
    "1m": 252 * 390,
}


@dataclass(frozen=True)
class KalmanConfig:
    """Filter tuning.

    transition_covariance_q: random-walk variance of the state (beta, alpha) per bar.
        Bigger = hedge ratio adapts faster but gets noisier. Smaller = smoother but laggy.
    measurement_covariance_r: observation noise variance (price units squared).
        None = estimate from an OLS fit over the warmup window.
    warmup: bars used to seed the filter. No trades are placed during warmup.
    sigma_halflife: half-life (bars) of the exponentially-weighted std of the spread.
    """

    transition_covariance_q: float = 1e-5
    measurement_covariance_r: float | None = None
    warmup: int = 50
    sigma_halflife: float = 60.0   # bars; half-life of the running std used for z-scores


@dataclass(frozen=True)
class SignalConfig:
    entry_z: float = 2.0          # enter when |z| >= this
    exit_z: float = 0.0           # exit when z crosses this (0 = mean)
    stop_z: float = 5.0           # adverse excursion that counts toward a structural break
    stop_bars: int = 3            # consecutive bars beyond stop_z before we bail
    cooldown_bars: int = 20       # no new entries for this many bars after a stop-out
    max_holding_bars: int | None = None  # optional divergence timeout
    max_loss_pct: float | None = 0.05     # hard stop: exit if a trade is down this much of its entry equity.
                                          # Independent of z, because a runaway spread inflates sigma and keeps z small.


@dataclass(frozen=True)
class CostModel:
    commission_per_share: float = 0.005   # dollars per share
    commission_pct: float = 0.0           # e.g. 0.0005 = 0.05% of notional
    slippage_per_share: float = 0.01      # one tick, always against you

    def commission(self, shares: float, price: float) -> float:
        return shares * self.commission_per_share + shares * price * self.commission_pct


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    allocation: float = 0.5               # fraction of equity put into the target leg
    execution: str = "close"              # fill at "open" or "close" of bar t+1
    interval: str = "1d"

    @property
    def periods_per_year(self) -> int:
        return PERIODS_PER_YEAR.get(self.interval, 252)


@dataclass(frozen=True)
class StrategyConfig:
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    costs: CostModel = field(default_factory=CostModel)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
