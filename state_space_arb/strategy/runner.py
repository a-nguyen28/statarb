"""PairsStrategy: the single per-bar brain shared by the backtester and the paper trader.

Per bar t, in this order (this ordering IS the look-ahead protection):
  1. execute the order decided at bar t-1 at bar t's open/close
  2. feed bar t's close to the Kalman filter
  3. mark equity
  4. run the signal state machine on z_t -> maybe queue an order for bar t+1
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config.settings import StrategyConfig
from ..engine.kalman import KalmanTracker
from .portfolio import Portfolio
from .signals import Position, SignalStateMachine


@dataclass(frozen=True)
class PairBar:
    timestamp: pd.Timestamp
    y_open: float
    y_close: float
    x_open: float
    x_close: float
    is_backfill: bool = False   # True = replayed after a disconnect; observe only, never trade


@dataclass(frozen=True)
class BarRecord:
    timestamp: pd.Timestamp
    beta: float
    alpha: float
    spread: float
    sigma: float
    z: float
    position: int
    equity: float
    event: str


class PairsStrategy:
    def __init__(self, tracker: KalmanTracker, cfg: StrategyConfig) -> None:
        self.cfg = cfg
        self.tracker = tracker
        self.sm = SignalStateMachine(cfg.signal)
        self.pf = Portfolio(cfg.backtest.initial_capital, cfg.costs)
        self.warmup = cfg.kalman.warmup
        self.bars_seen = 0
        self.pending: tuple[Position, float, str] | None = None
        self.trades: list[dict] = []
        self._open_trade: dict | None = None

    @classmethod
    def bootstrap(cls, y_hist: np.ndarray, x_hist: np.ndarray, cfg: StrategyConfig) -> "PairsStrategy":
        k = cfg.kalman
        w = min(k.warmup, len(y_hist))
        tracker = KalmanTracker.from_ols(y_hist[:w], x_hist[:w], q=k.transition_covariance_q,
                                         r=k.measurement_covariance_r, sigma_halflife=k.sigma_halflife)
        return cls(tracker, cfg)

    # -- execution ----------------------------------------------------------------
    def _exec_prices(self, bar: PairBar) -> tuple[float, float]:
        if self.cfg.backtest.execution == "open":
            return bar.y_open, bar.x_open
        return bar.y_close, bar.x_close

    def _execute(self, bar: PairBar) -> None:
        target, beta, reason = self.pending  # type: ignore[misc]
        py, px = self._exec_prices(bar)
        pf = self.pf
        if pf.position != Position.FLAT:
            pf.close_spread(py, px)
            t = self._open_trade
            if t is not None:
                t.update(exit_time=bar.timestamp, pnl=pf.equity(py, px) - t.pop("entry_equity"), exit_reason=reason)
                self.trades.append(t)
                self._open_trade = None
        if target != Position.FLAT:
            eq_before = pf.equity(py, px)
            if pf.open_spread(target, py, px, beta, self.cfg.backtest.allocation):
                self._open_trade = {
                    "entry_time": bar.timestamp, "direction": target.name,
                    "entry_equity": eq_before, "entry_beta": beta,
                }
            else:
                self.sm.position = Position.FLAT   # order couldn't be sized; keep state machine honest

    def _hard_stop(self, equity: float):
        """Dollar loss limit on the open trade. Returns a SignalEvent if it fired, else None."""
        limit = self.cfg.signal.max_loss_pct
        t = self._open_trade
        if limit is None or t is None or self.pf.position == Position.FLAT or self.sm.position == Position.FLAT:
            return None
        if equity - t["entry_equity"] <= -limit * t["entry_equity"]:
            return self.sm.force_stop("loss_stop")
        return None

    # -- main entry ---------------------------------------------------------------
    def on_bar(self, bar: PairBar, trade: bool = True) -> BarRecord:
        trade = trade and not bar.is_backfill
        if trade and self.pending is not None:
            self._execute(bar)
        self.pending = None   # a pending order is only valid for the very next bar

        res = self.tracker.update(bar.y_close, bar.x_close)
        z = res.spread / res.sigma
        equity = self.pf.equity(bar.y_close, bar.x_close)

        event = "warmup" if self.bars_seen < self.warmup else "observe"
        if trade and self.bars_seen >= self.warmup:
            ev = self._hard_stop(equity) or self.sm.step(z)
            event = ev.reason
            if ev.position != self.pf.position:
                self.pending = (ev.position, res.beta, ev.reason)
        self.bars_seen += 1
        return BarRecord(bar.timestamp, res.beta, res.alpha, res.spread, res.sigma, z,
                         int(self.pf.position), equity, event)
