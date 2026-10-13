import numpy as np
import pandas as pd
import pytest

from state_space_arb.config import BacktestConfig, CostModel, SignalConfig, StrategyConfig, KalmanConfig
from state_space_arb.data import make_synthetic_pair
from state_space_arb.engine.kalman import StepResult
from state_space_arb.strategy import Backtester, PairBar, PairsStrategy, Position, SignalStateMachine


# ---------- signal state machine ----------
def run_sm(zs, **kw):
    sm = SignalStateMachine(SignalConfig(**kw))
    return sm, [sm.step(z) for z in zs]


def test_long_entry_then_exit_at_zero():
    _, ev = run_sm([0, -2.5, -1.0, 0.1])
    assert [int(e.position) for e in ev] == [0, 1, 1, 0]
    assert ev[1].reason == "entry" and ev[3].reason == "exit"


def test_short_entry_then_exit():
    _, ev = run_sm([2.2, 1.0, -0.1])
    assert [int(e.position) for e in ev] == [-1, -1, 0]


def test_stop_needs_consecutive_bars_then_cools_down():
    sm, ev = run_sm([-2.5, -5.5, -5.6, -5.7, -2.5, -2.5], stop_bars=3, cooldown_bars=2)
    assert ev[3].reason == "stop" and ev[3].position == Position.FLAT
    assert sm.structural_breaks == 1
    assert ev[4].reason == "cooldown" and ev[5].reason == "cooldown"


def test_stop_counter_resets_if_spread_comes_back():
    _, ev = run_sm([-2.5, -5.5, -3.0, -5.5, -5.5], stop_bars=3)
    assert all(e.reason != "stop" for e in ev)


def test_no_entry_when_already_past_stop():
    _, ev = run_sm([-6.0])
    assert ev[0].position == Position.FLAT


# ---------- hand-checked accounting ----------
class ScriptedTracker:
    """Feeds a fixed z-score script with beta=2 and sigma=1."""
    def __init__(self, zs): self.zs, self.i = list(zs), 0
    def update(self, y, x):
        z = self.zs[self.i]; self.i += 1
        return StepResult(beta=2.0, alpha=0.0, spread=z, sigma=1.0)


def test_trade_tally_and_costs_match_hand_calculation():
    cfg = StrategyConfig(kalman=KalmanConfig(warmup=0),
                         costs=CostModel(commission_per_share=0.005, slippage_per_share=0.01),
                         backtest=BacktestConfig(initial_capital=100_000, allocation=0.5))
    strat = PairsStrategy(ScriptedTracker([0, -3, -3, 0, 0, 0]), cfg)
    idx = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    recs = [strat.on_bar(PairBar(t, 100, 100, 50, 50)) for t in idx]

    # t=1 signal -> fills at t=2.  qty_y = floor(50000 / 100.01) = 499, qty_x = round(2*499) = 998
    assert strat.pf.position == Position.FLAT and len(strat.trades) == 1
    # t=3 exit signal -> fills at t=4.
    # Costs per leg-set: slippage (499+998)*0.01 = 14.97 ; commission 0.005*(499+998) = 7.485 -> 22.455 each way
    assert strat.pf.total_costs == pytest.approx(2 * 22.455)
    # prices never moved, so final equity = start - costs, to the cent
    assert recs[-1].equity == pytest.approx(100_000 - 44.91, abs=1e-6)
    t = strat.trades[0]
    assert t["direction"] == "LONG_SPREAD" and t["exit_reason"] == "exit"
    assert t["entry_time"] == idx[2] and t["exit_time"] == idx[4]      # signal at t, fill at t+1
    assert t["pnl"] == pytest.approx(-44.91, abs=1e-6)


def test_fill_happens_on_bar_after_signal():
    cfg = StrategyConfig(kalman=KalmanConfig(warmup=0))
    strat = PairsStrategy(ScriptedTracker([-3, -3]), cfg)
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    r0 = strat.on_bar(PairBar(idx[0], 100, 100, 50, 50))
    assert r0.position == 0 and strat.pending is not None   # decided, NOT filled
    r1 = strat.on_bar(PairBar(idx[1], 100, 100, 50, 50))
    assert r1.position == 1


# ---------- look-ahead on the full pipeline ----------
def test_backtest_prefix_is_unaffected_by_future_data():
    d = make_synthetic_pair(n=600)
    full = Backtester().run(d, "AAA_close", "BBB_close").records
    d2 = d.copy()
    d2.iloc[400:, d2.columns.get_loc("AAA_close")] *= 2
    d2.iloc[400:, d2.columns.get_loc("AAA_open")] *= 2
    pert = Backtester().run(d2, "AAA_close", "BBB_close").records
    pd.testing.assert_frame_equal(full.iloc[:400], pert.iloc[:400])


def test_open_execution_mode_uses_open_price():
    cfg = StrategyConfig(kalman=KalmanConfig(warmup=0), backtest=BacktestConfig(execution="open"),
                         costs=CostModel(0, 0, 0))
    strat = PairsStrategy(ScriptedTracker([-3, -3]), cfg)
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    strat.on_bar(PairBar(idx[0], 100, 100, 50, 50))
    strat.on_bar(PairBar(idx[1], 90, 100, 45, 50))     # opens at 90/45, closes 100/50
    assert strat.pf.qty_y == int(50_000 // 90) and strat.pf.qty_x == -round(2 * strat.pf.qty_y)
