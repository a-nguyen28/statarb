import logging

import pandas as pd

from state_space_arb.config import SignalConfig, StrategyConfig
from state_space_arb.data import make_synthetic_pair, test_cointegration
from state_space_arb.main import main
from state_space_arb.strategy import Backtester, PaperTrader, ReplayFeed


def test_end_to_end_backtest_is_profitable_on_clean_synthetic_pair():
    d = make_synthetic_pair(n=1500, seed=7)
    assert test_cointegration(d["AAA_close"], d["BBB_close"]).is_cointegrated
    res = Backtester().run(d, "AAA_close", "BBB_close")
    assert res.metrics["num_trades"] > 10
    assert res.metrics["sharpe"] > 1.0
    assert res.metrics["max_drawdown"] > -0.20   # 5% stop + gap risk from t+1 fills
    assert res.total_costs > 0
    assert res.records["equity"].notna().all()


def test_paper_trader_survives_disconnect_and_never_trades_backfill(caplog):
    d = make_synthetic_pair(n=700, seed=11)
    hist, fut = d.iloc[:500], d.iloc[500:]
    feed = ReplayFeed(fut, "AAA_close", "BBB_close", disconnect_at={30, 90}, outage_bars=4)
    tr = PaperTrader.warm_start(hist, "AAA_close", "BBB_close", feed, StrategyConfig(), backoff=0, sleep=lambda s: None)
    with caplog.at_level(logging.INFO, logger="paper"):
        tr.run()
    assert tr.bars_processed == 200
    assert sum("disconnected" in r.message for r in caplog.records) == 2
    assert tr.status == "RUNNING"


def test_cli_synthetic_backtest_exit_code(capsys):
    assert main(["--backtest", "--synthetic"]) == 0
    out = capsys.readouterr().out
    assert "BACKTEST SUMMARY" in out and "p-value" in out


def test_cli_bad_args_fail_before_any_download(capsys):
    import pytest
    with pytest.raises(SystemExit):
        main(["--backtest", "--tickers", "XOM"])


import numpy as np
import pytest


@pytest.mark.parametrize("seed", [7, 11, 21])
def test_runaway_spread_is_stopped_and_loss_is_capped(seed):
    """Cointegration dies at bar 900 and the spread ramps away. Without the dollar stop this
    produced drawdowns of 40-80%, because z-based stops go blind when sigma inflates."""
    d = make_synthetic_pair(n=1200, seed=seed)
    drift = 8 * np.minimum(np.maximum(np.arange(len(d)) - 900, 0), 40)
    for c in ("AAA_close", "AAA_open"):
        d[c] = d[c] + drift
    res = Backtester().run(d, "AAA_close", "BBB_close")
    assert res.structural_breaks >= 1
    assert res.metrics["max_drawdown"] > -0.20   # 5% stop + gap risk from t+1 fills
    base = Backtester(StrategyConfig(signal=SignalConfig(max_loss_pct=None))).run(d, "AAA_close", "BBB_close")
    assert res.metrics["max_drawdown"] > base.metrics["max_drawdown"]


def test_level_jump_is_absorbed_by_filter_and_not_entered_blindly():
    """A one-off +40 jump: the bar with |z| >> stop_z must not trigger an entry."""
    d = make_synthetic_pair(n=1200, seed=7)
    for c in ("AAA_close", "AAA_open"):
        d.iloc[900:, d.columns.get_loc(c)] += 40.0
    rec = Backtester().run(d, "AAA_close", "BBB_close").records
    jump = rec.iloc[900]
    assert abs(jump["z"]) > 5 and jump["position"] == 0 and jump["event"] == "hold"
