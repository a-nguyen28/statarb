"""Event-loop backtester built on PairsStrategy, plus metrics and a text report."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config.settings import StrategyConfig
from .runner import BarRecord, PairBar, PairsStrategy


@dataclass
class BacktestResult:
    records: pd.DataFrame          # per-bar states, z, position, equity
    trades: pd.DataFrame           # one row per round trip
    metrics: dict[str, float]
    structural_breaks: int
    total_costs: float


def compute_metrics(equity: pd.Series, trades: pd.DataFrame, periods_per_year: int,
                    buy_hold_return: float | None = None) -> dict[str, float]:
    rets = equity.pct_change().dropna()
    sd = rets.std(ddof=1)
    sharpe = float(rets.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0
    dd = equity / equity.cummax() - 1
    m = {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()),
        "num_trades": float(len(trades)),
        "win_rate": float((trades["pnl"] > 0).mean()) if len(trades) else float("nan"),
    }
    if buy_hold_return is not None:
        m["buy_hold_return"] = buy_hold_return
    return m


class Backtester:
    def __init__(self, cfg: StrategyConfig | None = None) -> None:
        self.cfg = cfg or StrategyConfig()

    def run(self, frame: pd.DataFrame, y_col: str, x_col: str) -> BacktestResult:
        """`y_col`/`x_col` are close columns like 'XOM_close'; matching '_open' columns are used if present."""
        y_open_col, x_open_col = y_col.replace("_close", "_open"), x_col.replace("_close", "_open")
        yc, xc = frame[y_col].to_numpy(float), frame[x_col].to_numpy(float)
        yo = frame[y_open_col].to_numpy(float) if y_open_col in frame else yc
        xo = frame[x_open_col].to_numpy(float) if x_open_col in frame else xc
        strat = PairsStrategy.bootstrap(yc, xc, self.cfg)

        recs: list[BarRecord] = []
        for i, ts in enumerate(frame.index):
            recs.append(strat.on_bar(PairBar(ts, yo[i], yc[i], xo[i], xc[i])))

        df = pd.DataFrame(recs).set_index("timestamp")
        trades = pd.DataFrame(strat.trades)
        if trades.empty:
            trades = pd.DataFrame(columns=["entry_time", "exit_time", "direction", "pnl", "exit_reason", "entry_beta"])
        bh = float(yc[-1] / yc[0] - 1)
        metrics = compute_metrics(df["equity"], trades, self.cfg.backtest.periods_per_year, bh)
        return BacktestResult(df, trades, metrics, strat.sm.structural_breaks, strat.pf.total_costs)


def format_report(res: BacktestResult, cfg: StrategyConfig, title: str = "") -> str:
    m = res.metrics
    wr = "n/a" if np.isnan(m["win_rate"]) else f"{m['win_rate']:.1%}"
    bar = "=" * 52
    lines = [
        bar, f" BACKTEST SUMMARY  {title}", bar,
        f" Total return      : {m['total_return']:>10.2%}",
        f" Buy & hold (tgt)  : {m.get('buy_hold_return', float('nan')):>10.2%}",
        f" Sharpe (ann.)     : {m['sharpe']:>10.2f}",
        f" Max drawdown      : {m['max_drawdown']:>10.2%}",
        f" Round trips       : {int(m['num_trades']):>10d}",
        f" Win rate          : {wr:>10}",
        f" Costs paid ($)    : {res.total_costs:>10,.2f}",
        f" Structural breaks : {res.structural_breaks:>10d}",
        f" Final equity ($)  : {res.records['equity'].iloc[-1]:>10,.2f}",
        bar,
    ]
    if len(res.trades):
        by = res.trades.groupby("exit_reason")["pnl"].agg(["count", "mean"])
        lines.append(" Exit reasons (count, avg pnl $):")
        for reason, row in by.iterrows():
            lines.append(f"   {reason:<10} {int(row['count']):>5d}  {row['mean']:>10,.2f}")
        lines.append(bar)
    return "\n".join(lines)


def plot_equity(res: BacktestResult, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
    r = res.records
    ax[0].plot(r.index, r["equity"]); ax[0].set_title("Equity curve")
    ax[1].plot(r.index, r["z"], lw=0.7); ax[1].axhline(0, c="k", lw=0.5)
    for lv in (-2, 2): ax[1].axhline(lv, c="r", ls="--", lw=0.5)
    ax[1].set_title("Spread z-score")
    ax[2].plot(r.index, r["beta"]); ax[2].set_title("Dynamic hedge ratio (beta)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
