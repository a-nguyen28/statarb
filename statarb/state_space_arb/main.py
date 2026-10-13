"""CLI entry point.

  python main.py --backtest --tickers XOM CVX --start 2023-01-01
  python main.py --backtest --synthetic --plot equity.png
  python main.py --live --tickers XOM CVX --interval 1m
  python main.py --live --synthetic            # mock feed with a simulated network drop
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from .config.settings import BacktestConfig, CostModel, KalmanConfig, SignalConfig, StrategyConfig
from .data.cointegration import test_cointegration
from .data.downloader import DataDownloader
from .data.errors import DataError
from .data.synthetic import make_synthetic_pair
from .engine.kalman import MatrixSingularityError
from .strategy.backtester import Backtester, format_report, plot_equity
from .strategy.paper import FeedDisconnected, PaperTrader, ReplayFeed, YFinanceLiveFeed

RED, RESET = "\033[31m", "\033[0m"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="State-space statistical arbitrage engine")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backtest", action="store_true")
    mode.add_argument("--live", action="store_true", help="paper trading (no real orders, ever)")
    p.add_argument("--tickers", nargs="+", metavar="T", help="TARGET REFERENCE (pairs only for now)")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p.add_argument("--interval", default="1d")
    p.add_argument("--synthetic", action="store_true", help="use generated data (offline demo)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cache-dir", default=".cache")
    p.add_argument("--plot", metavar="PNG", help="save equity/z-score/beta plot (backtest)")
    p.add_argument("--max-bars", type=int, default=None, help="stop live loop after N bars")
    g = p.add_argument_group("model / strategy")
    g.add_argument("--q", type=float, default=KalmanConfig.transition_covariance_q)
    g.add_argument("--r", type=float, default=None, help="measurement noise; default = estimate from warmup OLS")
    g.add_argument("--warmup", type=int, default=KalmanConfig.warmup)
    g.add_argument("--entry-z", type=float, default=SignalConfig.entry_z)
    g.add_argument("--exit-z", type=float, default=SignalConfig.exit_z)
    g.add_argument("--stop-z", type=float, default=SignalConfig.stop_z)
    g.add_argument("--stop-bars", type=int, default=SignalConfig.stop_bars)
    g.add_argument("--max-loss", type=float, default=SignalConfig.max_loss_pct,
                   help="hard stop per trade as fraction of entry equity (default 0.05)")
    g.add_argument("--capital", type=float, default=BacktestConfig.initial_capital)
    g.add_argument("--allocation", type=float, default=BacktestConfig.allocation)
    g.add_argument("--execution", choices=["open", "close"], default="close")
    g.add_argument("--commission-per-share", type=float, default=CostModel.commission_per_share)
    g.add_argument("--commission-pct", type=float, default=0.0, help="e.g. 0.0005 for 0.05%%")
    g.add_argument("--slippage", type=float, default=CostModel.slippage_per_share)
    return p


def build_config(a: argparse.Namespace) -> StrategyConfig:
    return StrategyConfig(
        kalman=KalmanConfig(a.q, a.r, a.warmup),
        signal=SignalConfig(entry_z=a.entry_z, exit_z=a.exit_z, stop_z=a.stop_z, stop_bars=a.stop_bars,
                            max_loss_pct=a.max_loss),
        costs=CostModel(a.commission_per_share, a.commission_pct, a.slippage),
        backtest=BacktestConfig(a.capital, a.allocation, a.execution, a.interval),
    )


def load_data(a: argparse.Namespace) -> tuple[pd.DataFrame, str, str]:
    if a.synthetic:
        df = make_synthetic_pair(seed=a.seed)
        return df, "AAA_close", "BBB_close"
    if not a.tickers or len(a.tickers) < 2:
        raise SystemExit("--tickers needs a target and a reference, e.g. --tickers XOM CVX")
    if len(a.tickers) > 2:
        raise SystemExit("Baskets aren't supported yet: pass exactly one target and one reference.")
    tgt, ref = a.tickers
    logging.info("Initializing API downloads and executing historical alignment...")
    df = DataDownloader(cache_dir=a.cache_dir).fetch_and_align(tgt, [ref], a.start, a.end, a.interval)
    return df, f"{tgt}_close", f"{ref}_close"


def print_cointegration(df: pd.DataFrame, y_col: str, x_col: str) -> None:
    res = test_cointegration(df[y_col], df[x_col])
    print("=" * 52)
    print(f" COINTEGRATION (Engle-Granger)   bars={len(df)}")
    print("=" * 52)
    print(f" ADF statistic : {res.adf_statistic:.3f}")
    print(f" p-value       : {res.p_value:.4f}")
    print(" Critical vals : " + "  ".join(f"{k}={v:.2f}" for k, v in res.critical_values.items()))
    if res.warning:
        print(f" {RED}[WARN] {res.warning}{RESET}")
    else:
        print(" Verdict       : cointegrated at 5%")


def run_backtest(a: argparse.Namespace, cfg: StrategyConfig) -> None:
    df, y, x = load_data(a)
    logging.info("Data OK: %d aligned bars", len(df))
    print_cointegration(df, y, x)
    res = Backtester(cfg).run(df, y, x)
    print(format_report(res, cfg, title=f"{y.split('_')[0]} vs {x.split('_')[0]}"))
    if a.plot:
        plot_equity(res, a.plot)
        print(f" plot saved -> {a.plot}")


def run_live(a: argparse.Namespace, cfg: StrategyConfig) -> None:
    if a.synthetic:
        df, y, x = load_data(a)
        split = int(len(df) * 0.7)
        history, future = df.iloc[:split], df.iloc[split:]
        feed = ReplayFeed(future, y, x, disconnect_at={40}, outage_bars=3)
        trader = PaperTrader.warm_start(history, y, x, feed, cfg, backoff=0.1)
    else:
        df, y, x = load_data(a)     # history for warm start
        tgt, ref = y.split("_")[0], x.split("_")[0]
        trader = PaperTrader.warm_start(df, y, x, YFinanceLiveFeed(tgt, ref, a.interval), cfg)
    logging.info("PAPER TRADING started. No real orders will be sent.")
    try:
        trader.run(max_bars=a.max_bars)
    except KeyboardInterrupt:
        logging.info("Interrupted.")
    pf = trader.strategy.pf
    logging.info("Stopped. status=%s bars=%d position=%s costs=$%.2f",
                 trader.status, trader.bars_processed, pf.position.name, pf.total_costs)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    a = build_parser().parse_args(argv)
    cfg = build_config(a)
    try:
        (run_backtest if a.backtest else run_live)(a, cfg)
    except (DataError, MatrixSingularityError, FeedDisconnected) as exc:
        print(f"{RED}[ERROR] {type(exc).__name__}: {exc}{RESET}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
