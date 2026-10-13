# State-Space StatArb Engine

Kalman-filter pairs trading: track a drifting hedge ratio, trade the spread, backtest with real
friction, then paper-trade the exact same code path.

```bash
pip install -e ".[dev]"
python main.py --backtest --tickers XOM CVX --start 2021-01-01      # real data (needs internet)
python main.py --backtest --synthetic --plot equity.png            # offline demo
python main.py --live --synthetic                                  # mock feed + simulated network drop
python main.py --live --tickers XOM CVX --interval 1m              # yfinance polling paper trade
pytest
```

## Layout
```
config/    settings.py        every tunable in one place (dataclasses)
data/      downloader.py      yfinance + parquet cache + strict inner-join alignment
           cointegration.py   Engle-Granger test, warns if p >= 0.05
           synthetic.py       fake cointegrated pair for tests/demos
engine/    kalman.py          KalmanTracker (streaming) + KalmanFilterEngine.compute_states (batch)
strategy/  signals.py         z-score -> position state machine (entry/exit/stop/cooldown)
           portfolio.py       two-leg accounting, commission + slippage
           runner.py          PairsStrategy: ONE per-bar brain used by backtest AND live
           backtester.py      metrics, report, plot
           paper.py           feeds (replay + yfinance), PaperTrader w/ disconnect handling
tests/
main.py
```

## How look-ahead is prevented
1. Kalman `spread` is the innovation (y minus the prediction made *before* seeing y). `sigma` is the
   EW std of *previous* innovations.
2. A signal at bar t is queued and filled at bar t+1 (open or close, `--execution`).
3. Tests perturb future prices and assert nothing in the past changes (filter and full backtest).

## Risk guards (two layers, on purpose)
- **z-stop**: spread beyond `--stop-z` (5σ) for `--stop-bars` consecutive bars -> flatten + cooldown.
- **Dollar stop** (`--max-loss`, default 5% of entry equity): needed because σ is a running std. When a
  spread runs away, σ inflates and z stays modest, so the z-stop alone goes blind. In a synthetic
  break test the z-stop alone allowed -82% drawdown; adding the dollar stop cut it to -15%
  (the remainder is gap risk from filling at t+1).
- The Kalman filter can also *absorb* a break by rewriting beta/alpha. That's a feature for slow drift,
  and a trap for a real regime change. Watch `beta` in the plot; a hedge ratio that suddenly lurches is a smell.

## Tuning cheat sheet
- `--q` (default 1e-5): how fast beta may drift. Too high = beta chases noise and eats your signal.
  Too low = beta lags a real regime shift. Try 1e-6 .. 1e-4.
- `--r`: default is estimated from the warmup OLS. Override if spreads look too jumpy/sluggish.
- Warmup bars are never traded.

## Known limits
- Pairs only (one target, one reference). Baskets need a k-regressor state vector + multi-leg portfolio.
- yfinance 1-minute history only goes back ~7 days; live polling is for paper trading, not production.
- Fills assume liquid names (1-tick slippage). Illiquid assets will understate costs.
- Backtest numbers on synthetic data prove the plumbing, not the strategy. Real pairs are messier.
