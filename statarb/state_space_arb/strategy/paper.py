"""Live paper trading: feeds, disconnect handling, and a persistent console loop."""
from __future__ import annotations

import logging
import time
from typing import Callable, Iterator, Protocol

import pandas as pd

from ..config.settings import StrategyConfig
from .runner import PairBar, PairsStrategy

log = logging.getLogger("paper")


class FeedDisconnected(Exception):
    """Raised by a feed when the connection drops or the API rate-limits us."""


class BarFeed(Protocol):
    def stream(self, since: pd.Timestamp | None) -> Iterator[PairBar]:
        """Yield bars strictly after `since`. Bars missed during an outage must have is_backfill=True."""


class ReplayFeed:
    """Mock feed that replays a DataFrame, optionally dropping the connection at given bar offsets."""

    def __init__(self, frame: pd.DataFrame, y_col: str, x_col: str,
                 disconnect_at: set[int] | None = None, outage_bars: int = 3, delay: float = 0.0) -> None:
        self.frame, self.y_col, self.x_col = frame, y_col, x_col
        self.disconnect_at = set(disconnect_at or ())
        self.outage_bars, self.delay = outage_bars, delay
        self._tripped: set[int] = set()
        self._backfill_until = -1

    def _bar(self, i: int, backfill: bool = False) -> PairBar:
        f = self.frame
        yo, xo = self.y_col.replace("_close", "_open"), self.x_col.replace("_close", "_open")
        row = f.iloc[i]
        return PairBar(f.index[i], float(row.get(yo, row[self.y_col])), float(row[self.y_col]),
                       float(row.get(xo, row[self.x_col])), float(row[self.x_col]), backfill)

    def stream(self, since: pd.Timestamp | None) -> Iterator[PairBar]:
        i = 0 if since is None else int(self.frame.index.searchsorted(since, side="right"))
        while i < len(self.frame):
            if i in self.disconnect_at and i not in self._tripped:
                self._tripped.add(i)
                self._backfill_until = i + self.outage_bars   # bars that "happened" while we were offline
                raise FeedDisconnected(f"simulated drop before bar {i}")
            if self.delay:
                time.sleep(self.delay)
            yield self._bar(i, backfill=i < self._backfill_until)
            i += 1


class YFinanceLiveFeed:
    """Polls yfinance for the newest completed bar. Fine for paper trading; swap for Alpaca/IB sockets later."""

    def __init__(self, target: str, ref: str, interval: str = "1m", poll_seconds: float = 30.0) -> None:
        self.target, self.ref, self.interval, self.poll = target, ref, interval, poll_seconds

    def _download(self) -> pd.DataFrame:
        import yfinance as yf
        period = "5d" if self.interval.endswith("m") else "1mo"
        frames = {}
        for t in (self.target, self.ref):
            try:
                df = yf.download(t, period=period, interval=self.interval, auto_adjust=True, progress=False)
            except Exception as exc:  # network error, rate limit, etc.
                raise FeedDisconnected(str(exc)) from exc
            if df is None or df.empty:
                raise FeedDisconnected(f"empty response for {t}")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            frames[t] = df[["open", "close"]]
        out = pd.concat({t: f for t, f in frames.items()}, axis=1, join="inner").dropna()
        out.index = out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")
        return out.iloc[:-1]   # last row is still forming

    def stream(self, since: pd.Timestamp | None) -> Iterator[PairBar]:
        last = since
        first = True
        while True:
            df = self._download()
            new = df if last is None else df[df.index > last]
            for ts, row in new.iterrows():
                yield PairBar(ts, float(row[(self.target, "open")]), float(row[(self.target, "close")]),
                              float(row[(self.ref, "open")]), float(row[(self.ref, "close")]),
                              is_backfill=first or (last is not None and len(new) > 1))
                last = ts
            first = False
            time.sleep(self.poll)


class PaperTrader:
    """Runs PairsStrategy on live bars. On disconnect: cancel pending orders, hold, reconnect with
    backoff, and replay missed bars in observe-only mode so the Kalman state stays in sync."""

    def __init__(self, feed: BarFeed, strategy: PairsStrategy, max_retries: int = 5,
                 backoff: float = 2.0, sleep: Callable[[float], None] = time.sleep) -> None:
        self.feed, self.strategy = feed, strategy
        self.max_retries, self.backoff, self.sleep = max_retries, backoff, sleep
        self.last_ts: pd.Timestamp | None = None
        self.status = "RUNNING"
        self.bars_processed = 0

    @classmethod
    def warm_start(cls, history: pd.DataFrame, y_col: str, x_col: str, feed: BarFeed,
                   cfg: StrategyConfig, **kw) -> "PaperTrader":
        strat = PairsStrategy.bootstrap(history[y_col].to_numpy(float), history[x_col].to_numpy(float), cfg)
        tr = cls(feed, strat, **kw)
        yo, xo = y_col.replace("_close", "_open"), x_col.replace("_close", "_open")
        for ts, row in history.iterrows():            # replay history to sync the filter; no trading
            strat.on_bar(PairBar(ts, row.get(yo, row[y_col]), row[y_col], row.get(xo, row[x_col]), row[x_col]), trade=False)
            tr.last_ts = ts
        return tr

    def run(self, max_bars: int | None = None) -> None:
        retries = 0
        while True:
            try:
                for bar in self.feed.stream(self.last_ts):
                    retries = 0
                    self.status = "RUNNING"
                    rec = self.strategy.on_bar(bar)
                    self.last_ts = bar.timestamp
                    self.bars_processed += 1
                    upnl = rec.equity - self.strategy.cfg.backtest.initial_capital
                    log.info("%s | z=%+6.2f beta=%6.3f pos=%+d equity=$%s uPnL=$%s %s%s",
                             bar.timestamp.strftime("%Y-%m-%d %H:%M"), rec.z, rec.beta, rec.position,
                             f"{rec.equity:,.2f}", f"{upnl:+,.2f}", rec.event,
                             " [backfill]" if bar.is_backfill else "")
                    if max_bars and self.bars_processed >= max_bars:
                        return
                return  # stream ended cleanly
            except FeedDisconnected as exc:
                retries += 1
                self.status = "HOLD (disconnected)"
                self.strategy.pending = None       # never fire an order decided on stale data
                log.warning("Feed disconnected: %s. HOLD. retry %d/%d", exc, retries, self.max_retries)
                if retries > self.max_retries:
                    log.error("Giving up after %d retries. Positions left as-is: %s", self.max_retries,
                              self.strategy.pf.position.name)
                    raise
                self.sleep(self.backoff * 2 ** (retries - 1))
