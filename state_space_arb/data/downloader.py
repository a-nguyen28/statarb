"""Ingestion: fetch OHLCV, align on timestamps, cache as parquet."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import pandas as pd

from .errors import InsufficientDataError, InvalidTickerError

MIN_ROWS = 30
Fetcher = Callable[[str, str, str, str], pd.DataFrame]


def _to_utc(idx: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")


def align_frames(
    frames: dict[str, pd.DataFrame], min_rows: int = MIN_ROWS, fields: tuple[str, ...] = ("open", "close")
) -> pd.DataFrame:
    """Strict inner join on timestamps. Columns become `<TICKER>_<field>`.

    Inner join (not forward-fill) is the default because ffill invents prices on days
    one asset didn't trade, which fakes mean reversion.
    """
    parts = []
    for ticker, df in frames.items():
        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]
        sub = df[list(fields)].copy()
        sub.index = _to_utc(sub.index)
        sub = sub[~sub.index.duplicated(keep="last")].sort_index()
        sub.columns = [f"{ticker}_{c}" for c in fields]
        parts.append(sub)

    out = pd.concat(parts, axis=1, join="inner").dropna()
    if out.empty:
        raise InsufficientDataError("Tickers share no common timestamps.")
    out = out[(out > 0).all(axis=1)]
    if len(out) < min_rows:
        raise InsufficientDataError(
            f"Only {len(out)} overlapping bars; need at least {min_rows} for the model to stabilise."
        )
    return out


def _yfinance_fetch(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    import yfinance as yf

    # auto_adjust=True -> split/dividend-adjusted prices, so no fake jumps
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


class DataDownloader:
    """Provider-agnostic: pass any `fetcher(ticker, start, end, interval) -> OHLCV DataFrame`."""

    def __init__(self, cache_dir: str | Path | None = None, fetcher: Fetcher | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.fetcher = fetcher or _yfinance_fetch
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, ticker: str, start: str, end: str, interval: str) -> Path | None:
        if not self.cache_dir:
            return None
        key = hashlib.md5(f"{ticker}|{start}|{end}|{interval}".encode()).hexdigest()[:10]
        return self.cache_dir / f"{ticker}_{interval}_{key}.parquet"

    def _fetch_one(self, ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
        path = self._cache_path(ticker, start, end, interval)
        if path and path.exists():
            return pd.read_parquet(path)
        df = self.fetcher(ticker, start, end, interval)
        if df is None or df.empty:
            raise InvalidTickerError(f"No data returned for ticker '{ticker}'.")
        if path:
            df.to_parquet(path)
        return df

    def fetch_and_align(
        self, target_ticker: str, reference_tickers: list[str], start_date: str, end_date: str, interval: str = "1d"
    ) -> pd.DataFrame:
        frames = {t: self._fetch_one(t, start_date, end_date, interval) for t in [target_ticker, *reference_tickers]}
        return align_frames(frames)
