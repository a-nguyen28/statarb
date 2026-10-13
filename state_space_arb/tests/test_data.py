import numpy as np
import pandas as pd
import pytest

from state_space_arb.data import (DataDownloader, InsuffientDataError, InsufficientDataError,
                                  InvalidTickerError, align_frames, test_cointegration)


def _frame(start, n, seed=0):
    idx = pd.date_range(start, periods=n, freq="B")
    p = 100 + np.cumsum(np.random.default_rng(seed).normal(0, 1, n))
    return pd.DataFrame({"Open": p, "Close": p}, index=idx)


def test_no_common_timestamps_raises():
    with pytest.raises(InsuffientDataError):   # spec's spelling works too
        align_frames({"A": _frame("2020-01-01", 40), "B": _frame("2022-01-01", 40)})


def test_too_few_rows_raises():
    with pytest.raises(InsufficientDataError):
        align_frames({"A": _frame("2020-01-01", 20), "B": _frame("2020-01-01", 20)})


def test_inner_join_drops_mismatched_days_and_names_columns():
    a, b = _frame("2020-01-01", 60), _frame("2020-01-01", 60, seed=1).drop(pd.Timestamp("2020-01-15"))
    out = align_frames({"XOM": a, "CVX": b})
    assert len(out) == 59 and not out.isna().any().any()
    assert {"XOM_close", "CVX_close", "XOM_open", "CVX_open"} <= set(out.columns)
    assert str(out.index.tz) == "UTC"


def test_downloader_uses_cache_and_flags_bad_ticker(tmp_path):
    calls = []

    def fetcher(t, s, e, i):
        calls.append(t)
        return pd.DataFrame() if t == "BAD" else _frame("2020-01-01", 60, seed=len(calls))

    dl = DataDownloader(cache_dir=tmp_path, fetcher=fetcher)
    dl.fetch_and_align("A", ["B"], "2020-01-01", "2021-01-01")
    dl.fetch_and_align("A", ["B"], "2020-01-01", "2021-01-01")
    assert calls == ["A", "B"]                       # second call served from parquet
    with pytest.raises(InvalidTickerError):
        dl.fetch_and_align("A", ["BAD"], "2020-01-01", "2021-01-01")


def test_cointegration_flags_independent_random_walks():
    rng = np.random.default_rng(5)
    x, y = np.cumsum(rng.normal(size=1000)), np.cumsum(rng.normal(size=1000))
    res = test_cointegration(y, x)
    assert not res.is_cointegrated and res.warning
    z = x * 1.5 + rng.normal(size=1000)
    assert test_cointegration(z, x).is_cointegrated
