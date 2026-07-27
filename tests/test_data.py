from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backtest_engine.data import _preprocess, request

# ---- _preprocess (pure, no I/O) --------------------------------------------

def _make_raw_alpaca_df() -> pd.DataFrame:
    # Alpaca returns UTC-aware timestamps in a MultiIndex (symbol, timestamp); tz_convert()
    # requires an already-aware index, hence utc=True below.
    timestamps = pd.to_datetime([
        "2024-01-02 14:29:00",  # 09:29 ET -> before session open, dropped
        "2024-01-02 14:30:00",  # 09:30 ET -> valid
        "2024-01-02 14:31:00",  # 09:31 ET -> valid
        "2024-01-02 21:00:00",  # 16:00 ET -> after 15:59 cutoff, dropped
        "2024-01-06 14:30:00",  # Saturday 09:30 ET -> weekend, dropped
        "2024-01-08 14:30:00",  # Monday 09:30 ET, but close is NaN -> dropped by dropna()
    ], utc=True)

    index = pd.MultiIndex.from_arrays([["SPY"] * len(timestamps), timestamps], names=["symbol", "timestamp"])

    return pd.DataFrame({
        "open": [400.0, 400.1, 400.2, 401.0, 402.0, 403.0],
        "high": [400.5, 400.6, 400.7, 401.5, 402.5, 403.5],
        "low": [399.5, 399.6, 399.7, 400.5, 401.5, 402.5],
        "close": [400.2, 400.3, 400.4, 401.2, 402.2, np.nan],
        "volume": [1000, 1100, 1200, 1300, 1400, 1500],
        "trade_count": [10, 11, 12, 13, 14, 15],  # extra Alpaca column, must be dropped
        "vwap": [400.1, 400.2, 400.3, 401.1, 402.1, 402.9],  # extra Alpaca column, must be dropped
    }, index=index)

def test_preprocess_output_columns_are_ohlcv_plus_time():
    out = _preprocess(_make_raw_alpaca_df())
    assert list(out.columns) == ["open", "high", "low", "close", "volume", "time"]

def test_preprocess_localizes_to_america_new_york():
    out = _preprocess(_make_raw_alpaca_df())
    assert str(out.index.tz) == "America/New_York"

def test_preprocess_drops_outside_session_and_weekend_and_nan_rows():
    out = _preprocess(_make_raw_alpaca_df())

    # only the two 2024-01-02 09:30/09:31 ET bars survive: 09:29 (pre-open), 16:00 (post-close),
    # Saturday, and the NaN-close Monday bar are all dropped
    assert len(out) == 2
    assert list(out.index.time) == [pd.Timestamp("09:30").time(), pd.Timestamp("09:31").time()]
    assert (out.index.date == pd.Timestamp("2024-01-02").date()).all()

def test_preprocess_sorts_by_timestamp():
    raw = _make_raw_alpaca_df()
    shuffled = raw.iloc[[2, 0, 1, 3, 4, 5]]  # deliberately out of order

    out = _preprocess(shuffled)

    assert out.index.is_monotonic_increasing

# ---- request() cache-hit path (no network) ---------------------------------

def test_request_cache_hit_returns_cached_df_without_touching_alpaca(tmp_cwd, fake_alpaca_config):
    start, end = datetime(2024, 1, 1), datetime(2024, 1, 2)
    cache_file = tmp_cwd / f"cache_SPY_{start:%Y%m%d}_{end:%Y%m%d}.parquet"

    cached = pd.DataFrame({
        "open": [400.0], "high": [400.5], "low": [399.5], "close": [400.2], "volume": [1000.0],
    }, index=pd.DatetimeIndex(["2024-01-02 09:30:00"], tz="America/New_York"))
    cached["time"] = cached.index.time
    cached.to_parquet(cache_file)

    with patch("backtest_engine.data.StockHistoricalDataClient") as MockClient:
        result = request(ticker="SPY", config=fake_alpaca_config, start=start, end=end, use_cache=True)
        MockClient.assert_not_called()

    pd.testing.assert_frame_equal(result, cached)

def test_request_cache_filename_encodes_ticker_and_date_range(tmp_cwd, fake_alpaca_config):
    start, end = datetime(2020, 3, 15), datetime(2021, 6, 30)
    expected_name = "cache_QQQ_20200315_20210630.parquet"

    cached = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]})
    cached.to_parquet(tmp_cwd / expected_name)

    with patch("backtest_engine.data.StockHistoricalDataClient") as MockClient:
        request(ticker="QQQ", config=fake_alpaca_config, start=start, end=end, use_cache=True)
        MockClient.assert_not_called()

# ---- request() live-fetch path (mocked, no real network) -------------------

def test_request_live_fetch_calls_alpaca_with_correct_params_and_writes_cache(tmp_cwd, fake_alpaca_config):
    start, end = datetime(2024, 1, 1), datetime(2024, 1, 3)
    mock_bars = MagicMock()
    mock_bars.df = _make_raw_alpaca_df()

    with patch("backtest_engine.data.StockHistoricalDataClient") as MockClient:
        instance = MockClient.return_value
        instance.get_stock_bars.return_value = mock_bars

        result = request(ticker="SPY", config=fake_alpaca_config, start=start, end=end, use_cache=True)

    MockClient.assert_called_once_with(api_key="dummy-key", secret_key="dummy-secret")
    instance.get_stock_bars.assert_called_once()

    req_arg = instance.get_stock_bars.call_args[0][0]
    assert req_arg.symbol_or_symbols == ["SPY"]
    assert req_arg.start == start
    assert req_arg.end == end

    assert len(result) == 2  # matches _preprocess's filtering of the same raw fixture
    assert (tmp_cwd / f"cache_SPY_{start:%Y%m%d}_{end:%Y%m%d}.parquet").exists()

def test_request_use_cache_false_always_fetches_even_if_cache_exists(tmp_cwd, fake_alpaca_config):
    start, end = datetime(2024, 1, 1), datetime(2024, 1, 3)
    cache_file = tmp_cwd / f"cache_SPY_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
    pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]}).to_parquet(cache_file)

    mock_bars = MagicMock()
    mock_bars.df = _make_raw_alpaca_df()

    with patch("backtest_engine.data.StockHistoricalDataClient") as MockClient:
        instance = MockClient.return_value
        instance.get_stock_bars.return_value = mock_bars

        request(ticker="SPY", config=fake_alpaca_config, start=start, end=end, use_cache=False)

        instance.get_stock_bars.assert_called_once()
