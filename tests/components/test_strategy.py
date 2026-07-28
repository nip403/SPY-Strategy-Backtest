from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_engine.components.base import BacktestContext
from backtest_engine.components.strategy import (
    resolve_positions, BaseStrategy, RollingImmediateStopStrategy, RollingIntervalStopStrategy, QuarterHourSampleStrategy,
)

# ---- resolve_positions: the regression test for the entry/exit-gating bug ------------------
# Columns per bar: [long_entry, short_entry, long_exit, short_exit, end_of_day]

RESOLVE_POSITIONS_CASES = [
    pytest.param([(0, 0, 0, 0, 0)], [0], id="flat_no_signal_stays_flat"),
    pytest.param([(1, 0, 0, 0, 0)], [1], id="flat_long_entry_fires"),
    pytest.param([(0, 1, 0, 0, 0)], [-1], id="flat_short_entry_fires"),
    pytest.param([(1, 0, 1, 0, 0)], [1], id="flat_entry_ignores_incidental_exit_signal"),
    pytest.param([(1, 0, 0, 0, 0), (1, 0, 1, 0, 0)], [1, 0], id="held_long_stops_out_despite_entry_still_true"),
    pytest.param([(0, 1, 0, 0, 0), (0, 1, 0, 1, 0)], [-1, 0], id="held_short_stops_out_despite_entry_still_true"),
    pytest.param([(1, 0, 0, 0, 0), (1, 0, 0, 0, 0)], [1, 1], id="held_long_same_direction_reentry_noop"),
    pytest.param([(0, 1, 0, 0, 0), (0, 1, 0, 0, 0)], [-1, -1], id="held_short_same_direction_reentry_noop"),
    pytest.param([(1, 0, 0, 0, 0), (0, 1, 1, 0, 0)], [1, -1], id="held_long_opposite_entry_flip_beats_stopout"),
    pytest.param([(0, 1, 0, 0, 0), (1, 0, 0, 1, 0)], [-1, 1], id="held_short_opposite_entry_flip_beats_stopout"),
    pytest.param([(1, 0, 0, 0, 0), (1, 1, 0, 0, 1)], [1, 0], id="end_of_day_forces_flat_over_everything"),
    pytest.param([(1, 0, 0, 0, 0), (0, 0, 0, 0, 1), (0, 1, 0, 0, 0)], [1, 0, -1], id="post_eod_flat_allows_fresh_entry"),
]

@pytest.mark.parametrize("bars, expected", RESOLVE_POSITIONS_CASES)
def test_resolve_positions(bars, expected):
    result = resolve_positions(np.asarray(bars, dtype=float))
    np.testing.assert_array_equal(result, np.asarray(expected, dtype=float))

# ---- strategy classes' signal generation -----------------------------------

def _make_signal_df(times: list[str], *, close, upper_bound, lower_bound, long_stop, short_stop, std=None) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(f"2024-01-02 {t}") for t in times]).tz_localize("America/New_York")

    return pd.DataFrame({
        "close": close,
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "long_stop": long_stop,
        "short_stop": short_stop,
        "std": std or [0.01] * len(times),
    }, index=idx)

def _ctx(*, long_perm=True, short_perm=True, max_leverage=4.0) -> BacktestContext:
    return BacktestContext(aum=100_000, target_vol=0.02, long_perm=long_perm, short_perm=short_perm, max_leverage=max_leverage)

def test_base_strategy_only_fires_on_half_hour_intervals():
    df = _make_signal_df(
        ["09:59", "10:00", "10:01"],
        close=[110, 110, 110], upper_bound=[100] * 3, lower_bound=[90] * 3, long_stop=[105] * 3, short_stop=[95] * 3,
    )
    result = BaseStrategy().set(df, _ctx())

    assert result["position"].iloc[0] == 0  # 09:59 not an interval -> no entry yet
    assert result["position"].iloc[1] > 0  # 10:00 is an interval -> enters long
    assert result["position"].iloc[2] > 0  # 10:01 not an interval -> holds long from 10:00

def test_base_strategy_forces_flat_at_1559():
    df = _make_signal_df(
        ["15:30", "15:59"],
        close=[110, 110], upper_bound=[100] * 2, lower_bound=[90] * 2, long_stop=[105] * 2, short_stop=[95] * 2,
    )
    result = BaseStrategy().set(df, _ctx())

    assert result["position"].iloc[0] > 0  # entered long at 15:30
    assert result["position"].iloc[1] == 0  # forced flat at end of day

def test_base_strategy_respects_long_permissions_false():
    df = _make_signal_df(["10:00"], close=[110], upper_bound=[100], lower_bound=[90], long_stop=[105], short_stop=[95])
    result = BaseStrategy().set(df, _ctx(long_perm=False))

    assert result["position"].iloc[0] == 0

def test_base_strategy_respects_short_permissions_false():
    df = _make_signal_df(["10:00"], close=[80], upper_bound=[100], lower_bound=[90], long_stop=[105], short_stop=[95])
    result = BaseStrategy().set(df, _ctx(short_perm=False))

    assert result["position"].iloc[0] == 0

def test_base_strategy_position_scaled_and_clipped_to_plus_minus_4():
    df = _make_signal_df(["10:00"], close=[110], upper_bound=[100], lower_bound=[90], long_stop=[105], short_stop=[95], std=[0.001])
    result = BaseStrategy().set(df, _ctx())

    assert result["position"].iloc[0] == pytest.approx(4.0)  # target_vol/std = 20, clipped to 4

def test_rolling_immediate_stop_requires_full_confirmation_window():
    conf = 3
    times = [f"10:{i:02d}" for i in range(5)]
    df = _make_signal_df(
        times,
        close=[110] * 5, upper_bound=[100] * 5, lower_bound=[0] * 5, long_stop=[50] * 5, short_stop=[200] * 5,
    )
    result = RollingImmediateStopStrategy(entry_window=conf).set(df, _ctx())

    assert (result["position"].iloc[:2] == 0).all()  # not yet confirmed
    assert (result["position"].iloc[2:] > 0).all()  # confirmed from the 3rd consecutive bar onward

def test_rolling_immediate_stop_exit_not_gated_to_interval():
    df = _make_signal_df(
        ["10:01", "10:02"],  # neither is a :00/:30 interval
        close=[110, 40], upper_bound=[100, 100], lower_bound=[0, 0], long_stop=[50, 50], short_stop=[200, 200],
    )
    result = RollingImmediateStopStrategy(entry_window=1).set(df, _ctx())

    assert result["position"].iloc[0] > 0  # entered off-interval (window=1, immediate confirmation)
    assert result["position"].iloc[1] == 0  # exit fires off-interval too, immediately on the stop cross

def test_rolling_interval_stop_exit_gated_to_interval():
    df = _make_signal_df(
        ["10:00", "10:01"],  # 10:00 is an interval, 10:01 is not
        close=[110, 40], upper_bound=[100, 100], lower_bound=[0, 0], long_stop=[50, 50], short_stop=[200, 200],
    )
    result = RollingIntervalStopStrategy(entry_window=1).set(df, _ctx())

    assert result["position"].iloc[0] > 0  # entered at 10:00
    assert result["position"].iloc[1] > 0  # price crossed long_stop at 10:01, but that's not an interval -> stays held

def test_quarter_hour_sample_entries_fire_at_15min_marks():
    df = _make_signal_df(
        ["10:05", "10:15"],  # 10:05 is not an entry interval; 10:15 is (15-min entries)
        close=[110, 110], upper_bound=[100] * 2, lower_bound=[0] * 2, long_stop=[50] * 2, short_stop=[200] * 2,
    )
    result = QuarterHourSampleStrategy().set(df, _ctx())

    assert result["position"].iloc[0] == 0
    assert result["position"].iloc[1] > 0
