from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_engine.components.base import BacktestContext
from backtest_engine.components.execution import NaiveExecution, CappedVolumeExecution, CappedVolumeRolloverExecution

def _ctx(aum: float = 1e5) -> BacktestContext:
    return BacktestContext(aum=aum, target_vol=0.02, long_perm=True, short_perm=True)

# ---- NaiveExecution --------------------------------------------------------

def test_naive_execution_passes_position_through_unchanged():
    df = pd.DataFrame({"position": [0.0, 1.0, -2.0, 0.5]})
    result = NaiveExecution().fill(df.copy(), _ctx(), {})

    np.testing.assert_array_equal(result["position"], [0.0, 1.0, -2.0, 0.5])

def test_naive_execution_fill_matrix_tiles_across_aum():
    df = pd.DataFrame({"position": [1.0, -1.0]})
    cache = {}
    NaiveExecution().fill(df.copy(), _ctx(), cache)
    matrix = NaiveExecution().fill_matrix(np.array([1e4, 1e5, 1e6]), cache)

    assert matrix.shape == (2, 3)
    for col in range(3):
        np.testing.assert_array_equal(matrix[:, col], [1.0, -1.0])

# ---- CappedVolumeExecution (immediate-or-cancel) ---------------------------

def test_capped_ioc_caps_participation_to_ceiling():
    df = pd.DataFrame({"position": [1.0], "close": [100.0], "volume": [1000.0]})
    result = CappedVolumeExecution(participation_ceiling=0.1).fill(df.copy(), _ctx(), {})

    # raw_capacity = 1000*100*0.1 = 10_000; capacity(leverage) = 10_000/1e5 = 0.1
    assert result["position"].iloc[0] == pytest.approx(0.1)

def test_capped_ioc_does_not_roll_over_unfilled_quantity():
    df = pd.DataFrame({"position": [1.0, 1.0, 1.0], "close": [100.0] * 3, "volume": [1000.0] * 3})
    result = CappedVolumeExecution(participation_ceiling=0.1).fill(df.copy(), _ctx(), {})

    # target never changes after bar 0, so only bar 0 is a fill "event" - later bars forward-fill
    # that partial result rather than accumulating further toward the (unchanged) target
    np.testing.assert_allclose(result["position"], [0.1, 0.1, 0.1])

def test_capped_ioc_fill_matrix_carries_position_across_events():
    # regression: fill_matrix's loop used to never reassign p_current between signal events, so
    # every event resolved as if starting fresh from position 0, discarding whatever was actually
    # filled at prior events. Event 2 here has a binding capacity (0.1) that can only carry the
    # already-filled 0.6 up to 0.7 - the bug instead recomputed from a phantom position of 0.
    target = pd.Series([0.0, 1.0, 0.8])
    raw_capacity = pd.Series([0.0, 0.6, 0.1])
    signal = pd.Series([True, True, True])
    cache = {"ioc": pd.DataFrame({"target": target, "raw_capacity": raw_capacity, "signal": signal})}

    fills = CappedVolumeExecution(participation_ceiling=1.0).fill_matrix(np.array([1.0]), cache)[:, 0]

    np.testing.assert_allclose(fills, [0.0, 0.6, 0.7])

def test_capped_ioc_fill_matches_fill_matrix_at_same_aum():
    df = pd.DataFrame({"position": [0.0, 1.0, 1.0, -0.5], "close": [100.0] * 4, "volume": [1000.0] * 4})
    cache = {}
    exe = CappedVolumeExecution(participation_ceiling=0.1)
    result = exe.fill(df.copy(), _ctx(), cache)
    matrix = exe.fill_matrix(np.array([1e5]), cache)

    np.testing.assert_allclose(result["position"].to_numpy(), matrix[:, 0])

# ---- CappedVolumeRolloverExecution -----------------------------------------

def test_capped_rollover_accumulates_unfilled_quantity_across_bars():
    df = pd.DataFrame({"position": [1.0] * 4, "close": [100.0] * 4, "volume": [1000.0] * 4})
    result = CappedVolumeRolloverExecution(participation_ceiling=0.1).fill(df.copy(), _ctx(), {})

    np.testing.assert_allclose(result["position"], [0.1, 0.2, 0.3, 0.4])

def test_capped_rollover_eventually_reaches_target_once_capacity_allows():
    df = pd.DataFrame({"position": [1.0] * 12, "close": [100.0] * 12, "volume": [1000.0] * 12})
    result = CappedVolumeRolloverExecution(participation_ceiling=0.1).fill(df.copy(), _ctx(), {})

    assert result["position"].iloc[9] == pytest.approx(1.0)  # 10 bars * 0.1/bar = 1.0
    assert result["position"].iloc[11] == pytest.approx(1.0)  # holds at target, no overshoot

def test_capped_rollover_fill_matches_fill_matrix_at_same_aum():
    df = pd.DataFrame({"position": [0.0, 1.0, 1.0, -0.5, -0.5], "close": [100.0] * 5, "volume": [1000.0] * 5})
    cache = {}
    exe = CappedVolumeRolloverExecution(participation_ceiling=0.1)
    result = exe.fill(df.copy(), _ctx(), cache)
    matrix = exe.fill_matrix(np.array([1e5]), cache)

    np.testing.assert_allclose(result["position"].to_numpy(), matrix[:, 0])

# ---- shared edge cases ------------------------------------------------------

@pytest.mark.parametrize("execution_cls", [CappedVolumeExecution, CappedVolumeRolloverExecution])
def test_execution_aum_zero_yields_zero_position_no_warning(execution_cls, assert_no_warnings):
    df = pd.DataFrame({"position": [1.0, 1.0], "close": [100.0, 100.0], "volume": [1000.0, 1000.0]})
    cache = {}
    exe = execution_cls(participation_ceiling=0.1)
    exe.fill(df.copy(), _ctx(), cache)

    with assert_no_warnings():
        matrix = exe.fill_matrix(np.array([0.0]), cache)

    assert np.all(matrix == 0.0)
    assert np.all(np.isfinite(matrix))

@pytest.mark.parametrize("execution_cls", [CappedVolumeExecution, CappedVolumeRolloverExecution])
def test_execution_zero_volume_bar_yields_zero_capacity_no_warning(execution_cls, assert_no_warnings):
    df = pd.DataFrame({"position": [1.0], "close": [100.0], "volume": [0.0]})

    with assert_no_warnings():
        result = execution_cls(participation_ceiling=0.1).fill(df.copy(), _ctx(), {})

    assert result["position"].iloc[0] == 0.0
