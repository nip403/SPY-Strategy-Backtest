from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_engine.components.base import BacktestContext
from backtest_engine.components.cost_model import FlatCostModel, DynamicCostModel

def _ctx(aum: float = 1e5) -> BacktestContext:
    return BacktestContext(aum=aum, target_vol=0.02, long_perm=True, short_perm=True, max_leverage=4.0)

# ---- FlatCostModel ----------------------------------------------------------

def test_flat_cost_model_matches_hand_derived_commission_and_slippage():
    df = pd.DataFrame({
        "position": [0.0, 1.0, 1.0, -0.5],
        "close": [100.0, 101.0, 102.0, 103.0],
        "gross_ret": [0.0, 0.01, 0.0, -0.02],
    })
    model = FlatCostModel(commission=0.0035, slippage=0.001)
    result = model.expense(df.copy(), _ctx(), {})

    frictions = 0.0035 + 0.001
    turnover = [0.0, 1.0, 0.0, 1.5]  # |diff(position, prepend=first)|
    expected_net = [g - t * frictions / c for g, t, c in zip(df["gross_ret"], turnover, df["close"])]

    np.testing.assert_allclose(result["net_ret"], expected_net)

def test_flat_cost_model_zero_turnover_bar_has_zero_cost():
    df = pd.DataFrame({"position": [1.0, 1.0, 1.0], "close": [100.0] * 3, "gross_ret": [0.01, 0.02, -0.01]})
    result = FlatCostModel().expense(df.copy(), _ctx(), {})

    np.testing.assert_allclose(result["net_ret"], df["gross_ret"])

def test_flat_cost_model_returns_matrix_matches_expense_at_same_aum():
    df = pd.DataFrame({"position": [0.0, 1.0, -1.0], "close": [100.0, 101.0, 99.0], "gross_ret": [0.0, 0.01, -0.02]})
    cache = {}
    model = FlatCostModel()
    result = model.expense(df.copy(), _ctx(), cache)

    matrix = model.returns_matrix(np.array([1e5]), df[["position"]].to_numpy(), df[["gross_ret"]].to_numpy(), cache)

    np.testing.assert_allclose(result["net_ret"].to_numpy(), matrix[:, 0])

# ---- DynamicCostModel --------------------------------------------------------

def test_dynamic_cost_model_matches_hand_derived_kissel_formula():
    position_matrix = np.array([[0.0], [1.0], [1.0]])
    gross_matrix = np.array([[0.0], [0.01], [0.02]])
    cache = {
        "close": pd.Series([100.0, 101.0, 102.0]),
        "volume": pd.Series([10_000.0, 10_000.0, 10_000.0]),
        "dynamic_cost": pd.DataFrame({"adv": [500_000.0, 500_000.0, 500_000.0], "ann_vol": [0.2, 0.2, 0.2]}),
    }

    model = DynamicCostModel()  # defaults: a1=700, a2=0.5, a3=1.0, a4=0.5, b1=0.85, commission=0.0035
    matrix = model.returns_matrix(np.array([1e5]), position_matrix, gross_matrix, cache)

    # independently hand-derive bar 1 from the Kissel I-Star formula in the docstring
    dl, close, adv, vol, ann_vol, aum = 1.0, 101.0, 500_000.0, 10_000.0, 0.2, 1e5
    adv_term = dl / (close * adv)
    participation = dl / (close * vol)
    perm = 700 * (adv_term ** 0.5) * (ann_vol ** 1.0) * dl
    temp = perm * (participation ** 0.5)
    commission = 0.0035 * dl / close
    mkt = (0.85 * temp * (aum ** 1.0) + 0.15 * perm * (aum ** 0.5)) / 10_000
    expected = gross_matrix[1, 0] - mkt - commission

    assert matrix[1, 0] == pytest.approx(expected, rel=1e-9)

def test_dynamic_cost_model_zero_adv_guarded_no_warning(assert_no_warnings):
    position_matrix = np.array([[0.0], [1.0]])
    gross_matrix = np.array([[0.0], [0.01]])
    cache = {
        "close": pd.Series([100.0, 101.0]),
        "volume": pd.Series([10_000.0, 10_000.0]),
        "dynamic_cost": pd.DataFrame({"adv": [0.0, 0.0], "ann_vol": [0.2, 0.2]}),
    }

    with assert_no_warnings():
        matrix = DynamicCostModel().returns_matrix(np.array([1e5]), position_matrix, gross_matrix, cache)

    assert np.all(np.isfinite(matrix))

def test_dynamic_cost_model_zero_volume_bar_guarded_no_warning(assert_no_warnings):
    position_matrix = np.array([[0.0], [1.0]])
    gross_matrix = np.array([[0.0], [0.01]])
    cache = {
        "close": pd.Series([100.0, 101.0]),
        "volume": pd.Series([10_000.0, 0.0]),
        "dynamic_cost": pd.DataFrame({"adv": [500_000.0, 500_000.0], "ann_vol": [0.2, 0.2]}),
    }

    with assert_no_warnings():
        matrix = DynamicCostModel().returns_matrix(np.array([1e5]), position_matrix, gross_matrix, cache)

    assert np.all(np.isfinite(matrix))

def test_dynamic_cost_model_commission_independent_of_aum():
    position_matrix = np.array([[0.0], [1.0]])
    gross_matrix = np.array([[0.0], [0.01]])
    cache = {
        "close": pd.Series([100.0, 101.0]),
        "volume": pd.Series([10_000.0, 10_000.0]),
        "dynamic_cost": pd.DataFrame({"adv": [500_000.0, 500_000.0], "ann_vol": [0.2, 0.2]}),
    }

    # aum=0 zeroes the market-impact term entirely (0**0.5 = 0), leaving only commission
    matrix = DynamicCostModel(commission=0.0035).returns_matrix(np.array([0.0]), position_matrix, gross_matrix, cache)

    expected_commission = 0.0035 * 1.0 / 101.0
    assert matrix[1, 0] == pytest.approx(gross_matrix[1, 0] - expected_commission, rel=1e-9)

def test_dynamic_cost_model_custom_coeff_config_overrides_defaults():
    model = DynamicCostModel(coeff_config={"a1": 1000, "lookback": 10})

    assert model.a1 == 1000
    assert model.a2 == 0.5  # untouched default
    assert model.lookback_window == 10

def test_dynamic_cost_model_market_impact_scales_with_aum_power_law():
    position_matrix = np.array([[0.0], [1.0]])
    gross_matrix = np.array([[0.0], [0.0]])
    cache = {
        "close": pd.Series([100.0, 100.0]),
        "volume": pd.Series([10_000.0, 10_000.0]),
        "dynamic_cost": pd.DataFrame({"adv": [500_000.0, 500_000.0], "ann_vol": [0.2, 0.2]}),
    }
    model = DynamicCostModel()

    small = model.returns_matrix(np.array([1e4]), position_matrix, gross_matrix, cache)
    large = model.returns_matrix(np.array([1e8]), position_matrix, gross_matrix, cache)

    # gross_ret is 0 here, so net_ret is purely -cost: larger AUM -> larger impact -> more negative
    assert large[1, 0] < small[1, 0]

def test_dynamic_cost_model_expense_matches_returns_matrix_at_same_aum(synthetic_ohlcv):
    raw = synthetic_ohlcv(n_days=25)
    df = raw[["close", "volume"]].copy()
    df["position"] = 1.0
    df["gross_ret"] = 0.001

    cache = {}
    model = DynamicCostModel()
    result = model.expense(df.copy(), _ctx(), cache)

    matrix = model.returns_matrix(np.array([1e5]), df[["position"]].to_numpy(), df[["gross_ret"]].to_numpy(), cache)

    np.testing.assert_allclose(result["net_ret"].to_numpy(), matrix[:, 0])

def test_dynamic_cost_model_adv_and_ann_vol_match_hand_derived_groupby_across_weekend_gap(synthetic_ohlcv, random_seed):
    # pins expense()'s resample("D")+.floor("D") rewrite against the original groupby(df.index.date)
    # formula. n_days=25 spans weekend gaps, exactly where a naive resample().sum()/.last() (without
    # restricting to real trading days first) would inject zero-volume phantom days into the
    # trailing rolling ADV/vol average - this asserts against the groupby ground truth, which never
    # had that failure mode, so any reintroduced dilution bug would show up as a mismatch here.
    raw = synthetic_ohlcv(n_days=25)
    df = raw[["close", "volume"]].copy()
    df["position"] = 1.0
    df["gross_ret"] = 0.001

    lookback = DynamicCostModel.DEFAULT_PARAMS["lookback"]

    old_dates = pd.Series(df.index.date, index=df.index)
    old_buckets = df.groupby(df.index.date)
    expected_adv = old_dates.map(old_buckets["volume"].sum().rolling(lookback).mean().shift(1)).fillna(0)
    expected_ann_vol = old_dates.map(old_buckets["close"].last().pct_change().rolling(lookback).std().shift(1) * np.sqrt(252)).fillna(0)

    cache = {}
    DynamicCostModel().expense(df.copy(), _ctx(), cache)

    # Index.map() (unlike Series.map()) returns a bare Index with no index of its own, so building
    # dynamic_cost straight from it silently falls back to a RangeIndex instead of df's minute
    # DatetimeIndex - assert on the actual index, not just .to_numpy() values, or this regresses silently
    assert cache["dynamic_cost"].index.equals(df.index)
    np.testing.assert_allclose(cache["dynamic_cost"]["adv"].to_numpy(), expected_adv.to_numpy())
    np.testing.assert_allclose(cache["dynamic_cost"]["ann_vol"].to_numpy(), expected_ann_vol.to_numpy())

def test_dynamic_cost_model_full_portfolio_pipeline_smoke(synthetic_ohlcv, random_seed):
    # DynamicCostModel is never the default cost model, so nothing else in this module runs it
    # through a real Portfolio end-to-end - this is what actually catches an expense()-internal
    # index bug: _backtest() re-indexes every cache entry down to the post-warmup df.index
    # (`series.loc[df.index]`), which raises KeyError if dynamic_cost silently held a RangeIndex
    from backtest_engine.core import Portfolio

    df = synthetic_ohlcv(n_days=25)
    p = Portfolio(df, cost_model=DynamicCostModel())

    assert not p.df["net_ret"].isna().any()
