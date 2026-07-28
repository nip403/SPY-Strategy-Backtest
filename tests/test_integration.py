from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from backtest_engine.core import Portfolio
from backtest_engine.analysis.tearsheet import Tearsheet
from backtest_engine.analysis.decomposition import PortfolioDecomposer
from backtest_engine.analysis.connector import StrategyConnector
from backtest_engine.components.strategy import BaseStrategy, RollingImmediateStopStrategy, RollingIntervalStopStrategy, QuarterHourSampleStrategy
from backtest_engine.components.execution import NaiveExecution, CappedVolumeExecution, CappedVolumeRolloverExecution
from backtest_engine.components.cost_model import FlatCostModel, DynamicCostModel
from backtest_engine.utils import generate_toy_equity

# ---- full-pipeline smoke tests, mirroring main.ipynb's usage patterns ---------

@pytest.mark.slow
def test_full_pipeline_default_portfolio_report_modes(synthetic_ohlcv):
    df = synthetic_ohlcv(n_days=30)

    p = Portfolio(df)

    assert isinstance(str(p), str)
    assert isinstance(p.sharpe, float)

    t = p.report(plot=False)
    assert isinstance(t, Tearsheet)

    d = p.report(decompose=True, plot=False)
    assert isinstance(d, PortfolioDecomposer)

    windowed = p.report(start=p.t0, end=p.t1, plot=False)
    assert isinstance(windowed, Tearsheet)

    snapshot = p.report(day=p.t0, plot=False)
    assert snapshot.strat_cum_return is not None

@pytest.mark.slow
@pytest.mark.parametrize("strategy_cls", [BaseStrategy, RollingImmediateStopStrategy, RollingIntervalStopStrategy, QuarterHourSampleStrategy])
def test_full_pipeline_runs_for_every_real_strategy(strategy_cls, synthetic_ohlcv):
    """End-to-end wiring smoke test: each real strategy plugged into the full Portfolio
    pipeline (preprocessing -> signal -> execution -> cost -> aggregation) without crashing,
    regardless of whether it actually fires any trades on tame synthetic data (that's covered
    directly, in isolation, by components/test_strategy.py)."""

    df = synthetic_ohlcv(n_days=30)

    p = Portfolio(df, strategy_model=strategy_cls())

    assert len(p.df) > 0
    assert isinstance(p.sharpe, float)
    assert not np.isnan(p.df["net_ret"]).any()

@pytest.mark.slow
def test_sharpe_curve_with_dynamic_cost_and_capped_execution_smoke(synthetic_ohlcv):
    """Mirrors main.ipynb's capacity-analysis cells: DynamicCostModel combined with each
    capped-participation execution model, plus a non-default strategy."""

    df = synthetic_ohlcv(n_days=30)
    figs_before = len(plt.get_fignums())

    Portfolio.sharpe_curve(df=df, max_aum=1e8, resolution=5, cost_model=DynamicCostModel(), execution_model=CappedVolumeRolloverExecution())
    Portfolio.sharpe_curve(df=df, max_aum=1e8, resolution=5, cost_model=DynamicCostModel(), execution_model=CappedVolumeExecution())
    Portfolio.sharpe_curve(df=df, max_aum=1e8, resolution=5, strategy_model=QuarterHourSampleStrategy(), cost_model=DynamicCostModel(), execution_model=CappedVolumeExecution())

    assert len(plt.get_fignums()) == figs_before + 3

@pytest.mark.slow
def test_strategy_connector_full_pipeline_smoke(synthetic_ohlcv, random_seed):
    df = synthetic_ohlcv(n_days=30)

    p = Portfolio(df, cost_model=DynamicCostModel(), long_permissions=False)

    toy_book, _ = generate_toy_equity(
        portfolio=p,
        sharpe=1,
        volatility=0.18,
        beta=0.9,
        benchmark=df["close"],
        random_seed=random_seed,
    )

    sc = StrategyConnector(p, toy_book, df["close"])

    figs_before = len(plt.get_fignums())
    sc.report()

    assert len(plt.get_fignums()) > figs_before
    assert isinstance(str(sc), str)

# ---- AUM-sweep vectorization equivalence -------------------------------------

_EXECUTION_FACTORIES = {
    "NaiveExecution": lambda: NaiveExecution(),
    "CappedVolumeExecution": lambda: CappedVolumeExecution(participation_ceiling=0.1),
    "CappedVolumeRolloverExecution": lambda: CappedVolumeRolloverExecution(participation_ceiling=0.1),
}

_COST_MODEL_FACTORIES = {
    "FlatCostModel": lambda: FlatCostModel(),
    "DynamicCostModel": lambda: DynamicCostModel(),
}

@pytest.mark.slow
@pytest.mark.parametrize("cost_name", _COST_MODEL_FACTORIES)
@pytest.mark.parametrize("execution_name", _EXECUTION_FACTORIES)
def test_returns_matrix_matches_fresh_single_aum_construction(execution_name, cost_name, synthetic_ohlcv, cycling_strategy):
    """returns_matrix's vectorised AUM sweep must agree with independently constructing a fresh
    Portfolio at each swept AUM, since every ExecutionComponent.fill() delegates internally to its
    own fill_matrix(np.array([ctx.aum]), cache) - there is no independent single-AUM code path to
    drift out of sync with the vectorised sweep."""

    df = synthetic_ohlcv(n_days=30)
    own_aum = 200_000.0
    sweep = np.array([50_000.0, own_aum, 2_000_000.0])

    execution = _EXECUTION_FACTORIES[execution_name]()
    cost_model = _COST_MODEL_FACTORIES[cost_name]()

    p = Portfolio(df, aum=own_aum, strategy_model=cycling_strategy, execution_model=execution, cost_model=cost_model)
    matrix = p.returns_matrix(sweep)

    assert matrix.shape == (len(p.df), len(sweep))

    for i, a in enumerate(sweep):
        fresh_execution = _EXECUTION_FACTORIES[execution_name]()
        fresh_cost_model = _COST_MODEL_FACTORIES[cost_name]()
        fresh = Portfolio(df, aum=a, strategy_model=cycling_strategy, execution_model=fresh_execution, cost_model=fresh_cost_model)

        rtol = 1e-12 if a == own_aum else 1e-9
        np.testing.assert_allclose(matrix[:, i], fresh.df["net_ret"].to_numpy(), rtol=rtol, atol=1e-12)

# ---- CappedVolumeExecution regression coverage (two real bugs found & fixed while writing this suite) --

def test_capped_volume_execution_fill_matrix_carries_position_across_events():
    """Regression test for a bug found while writing this suite: fill_matrix's greedy resolution
    loop never carried p_current forward between signal events (missing the equivalent of
    CappedVolumeRolloverExecution's `p_current = np.where(...)` reassignment), so every event
    resolved as if starting fresh from position 0 - discarding whatever was actually filled at
    every prior event. Fixed by reassigning p_current = fills[i] each iteration.

    Hand-derived case: event 1 ramps 0 -> 0.6 (capacity 0.6, unconstrained). Event 2 targets 0.8;
    with correct carry-over and a binding capacity of 0.1, the fill can only reach 0.6+0.1=0.7 -
    the bug instead recomputed from a phantom position of 0, capping the fill at 0.1."""

    target = pd.Series([0.0, 1.0, 0.8])
    raw_capacity = pd.Series([0.0, 0.6, 0.1])
    signal = pd.Series([True, True, True])
    cache = {"ioc": pd.DataFrame({"target": target, "raw_capacity": raw_capacity, "signal": signal})}

    fills = CappedVolumeExecution(participation_ceiling=1.0).fill_matrix(np.array([1.0]), cache)[:, 0]

    np.testing.assert_allclose(fills, [0.0, 0.6, 0.7])

def test_capped_volume_execution_returns_matrix_matches_own_construction(synthetic_ohlcv, cycling_strategy):
    """Regression test for a second bug found while writing this suite: CappedVolumeExecution's
    "signal" mask forces signal=True on row 0 of the RAW dataframe (before preprocessing-warmup
    rows are dropped), which fill_matrix's event-driven resolution relies on to seed p_current.
    Portfolio._backtest used that cache once, correctly, to compute the instance's own
    df["net_ret"] - then overwrote self.cache with a version trimmed to the post-dropna index.
    Whenever indicator warmup (e.g. sigma's 14-trading-day rolling window) removed the original
    row 0, the trimmed cache's "signal" column had no True value until the first *real* position
    change, so any later fill_matrix call through the trimmed cache (returns_matrix(),
    sharpe_curve()) yielded NaN positions for every bar up to that point - silently diverging
    from the values computed during __init__. Fixed by defaulting unresolved leading rows to
    flat (0) rather than leaving them NaN, matching the portfolio's actual starting state.

    This affected Portfolio.returns_matrix() and Portfolio.sharpe_curve(...,
    execution_model=CappedVolumeExecution()) on any real dataset - main.ipynb calls the latter
    directly."""

    df = synthetic_ohlcv(n_days=30)
    p = Portfolio(df, aum=200_000.0, strategy_model=cycling_strategy, execution_model=CappedVolumeExecution(participation_ceiling=0.1))

    assert not p.df["net_ret"].isna().any()

    self_matrix = p.returns_matrix(np.array([200_000.0]))
    assert not np.isnan(self_matrix[:, 0]).any()
    np.testing.assert_allclose(self_matrix[:, 0], p.df["net_ret"].to_numpy(), rtol=1e-12, atol=1e-12)
