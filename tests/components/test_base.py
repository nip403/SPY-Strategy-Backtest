from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_engine.components.base import BacktestContext, StrategyComponent, ExecutionComponent, CostComponent

def test_strategy_component_is_not_instantiable():
    with pytest.raises(TypeError):
        StrategyComponent()

def test_execution_component_is_not_instantiable():
    with pytest.raises(TypeError):
        ExecutionComponent()

def test_cost_component_is_not_instantiable():
    with pytest.raises(TypeError):
        CostComponent()

def test_subclass_missing_abstract_method_raises_typeerror():
    class IncompleteStrategy(StrategyComponent):
        pass  # doesn't implement set()

    with pytest.raises(TypeError):
        IncompleteStrategy()

def test_subclass_implementing_abstract_method_is_instantiable():
    class MinimalStrategy(StrategyComponent):
        def set(self, df, ctx):
            return df

    MinimalStrategy()  # should not raise

def test_execution_component_default_fill_matrix_tiles_cached_position():
    class PassthroughExecution(ExecutionComponent):
        def fill(self, df, ctx, cache):
            cache["position"] = df["position"]
            return df

    exe = PassthroughExecution()
    cache = {"position": pd.Series([1.0, -1.0, 0.0])}
    matrix = exe.fill_matrix(np.array([1e4, 1e5, 1e6]), cache)

    assert matrix.shape == (3, 3)
    for col in range(3):
        np.testing.assert_array_equal(matrix[:, col], [1.0, -1.0, 0.0])

def test_backtest_context_stores_all_fields():
    ctx = BacktestContext(aum=100_000.0, target_vol=0.02, long_perm=True, short_perm=False)

    assert ctx.aum == 100_000.0
    assert ctx.target_vol == 0.02
    assert ctx.long_perm is True
    assert ctx.short_perm is False
