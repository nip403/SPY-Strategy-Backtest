from __future__ import annotations

import backtest_engine

# ---- __all__ shape --------------------------------------------------------

def test_all_names_are_actually_importable_attributes():
    for name in backtest_engine.__all__:
        assert hasattr(backtest_engine, name), f"{name!r} listed in __all__ but not importable"

def test_all_has_no_duplicates():
    assert len(backtest_engine.__all__) == len(set(backtest_engine.__all__))

def test_core_public_classes_exported():
    for name in ["Portfolio", "Tearsheet", "PortfolioDecomposer", "StrategyConnector", "CapacityEstimator", "AnalysisReport"]:
        assert name in backtest_engine.__all__

def test_component_abcs_exported():
    for name in ["BacktestContext", "StrategyComponent", "ExecutionComponent", "CostComponent"]:
        assert name in backtest_engine.__all__

def test_strategy_implementations_exported():
    for name in ["BaseStrategy", "RollingImmediateStopStrategy", "RollingIntervalStopStrategy", "QuarterHourSampleStrategy"]:
        assert name in backtest_engine.__all__

def test_cost_model_implementations_exported():
    for name in ["FlatCostModel", "DynamicCostModel"]:
        assert name in backtest_engine.__all__

def test_metrics_dataclasses_exported():
    for name in ["SeriesMetrics", "TradeMetrics", "RelativeMetrics", "ConnectorExtras", "DailySnapshot"]:
        assert name in backtest_engine.__all__

def test_utility_functions_exported():
    for name in ["request", "generate_toy_returns", "generate_toy_equity"]:
        assert name in backtest_engine.__all__

# ---- known asymmetry: NaiveExecution is importable but not re-exported --------

def test_naive_execution_importable_but_deliberately_not_in_all():
    """NOTE: characterizes current behaviour, not asserting it's ideal. NaiveExecution is
    imported into backtest_engine/__init__.py (so `backtest_engine.NaiveExecution` works)
    but is omitted from __all__, unlike its sibling CappedVolumeExecution/
    CappedVolumeRolloverExecution, which are both imported *and* exported. This means
    `from backtest_engine import *` will not pull in NaiveExecution even though it's the
    default ExecutionComponent used by Portfolio."""

    assert hasattr(backtest_engine, "NaiveExecution")
    assert "NaiveExecution" not in backtest_engine.__all__

def test_capped_volume_executions_exported_unlike_naive():
    for name in ["CappedVolumeExecution", "CappedVolumeRolloverExecution"]:
        assert name in backtest_engine.__all__
