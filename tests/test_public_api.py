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

# ---- execution implementations exported ------------------------------------

def test_execution_implementations_exported():
    for name in ["NaiveExecution", "CappedVolumeExecution", "CappedVolumeRolloverExecution"]:
        assert name in backtest_engine.__all__
