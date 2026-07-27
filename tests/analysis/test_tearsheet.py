from __future__ import annotations

import io
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import pytest

from backtest_engine.analysis.metrics import SeriesMetrics, TradeMetrics, RelativeMetrics
from backtest_engine.analysis.tearsheet import Tearsheet

@pytest.fixture
def tearsheet(portfolio_factory) -> Tearsheet:
    portfolio = portfolio_factory()
    return Tearsheet(portfolio.stats)

def test_init_computes_strategy_benchmark_trades_relative(tearsheet):
    assert isinstance(tearsheet.strategy, SeriesMetrics)
    assert isinstance(tearsheet.benchmark, SeriesMetrics)
    assert isinstance(tearsheet.trades, TradeMetrics)
    assert isinstance(tearsheet.relative, RelativeMetrics)
    assert tearsheet.strategy.total_days > 0

def test_plot_creates_single_figure(tearsheet):
    figs_before = len(plt.get_fignums())
    tearsheet.plot()

    assert len(plt.get_fignums()) == figs_before + 1

def test_plot_savepath_none_leaves_figure_open(tearsheet):
    figs_before = len(plt.get_fignums())
    tearsheet.plot(savepath=None)

    assert len(plt.get_fignums()) == figs_before + 1

def test_plot_savepath_saves_and_closes_figure(tearsheet, tmp_path):
    figs_before = len(plt.get_fignums())
    tearsheet.plot(savepath=tmp_path)

    assert len(plt.get_fignums()) == figs_before

    created = list(tmp_path.iterdir())
    assert len(created) == 1
    assert created[0].name.startswith("Tearsheet_")
    assert (created[0] / "daily_returns_distribution.png").exists()

def test_str_contains_all_expected_section_headers(tearsheet):
    text = str(tearsheet)

    for section in ["Return Profile", "Trades", "Drawdowns", "Risk", "Ratios", "Relative (vs Benchmark)"]:
        assert section in text

def test_str_contains_both_column_headers(tearsheet):
    text = str(tearsheet)

    assert "Strategy" in text
    assert "Benchmark" in text

def test_report_plots_and_prints(tearsheet):
    figs_before = len(plt.get_fignums())
    buf = io.StringIO()

    with redirect_stdout(buf):
        tearsheet.report()

    assert len(plt.get_fignums()) == figs_before + 1
    assert "Sharpe Ratio" in buf.getvalue()

def test_report_savepath_saves_and_closes_figure(tearsheet, tmp_path):
    figs_before = len(plt.get_fignums())

    tearsheet.report(savepath=tmp_path)

    assert len(plt.get_fignums()) == figs_before
    assert len(list(tmp_path.iterdir())) == 1
