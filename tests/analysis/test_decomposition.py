from __future__ import annotations

import io
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import pytest

from backtest_engine.components.base import StrategyComponent
from backtest_engine.analysis.tearsheet import Tearsheet
from backtest_engine.analysis.decomposition import PortfolioDecomposer

@pytest.fixture
def decomposer(portfolio_factory) -> PortfolioDecomposer:
    portfolio = portfolio_factory()
    return PortfolioDecomposer(portfolio, portfolio.t0, portfolio.t1)

def test_components_keyed_strategy_long_short(decomposer):
    assert set(decomposer.components.keys()) == {"strategy", "long", "short"}
    assert all(isinstance(v, Tearsheet) for v in decomposer.components.values())

def test_date_range_slicing_matches_requested_window(portfolio_factory):
    portfolio = portfolio_factory(n_days=20)
    full_days = portfolio.stats.shape[0]

    half_end = portfolio.stats.index[full_days // 2]
    decomp = PortfolioDecomposer(portfolio, portfolio.t0, half_end)

    assert decomp.components["strategy"].strategy.total_days < full_days
    assert decomp.components["strategy"].strategy.total_days == portfolio.stats.loc[portfolio.t0: half_end].shape[0]

def test_long_short_split_correctly_isolates_exposure(portfolio_factory):
    class AlwaysLongStrategy(StrategyComponent):
        def set(self, df, ctx):
            df["position"] = 1.0
            return df

    portfolio = portfolio_factory(strategy=AlwaysLongStrategy())
    decomp = PortfolioDecomposer(portfolio, portfolio.t0, portfolio.t1)

    assert decomp.components["short"].trades.total_trades == 0
    assert decomp.components["long"].trades.total_trades > 0

def test_plot_returns_two_figures(decomposer):
    figs_before = len(plt.get_fignums())
    figs = decomposer.plot()

    assert len(figs) == 2
    assert len(plt.get_fignums()) == figs_before + 2

def test_str_reports_four_columns_and_no_stale_label_typo(decomposer):
    text = str(decomposer)

    for header in ["Strategy", "Long-Only", "Short-Only", "Benchmark"]:
        assert header in text

    assert "Strat_Long" not in text  # regression: old copy-paste label bug

def test_report_plots_and_prints(decomposer):
    figs_before = len(plt.get_fignums())
    buf = io.StringIO()

    with redirect_stdout(buf):
        decomposer.report()

    assert len(plt.get_fignums()) == figs_before + 2
    assert "Long-Only" in buf.getvalue()
