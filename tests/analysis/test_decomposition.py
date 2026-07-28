from __future__ import annotations

import io
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pytest

from backtest_engine.components.base import StrategyComponent
from backtest_engine.analysis.tearsheet import Tearsheet
from backtest_engine.analysis.decomposition import PortfolioDecomposer

@pytest.fixture
def decomposer(portfolio_factory) -> PortfolioDecomposer:
    portfolio = portfolio_factory()
    return PortfolioDecomposer(portfolio, portfolio.t0, portfolio.t1)

# ---- split_long_short ----------------------------------------------------

def test_split_long_short_hand_derived_with_sign_flip():
    # bar 0: long entry; bar 1: flips long->short (crosses both legs in one trade); bar 2: holds short
    position = np.array([1.0, -0.5, -0.5]).reshape(-1, 1)
    ret = np.array([0.01, 0.02, -0.01]).reshape(-1, 1)
    gross = np.array([0.0, 0.02, 0.005]).reshape(-1, 1)  # shift(position,1)*ret
    net = np.array([0.0, 0.017, 0.005]).reshape(-1, 1)   # cost of 0.003 charged only at the flip bar

    legs = PortfolioDecomposer.split_long_short(position, ret, gross, net)

    np.testing.assert_allclose(legs["long"]["position"][:, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(legs["short"]["position"][:, 0], [0.0, -0.5, -0.5])

    # flip bar's cost (0.003) splits 2:1 proportional to each leg's own |position change| (1.0 vs 0.5)
    np.testing.assert_allclose(legs["long"]["net_ret"][:, 0], [0.0, 0.018, 0.0])
    np.testing.assert_allclose(legs["short"]["net_ret"][:, 0], [0.0, -0.001, 0.005])

    # legs must always recombine to the original combined net_ret (cost is split, never created/destroyed)
    recombined = legs["long"]["net_ret"] + legs["short"]["net_ret"]
    np.testing.assert_allclose(recombined, net)

def test_split_long_short_broadcasts_ret_across_scenarios():
    position = np.array([[1.0, 1.0], [-0.5, -0.5]])
    ret = np.array([[0.01], [0.02]])
    gross = np.array([[0.0, 0.0], [0.02, 0.02]])
    net = np.array([[0.0, 0.0], [0.017, 0.017]])

    legs = PortfolioDecomposer.split_long_short(position, ret, gross, net)

    assert legs["long"]["net_ret"].shape == (2, 2)
    np.testing.assert_allclose(legs["long"]["net_ret"][:, 0], legs["long"]["net_ret"][:, 1])

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

def test_plot_shows_and_closes_two_figures(decomposer, captured_figures):
    figs_before = len(plt.get_fignums())
    decomposer.plot()

    assert len(captured_figures) == 2
    assert len(plt.get_fignums()) == figs_before

def test_plot_savepath_saves_and_closes_figures(decomposer, captured_figures, tmp_path):
    figs_before = len(plt.get_fignums())
    decomposer.plot(savepath=tmp_path)

    assert len(captured_figures) == 2
    assert len(plt.get_fignums()) == figs_before

    created = list(tmp_path.iterdir())
    assert len(created) == 1
    assert created[0].name.startswith("PortfolioDecomposer_")
    assert sorted(p.name for p in created[0].iterdir()) == ["equity_curves.png", "returns_distributions.png"]

def test_str_reports_four_columns_and_no_stale_label_typo(decomposer):
    text = str(decomposer)

    for header in ["Strategy", "Long-Only", "Short-Only", "Benchmark"]:
        assert header in text

    assert "Strat_Long" not in text  # regression: old copy-paste label bug

def test_report_plots_and_prints(decomposer, captured_figures):
    figs_before = len(plt.get_fignums())
    buf = io.StringIO()

    with redirect_stdout(buf):
        decomposer.report()

    assert len(captured_figures) == 2
    assert len(plt.get_fignums()) == figs_before
    assert "Long-Only" in buf.getvalue()
