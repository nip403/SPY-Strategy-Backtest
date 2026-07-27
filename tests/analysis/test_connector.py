from __future__ import annotations

import io
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from backtest_engine.analysis.metrics import ConnectorExtras
from backtest_engine.analysis.connector import StrategyConnector
from backtest_engine.utils import generate_toy_equity

@pytest.fixture
def connector(portfolio_factory) -> StrategyConnector:
    portfolio = portfolio_factory(n_days=25)
    book, _ = generate_toy_equity(portfolio=portfolio, sharpe=1.0, volatility=0.15, beta=0.5, benchmark=portfolio.df["close"], random_seed=1)
    return StrategyConnector(portfolio, book, portfolio.df["close"])

def test_returns_intersect_strategy_book_benchmark_indices(connector, portfolio_factory):
    portfolio = portfolio_factory(n_days=25)

    assert len(connector.df) <= len(portfolio.df)
    assert connector.df.index.isin(portfolio.df.index).all()

def test_naive_weight_is_point_five(connector):
    assert connector._naive_w == 0.5

def test_optimised_weight_within_zero_one_bounds(connector):
    assert 0 <= connector._opt_w <= 1

# ---- _mix_returns, tested via a hand-derived single-rebalance-block case -------------------

class _FakeConnector:
    """Minimal stand-in exposing only what _mix_returns actually reads."""

    def __init__(self, book_daily: list[float], rebalance_period: int) -> None:
        self.daily = pd.DataFrame({"book": book_daily})
        self.rebalance_period = rebalance_period

def test_mix_returns_hand_derived_single_rebalance_block():
    book_daily = [0.005, 0.01, -0.002]
    strat_daily = [0.01, 0.02, 0.03]
    fake = _FakeConnector(book_daily=book_daily, rebalance_period=20)  # all 3 days fall in one block

    result = StrategyConnector._mix_returns(fake, np.array([0.5]), np.array(strat_daily).reshape(-1, 1))

    # within a single unbroken rebalance block, portfolio value each day is exactly the
    # weighted sum of the two legs' own cumulative growth (both start at 1)
    cum_s = np.cumprod(1 + np.array(strat_daily))
    cum_b = np.cumprod(1 + np.array(book_daily))
    port_value = 0.5 * cum_s + 0.5 * cum_b
    expected = port_value / np.concatenate([[1.0], port_value[:-1]]) - 1

    np.testing.assert_allclose(result[:, 0], expected)

# ---- extras / reproducibility -----------------------------------------------

def test_extras_populated_after_init(connector):
    assert isinstance(connector.extras, ConnectorExtras)
    assert set(connector.extras.strategy_weight.keys()) == {"combined", "optimised"}

def test_cvar_monte_carlo_seeded_reproducible_across_runs(portfolio_factory):
    portfolio = portfolio_factory(n_days=25)
    book, _ = generate_toy_equity(portfolio=portfolio, sharpe=1.0, volatility=0.15, beta=0.5, benchmark=portfolio.df["close"], random_seed=1)

    sc1 = StrategyConnector(portfolio, book, portfolio.df["close"])
    sc2 = StrategyConnector(portfolio, book, portfolio.df["close"])

    assert sc1.extras.cvar_monte_carlo == sc2.extras.cvar_monte_carlo

# ---- plot / str / report ------------------------------------------------

def test_plot_creates_single_figure(connector):
    figs_before = len(plt.get_fignums())
    connector.plot()

    assert len(plt.get_fignums()) == figs_before + 1

def test_plot_savepath_saves_and_closes_figure(connector, tmp_path):
    figs_before = len(plt.get_fignums())
    connector.plot(savepath=tmp_path)

    assert len(plt.get_fignums()) == figs_before

    created = list(tmp_path.iterdir())
    assert len(created) == 1
    assert created[0].name.startswith("StrategyConnector_")
    assert (created[0] / "integration_overview.png").exists()

def test_str_contains_all_five_columns(connector):
    text = str(connector)

    for header in ["Strategy", "Book", "Bench", "Combined", "Optimised"]:
        assert header in text

def test_str_contains_incremental_and_other_sections(connector):
    text = str(connector)

    assert "Incremental (vs Book)" in text
    assert "Incremental Sharpe (Marginal)" in text
    assert "Incremental Sharpe (Realised)" in text

def test_report_plots_and_prints(connector):
    figs_before = len(plt.get_fignums())
    buf = io.StringIO()

    with redirect_stdout(buf):
        connector.report()

    assert len(plt.get_fignums()) == figs_before + 1
    assert "Sharpe Ratio" in buf.getvalue()
