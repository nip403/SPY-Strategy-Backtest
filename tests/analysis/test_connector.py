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

def test_has_long_leg_true_for_default_mixed_strategy(connector):
    assert connector._has_long_leg is True

def test_has_long_leg_false_for_short_only_strategy(portfolio_factory):
    portfolio = portfolio_factory(n_days=25, long_permissions=False)
    book, _ = generate_toy_equity(portfolio=portfolio, sharpe=1.0, volatility=0.15, beta=0.5, benchmark=portfolio.df["close"], random_seed=1)
    sc = StrategyConnector(portfolio, book, portfolio.df["close"])

    assert sc._has_long_leg is False

def test_short_only_combined_growth_matches_book_plus_additive_short(portfolio_factory):
    # single-rebalance-block approximation: rebalance_period=20 < 25 days means 2 blocks (each
    # re-based to the mixed portfolio's own growth at that point), so this is a loose sanity
    # check on magnitude/direction, not exact - exact formula correctness is covered by the
    # hand-derived _mix_returns unit tests above
    portfolio = portfolio_factory(n_days=25, long_permissions=False)
    book, _ = generate_toy_equity(portfolio=portfolio, sharpe=1.0, volatility=0.15, beta=0.5, benchmark=portfolio.df["close"], random_seed=1)
    sc = StrategyConnector(portfolio, book, portfolio.df["close"])

    w = sc._naive_w
    combined_growth = (1 + sc.daily["combined"]).prod()
    book_growth = (1 + sc.daily["book"]).prod()
    short_growth = (1 + sc._daily_short).prod()

    approx_expected = book_growth + w * (short_growth - 1)
    assert combined_growth == pytest.approx(approx_expected, rel=0.05)

def test_daily_has_no_long_short_columns(connector):
    # long/short legs are internal-only calc intermediates (self._daily_long/_daily_short),
    # never exposed on the public df/daily frames
    assert "long" not in connector.daily.columns
    assert "short" not in connector.daily.columns
    assert "long" not in connector.df.columns
    assert "short" not in connector.df.columns

# ---- _mix_returns, tested via a hand-derived single-rebalance-block case -------------------

class _FakeConnector:
    """Minimal stand-in exposing only what _mix_returns actually reads."""

    def __init__(self, book_daily: list[float], rebalance_period: int, has_long_leg: bool = True) -> None:
        self.daily = pd.DataFrame({"book": book_daily})
        self.rebalance_period = rebalance_period
        self._has_long_leg = has_long_leg

def test_mix_returns_zero_short_leg_reduces_to_book_long_blend():
    # anchors old->new behaviour: with an all-zero short leg (no additive contribution), the
    # 3-term mix must reduce exactly to the old 2-term capital-reallocation blend
    book_daily = [0.005, 0.01, -0.002]
    long_daily = [0.01, 0.02, 0.03]
    short_daily = [0.0, 0.0, 0.0]
    fake = _FakeConnector(book_daily=book_daily, rebalance_period=20, has_long_leg=True)  # all 3 days fall in one block

    result = StrategyConnector._mix_returns(
        fake, np.array([0.5]), np.array(long_daily).reshape(-1, 1), np.array(short_daily).reshape(-1, 1)
    )

    # within a single unbroken rebalance block, portfolio value each day is exactly the
    # weighted sum of the two legs' own cumulative growth (both start at 1)
    cum_l = np.cumprod(1 + np.array(long_daily))
    cum_b = np.cumprod(1 + np.array(book_daily))
    port_value = 0.5 * cum_l + 0.5 * cum_b
    expected = port_value / np.concatenate([[1.0], port_value[:-1]]) - 1

    np.testing.assert_allclose(result[:, 0], expected)

def test_mix_returns_short_leg_layers_additively_on_top():
    # the short leg is margin-funded: its dollar P&L (scaled by the same weight used for the
    # long leg's capital withdrawal) is ADDED on top of the book+long blend, not reallocated
    # away from it - book+long alone would still sum to less than the full 3-term result
    book_daily = [0.005, 0.01, -0.002]
    long_daily = [0.01, 0.02, 0.03]
    short_daily = [0.02, -0.01, 0.015]
    fake = _FakeConnector(book_daily=book_daily, rebalance_period=20, has_long_leg=True)

    result = StrategyConnector._mix_returns(
        fake, np.array([0.4]), np.array(long_daily).reshape(-1, 1), np.array(short_daily).reshape(-1, 1)
    )

    cum_l = np.cumprod(1 + np.array(long_daily))
    cum_s = np.cumprod(1 + np.array(short_daily))
    cum_b = np.cumprod(1 + np.array(book_daily))
    port_value = 0.4 * cum_l + 0.6 * cum_b + 0.4 * (cum_s - 1)
    expected = port_value / np.concatenate([[1.0], port_value[:-1]]) - 1

    np.testing.assert_allclose(result[:, 0], expected)

def test_mix_returns_no_long_leg_assumes_no_capital_withdrawal():
    # with has_long_leg=False, the long_daily_matrix argument must be ignored entirely: book
    # keeps its full, undiminished growth, and weight purely scales the short leg's additive term
    book_daily = [0.005, 0.01, -0.002]
    long_daily = [0.5, -0.9, 10.0]  # deliberately extreme/absurd - must have zero effect
    short_daily = [0.02, -0.01, 0.015]
    fake = _FakeConnector(book_daily=book_daily, rebalance_period=20, has_long_leg=False)

    result = StrategyConnector._mix_returns(
        fake, np.array([0.4]), np.array(long_daily).reshape(-1, 1), np.array(short_daily).reshape(-1, 1)
    )

    cum_s = np.cumprod(1 + np.array(short_daily))
    cum_b = np.cumprod(1 + np.array(book_daily))
    port_value = cum_b + 0.4 * (cum_s - 1)  # book at full weight, no (1-weight) reduction
    expected = port_value / np.concatenate([[1.0], port_value[:-1]]) - 1

    np.testing.assert_allclose(result[:, 0], expected)

def test_mix_returns_no_long_leg_zero_weight_matches_book_exactly():
    book_daily = [0.005, 0.01, -0.002]
    short_daily = [0.02, -0.01, 0.015]
    fake = _FakeConnector(book_daily=book_daily, rebalance_period=20, has_long_leg=False)

    result = StrategyConnector._mix_returns(
        fake, np.array([0.0]), np.zeros((3, 1)), np.array(short_daily).reshape(-1, 1)
    )

    np.testing.assert_allclose(result[:, 0], book_daily)

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
