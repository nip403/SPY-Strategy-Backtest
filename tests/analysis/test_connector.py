from __future__ import annotations

import io
from contextlib import redirect_stdout
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from backtest_engine.analysis.metrics import ConnectorExtras
from backtest_engine.analysis.connector import StrategyConnector
from backtest_engine.utils import generate_toy_equity

@pytest.fixture
def connector(portfolio_factory, random_seed) -> StrategyConnector:
    portfolio = portfolio_factory(n_days=25)
    book, _ = generate_toy_equity(portfolio=portfolio, sharpe=1.0, volatility=0.15, beta=0.5, benchmark=portfolio.df["close"], random_seed=random_seed)
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

def test_has_long_leg_false_for_short_only_strategy(portfolio_factory, random_seed):
    portfolio = portfolio_factory(n_days=25, long_permissions=False)
    book, _ = generate_toy_equity(portfolio=portfolio, sharpe=1.0, volatility=0.15, beta=0.5, benchmark=portfolio.df["close"], random_seed=random_seed)
    sc = StrategyConnector(portfolio, book, portfolio.df["close"])

    assert sc._has_long_leg is False

def test_short_only_combined_growth_matches_book_plus_additive_short(portfolio_factory, random_seed):
    # single-rebalance-block approximation: rebalance_period=20 < 25 days means 2 blocks (each
    # re-based to the mixed portfolio's own growth at that point), so this is a loose sanity
    # check on magnitude/direction, not exact - exact formula correctness is covered by the
    # hand-derived _mix_returns unit tests above
    portfolio = portfolio_factory(n_days=25, long_permissions=False)
    book, _ = generate_toy_equity(portfolio=portfolio, sharpe=1.0, volatility=0.15, beta=0.5, benchmark=portfolio.df["close"], random_seed=random_seed)
    sc = StrategyConnector(portfolio, book, portfolio.df["close"])

    w = sc._naive_w
    combined_growth = (1 + sc.daily["combined"]).prod()
    book_growth = (1 + sc.daily["book"]).prod()
    short_growth = (1 + sc._daily_short).prod()

    approx_expected = book_growth + w * (short_growth - 1)
    assert combined_growth == pytest.approx(approx_expected, rel=0.05)

def test_daily_matches_hand_derived_compounded_returns_across_weekend_gap(connector):
    # pins self.daily's resample("D") rewrite against the original groupby(df.index.date) formula
    # (connector fixture uses n_days=25, which spans weekend gaps)
    expected = (1 + connector.df).groupby(connector.df.index.date).prod() - 1
    pd.testing.assert_frame_equal(connector.daily[["strat", "book", "bench"]], expected, check_names=False)

def test_daily_index_is_date_objects_with_no_extra_weekend_rows(connector):
    # regression guard: resample("D") bins every calendar day incl. weekends with no data - self.daily
    # must filter those back out and relabel to plain `date` objects (_exact_leg_returns.to_daily's
    # .loc[self.daily.index] depends on this exact index contract to line up correctly)
    expected_days = sorted(set(connector.df.index.date))
    assert list(connector.daily.index) == expected_days
    assert all(type(d) is date for d in connector.daily.index)

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

def test_cvar_monte_carlo_seeded_reproducible_across_runs(portfolio_factory, random_seed):
    portfolio = portfolio_factory(n_days=25)
    book, _ = generate_toy_equity(portfolio=portfolio, sharpe=1.0, volatility=0.15, beta=0.5, benchmark=portfolio.df["close"], random_seed=random_seed)

    sc1 = StrategyConnector(portfolio, book, portfolio.df["close"])
    sc2 = StrategyConnector(portfolio, book, portfolio.df["close"])

    assert sc1.extras.cvar_monte_carlo == sc2.extras.cvar_monte_carlo

# ---- _tradeoff_profile / _format_tradeoff_table --------------------------

def test_tradeoff_profile_columns_and_index_bounds(connector):
    tp = connector.tradeoff_series

    assert list(tp.columns) == ["sharpe", "exp_ret", "vol", "drag", "maxdd", "dd_days", "recovery", "cvar", "dd_relief_per_drag", "cvar_relief_per_drag"]
    assert tp.index[0] == pytest.approx(0.0)
    assert tp.index[-1] == pytest.approx(1.0)
    # >=, not ==: the sweep also unions in the 10% guarantee points _format_tradeoff_table relies on
    assert len(tp) >= round(1 / connector.weight_intervals) + 1

def test_tradeoff_profile_respects_custom_weight_intervals(portfolio_factory, random_seed):
    portfolio = portfolio_factory(n_days=25)
    book, _ = generate_toy_equity(portfolio=portfolio, sharpe=1.0, volatility=0.15, beta=0.5, benchmark=portfolio.df["close"], random_seed=random_seed)
    sc = StrategyConnector(portfolio, book, portfolio.df["close"], config={"weight_intervals": 0.25})

    index = sc.tradeoff_series.index.to_numpy()
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        assert np.any(np.isclose(index, w))

def test_format_tradeoff_table_works_with_coarse_weight_intervals(portfolio_factory, random_seed):
    # _format_tradeoff_table does an exact .loc[] lookup on the 10%-step points - _tradeoff_profile
    # must guarantee those exact points exist even when weight_intervals doesn't evenly divide 0.1
    portfolio = portfolio_factory(n_days=25)
    book, _ = generate_toy_equity(portfolio=portfolio, sharpe=1.0, volatility=0.15, beta=0.5, benchmark=portfolio.df["close"], random_seed=random_seed)
    sc = StrategyConnector(portfolio, book, portfolio.df["close"], config={"weight_intervals": 0.3})

    text = sc._format_tradeoff_table()

    for pct in ["0%", "10%", "50%", "100%"]:
        assert pct in text

def test_exact_block_leg_returns_batched_matches_individual_calls(connector, portfolio_factory):
    # _exact_block_leg_returns batches every weight scenario into one flattened (block, weight)
    # capacity-cost lookup rather than looping weight-by-weight (kept general even though current
    # callers only ever pass a single final weight) - this pins that the reshape/pick indexing
    # recovers each weight's own column correctly under batching
    portfolio = portfolio_factory(n_days=25)
    weights = np.array([0.2, 0.5, 0.9])
    batched_long, batched_short = connector._exact_block_leg_returns(portfolio, weights)

    for i, w in enumerate(weights):
        single_long, single_short = connector._exact_block_leg_returns(portfolio, np.array([w]))
        np.testing.assert_allclose(batched_long[:, i], single_long[:, 0])
        np.testing.assert_allclose(batched_short[:, i], single_short[:, 0])

def test_tradeoff_profile_weight_zero_exactly_matches_book(connector):
    # weight=0 zeroes out both legs' contribution inside _mix_returns regardless of what leg
    # returns were fed in, so this must match the book's own stats exactly, not just approximately
    book = connector.daily["book"].to_numpy()

    equity = np.cumprod(1 + book)
    peak = np.maximum.accumulate(equity)
    expected_maxdd = ((equity - peak) / peak).min()

    row0 = connector.tradeoff_series.iloc[0]
    assert row0["sharpe"] == pytest.approx(book.mean() / book.std() * np.sqrt(252))
    assert row0["maxdd"] == pytest.approx(expected_maxdd)
    assert row0["drag"] == pytest.approx(0.0, abs=1e-9)
    # relief vs. book is measured against this same row, so both ratios collapse to the 0/0 guard branch
    assert row0["dd_relief_per_drag"] == 0.0
    assert row0["cvar_relief_per_drag"] == 0.0

def test_tradeoff_profile_drawdown_stats_have_expected_sign(connector):
    tp = connector.tradeoff_series

    assert (tp["maxdd"] <= 0).all()
    assert (tp["dd_days"] >= 0).all()
    assert (tp["recovery"] >= 0).all()

def test_tradeoff_profile_crash_mask_finds_days_with_enough_history(portfolio_factory, random_seed):
    # regression guard: self.bench is tz-aware (minute OHLCV data), so reindexing its resampled
    # DatetimeIndex straight against self.daily's plain-date index without first stripping the tz/time
    # silently matches nothing, leaving every row at fill_value=False - connector's default n_days=25
    # fixture is also too short for the centred lookback window to reliably flag any crash day
    portfolio = portfolio_factory(n_days=80)
    book, _ = generate_toy_equity(portfolio=portfolio, sharpe=1.0, volatility=0.15, beta=0.5, benchmark=portfolio.df["close"], random_seed=random_seed)
    sc = StrategyConnector(portfolio, book, portfolio.df["close"])

    bench = sc.bench.resample("D").last().dropna()
    half_window = sc.lookback_window // 2
    centred_ret = bench.shift(-half_window) / bench.shift(half_window) - 1
    crash = centred_ret < centred_ret.quantile(0.10)
    crash.index = crash.index.date
    crash = crash.reindex(sc.daily.index, fill_value=False).to_numpy()

    assert crash.sum() > 0

def test_tradeoff_profile_cvar_is_mean_of_own_worst_five_percent(connector, portfolio_factory):
    # cvar must be conditioned on each scenario's OWN return distribution (95% VaR threshold), not
    # on the exogenous benchmark crash mask used for drag - cross-check via an independent, unvectorized
    # per-column computation against the vectorized np.where/nanmean implementation
    portfolio = portfolio_factory(n_days=25)
    tp = connector.tradeoff_series
    weights = tp.index.to_numpy()

    long_daily, short_daily = connector._approx_aum_leg_returns(portfolio, weights)
    returns_matrix = connector._mix_returns(weights, long_daily, short_daily)
    var_95 = np.quantile(returns_matrix, 0.05, axis=0)

    expected_cvar = np.array([
        returns_matrix[returns_matrix[:, i] <= var_95[i], i].mean()
        for i in range(returns_matrix.shape[1])
    ])

    np.testing.assert_allclose(tp["cvar"].to_numpy(), expected_cvar)

def test_tradeoff_profile_relief_ratios_match_formula(connector):
    tp = connector.tradeoff_series
    maxdd0, cvar0 = tp["maxdd"].iloc[0], tp["cvar"].iloc[0]

    nonzero_drag = tp[tp["drag"] != 0]
    expected_dd_relief = (nonzero_drag["maxdd"] - maxdd0) / nonzero_drag["drag"]
    expected_cvar_relief = (nonzero_drag["cvar"] - cvar0) / nonzero_drag["drag"]

    np.testing.assert_allclose(nonzero_drag["dd_relief_per_drag"].to_numpy(), expected_dd_relief.to_numpy())
    np.testing.assert_allclose(nonzero_drag["cvar_relief_per_drag"].to_numpy(), expected_cvar_relief.to_numpy())

def test_format_tradeoff_table_decimates_to_ten_percent_steps(connector):
    text = connector._format_tradeoff_table()

    for pct in ["0%", "10%", "50%", "100%"]:
        assert pct in text

    assert "Drag (Calm-Period Cost)" in text
    assert "95% CVaR" in text
    assert "DD Relief / Drag" in text
    assert "CVaR Relief / Drag" in text

def test_tradeoff_profile_includes_optimised_weight_point(connector):
    # _format_tradeoff_table does an exact .loc[] lookup on self._opt_w, so _tradeoff_profile must
    # guarantee that exact point exists in the sweep alongside the 10%-step guarantee points
    assert np.any(np.isclose(connector.tradeoff_series.index.to_numpy(), connector._opt_w, atol=1e-4))

def test_format_tradeoff_table_places_optimum_between_bracketing_steps(connector, monkeypatch):
    # pins the user-facing example: a 36% optimum should be inserted strictly between the 30% and
    # 40% columns, not just tacked onto the end - default weight_intervals=0.01 guarantees 0.36
    # already sits on the tradeoff_series grid so the .loc[] lookup in _format_tradeoff_table succeeds
    monkeypatch.setattr(connector, "_opt_w", 0.36)
    text = connector._format_tradeoff_table()
    header_line = text.splitlines()[0]

    assert "36% (Optimal)" in header_line
    assert header_line.index("30%") < header_line.index("36% (Optimal)") < header_line.index("40%")

def test_format_tradeoff_table_optimum_on_existing_step_is_not_duplicated(connector, monkeypatch):
    # when the optimum lands exactly on a 10%-step (e.g. 40%), it should be tagged in place rather
    # than producing a second, duplicate 40% column
    monkeypatch.setattr(connector, "_opt_w", 0.4)
    text = connector._format_tradeoff_table()
    header_line = text.splitlines()[0]

    assert header_line.count("40%") == 1
    assert "40% (Optimal)" in header_line

# ---- plot / str / report ------------------------------------------------

def test_plot_shows_and_closes_figure(connector, captured_figures):
    figs_before = len(plt.get_fignums())
    connector.plot()

    assert len(captured_figures) == 3
    assert len(plt.get_fignums()) == figs_before

def test_plot_savepath_saves_and_closes_figure(connector, captured_figures, tmp_path):
    figs_before = len(plt.get_fignums())
    connector.plot(savepath=tmp_path)

    assert len(captured_figures) == 3
    assert len(plt.get_fignums()) == figs_before

    created = list(tmp_path.iterdir())
    assert len(created) == 1
    assert created[0].name.startswith("StrategyConnector_")

    saved = {p.name for p in created[0].iterdir()}
    assert saved == {"integration_overview.png", "tradeoff_sharpe_drag_es.png", "tradeoff_drawdown_profile.png"}

def test_str_contains_all_five_columns(connector):
    text = str(connector)

    for header in ["Strategy", "Book", "Bench", "Combined", "Optimised"]:
        assert header in text

def test_str_contains_incremental_and_other_sections(connector):
    text = str(connector)

    assert "Incremental (vs Book)" in text
    assert "Incremental Sharpe (Marginal)" in text
    assert "Incremental Sharpe (Realised)" in text

def test_report_plots_and_prints(connector, captured_figures):
    figs_before = len(plt.get_fignums())
    buf = io.StringIO()

    with redirect_stdout(buf):
        connector.report()

    assert len(captured_figures) == 3
    assert len(plt.get_fignums()) == figs_before
    assert "Sharpe Ratio" in buf.getvalue()
    assert "Strategy/Book Weight Tradeoff Profile" in buf.getvalue()
