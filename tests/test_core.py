from __future__ import annotations

import io
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from backtest_engine.core import Portfolio
from backtest_engine.analysis.metrics import DailySnapshot
from backtest_engine.analysis.tearsheet import Tearsheet
from backtest_engine.analysis.decomposition import PortfolioDecomposer
from backtest_engine.components.strategy import BaseStrategy
from backtest_engine.components.execution import NaiveExecution
from backtest_engine.components.cost_model import FlatCostModel

# ---- construction / defaults ------------------------------------------------

def test_defaults_are_base_strategy_naive_execution_flat_cost(synthetic_ohlcv):
    df = synthetic_ohlcv(n_days=20)
    p = Portfolio(df)

    assert isinstance(p.strategy, BaseStrategy)
    assert isinstance(p.execution, NaiveExecution)
    assert isinstance(p.cost_model, FlatCostModel)

def test_init_populates_df_stats_t0_t1(portfolio_factory):
    p = portfolio_factory()

    assert isinstance(p.df, pd.DataFrame)
    assert isinstance(p.stats, pd.DataFrame)
    assert len(p.df) > 0
    assert len(p.stats) > 0
    assert p.t0 <= p.t1

def test_cache_trimmed_to_final_df_index(portfolio_factory):
    p = portfolio_factory()

    for series in p.cache.values():
        assert series.index.equals(p.df.index)

# ---- _preprocess --------------------------------------------------------

def test_preprocess_adds_all_indicator_columns(portfolio_factory):
    p = portfolio_factory()

    for col in ["vwap", "daily_open", "prev_close", "std", "ret", "deviation", "sigma", "upper_bound", "lower_bound", "long_stop", "short_stop"]:
        assert col in p.df.columns

def test_preprocess_bounds_ordering(portfolio_factory):
    p = portfolio_factory()

    # sigma/std are non-negative deviations, so upper >= lower always (post-warmup, where defined)
    valid = p.df[["upper_bound", "lower_bound", "long_stop", "short_stop"]].dropna()
    assert not valid.empty
    assert (valid["upper_bound"] >= valid["lower_bound"]).all()
    assert (valid["long_stop"] >= valid["short_stop"]).all()

# ---- _preprocess / _aggregate: resample("D")/.floor("D") rewrite vs. the original groupby(date) formula ----
# n_days=25 (> 1 trading week) guarantees these span weekend gaps, which is exactly the case where
# a naive resample would diverge from groupby(df.index.date) (see core.py's _preprocess/_aggregate)

def test_preprocess_prev_close_and_std_match_hand_derived_groupby(portfolio_factory, synthetic_ohlcv, random_seed):
    raw = synthetic_ohlcv(n_days=25)
    p = portfolio_factory(n_days=25)

    old_closes = raw.groupby(raw.index.date)["close"].last()
    expected_prev_close = pd.Series(raw.index.date, index=raw.index).map(old_closes.shift(1))
    expected_std = pd.Series(raw.index.date, index=raw.index).map(old_closes.pct_change().rolling(window=14).std().shift(1))

    pd.testing.assert_series_equal(p.df["prev_close"], expected_prev_close.loc[p.df.index], check_names=False)
    pd.testing.assert_series_equal(p.df["std"], expected_std.loc[p.df.index].astype(float), check_names=False)

def test_preprocess_vwap_matches_hand_derived_groupby(portfolio_factory, synthetic_ohlcv, random_seed):
    raw = synthetic_ohlcv(n_days=25)
    p = portfolio_factory(n_days=25)

    pv = raw["volume"] * (raw["high"] + raw["low"] + raw["close"]) / 3
    expected_vwap = pv.groupby(raw.index.date).cumsum() / raw["volume"].groupby(raw.index.date).cumsum()

    pd.testing.assert_series_equal(p.df["vwap"], expected_vwap.loc[p.df.index], check_names=False)

def test_aggregate_index_has_no_extra_weekend_rows_and_stays_date_typed(portfolio_factory):
    # regression guard: resample("D") bins every calendar day including weekends with no data -
    # _aggregate must filter those back out and relabel to plain `date` objects (self.t0/t1 and
    # every .loc[date...] slice elsewhere depend on this exact index contract)
    p = portfolio_factory(n_days=25)

    expected_days = sorted(set(p.df.index.date))
    assert list(p.stats.index) == expected_days
    assert all(type(d) is type(p.t0) for d in p.stats.index)

# ---- _backtest -----------------------------------------------------------

def test_backtest_drops_warmup_nan_rows(portfolio_factory):
    p = portfolio_factory(n_days=20)

    assert not p.df[["close", "volume", "ret", "position", "net_ret"]].isna().any().any()

def test_backtest_gross_ret_matches_shifted_position_times_ret(portfolio_factory):
    p = portfolio_factory()

    expected = p.df["position"].shift(1).fillna(0) * p.df["ret"]
    # boundary row (first surviving bar) shifts against pre-dropna history, so only
    # interior rows are re-derivable purely from the already-trimmed frame
    pd.testing.assert_series_equal(p.df["gross_ret"].iloc[1:], expected.iloc[1:], check_names=False)

def test_backtest_equity_curve_matches_cumprod_of_net_ret(portfolio_factory):
    p = portfolio_factory(aum=50_000)

    expected = 50_000 * (1 + p.df["net_ret"].fillna(0)).cumprod()
    pd.testing.assert_series_equal(p.df["equity_curve"], expected, check_names=False)

def test_backtest_benchmark_matches_buy_and_hold_on_ret(portfolio_factory):
    p = portfolio_factory(aum=50_000)

    expected = (1 + p.df["ret"].fillna(0)).cumprod() * 50_000
    pd.testing.assert_series_equal(p.df["benchmark"], expected, check_names=False)

# ---- returns_matrix --------------------------------------------------------

def test_returns_matrix_shape_matches_aum_length(portfolio_factory):
    p = portfolio_factory()

    matrix = p.returns_matrix(np.array([1e4, 1e5, 1e6, 1e7]))

    assert matrix.shape == (len(p.df), 4)

def test_returns_matrix_values_finite_or_nan_only(portfolio_factory):
    p = portfolio_factory()

    matrix = p.returns_matrix(np.array([1e4, 1e6]))

    assert np.all(np.isfinite(matrix) | np.isnan(matrix))

def test_returns_matrix_components_net_matches_returns_matrix(portfolio_factory):
    p = portfolio_factory()
    aum = np.array([1e4, 1e5, 1e6])

    position_matrix, gross_matrix, net_matrix = p.returns_matrix_components(aum)

    assert position_matrix.shape == (len(p.df), 3)
    assert gross_matrix.shape == (len(p.df), 3)
    assert net_matrix.shape == (len(p.df), 3)
    np.testing.assert_array_equal(net_matrix, p.returns_matrix(aum))

def test_returns_matrix_components_gross_matches_shifted_position_times_ret(portfolio_factory):
    p = portfolio_factory()
    aum = np.array([1e5])

    position_matrix, gross_matrix, _ = p.returns_matrix_components(aum)

    ret = p.df["ret"].to_numpy()[:, None]
    expected_gross = np.vstack([np.zeros((1, 1)), position_matrix[:-1]]) * ret

    np.testing.assert_array_equal(gross_matrix, expected_gross)

# ---- _aggregate --------------------------------------------------------

def test_aggregate_first_day_return_baselined_on_aum(portfolio_factory):
    p = portfolio_factory(aum=100_000)

    expected_first = (p.stats["strat_equity"].iloc[0] / 100_000) - 1
    assert p.stats["strat_ret"].iloc[0] == pytest.approx(expected_first)

def test_aggregate_trade_counts_are_non_negative_integers(portfolio_factory):
    p = portfolio_factory()

    assert (p.stats["trade_count"] >= 0).all()
    assert (p.stats["trade_wins"] >= 0).all()
    assert p.stats["trade_count"].dtype.kind in "iu"

# ---- sharpe property --------------------------------------------------------

def test_sharpe_property_matches_hand_computed_value(portfolio_factory):
    p = portfolio_factory()

    r = p.stats["strat_ret"]
    expected = r.mean() / r.std() * 252 ** 0.5

    assert p.sharpe == pytest.approx(expected)

# ---- report() ----------------------------------------------------------

def test_report_default_period_returns_tearsheet(portfolio_factory):
    p = portfolio_factory()

    result = p.report(plot=False)

    assert isinstance(result, Tearsheet)

def test_report_decompose_returns_portfolio_decomposer(portfolio_factory):
    p = portfolio_factory()

    result = p.report(decompose=True, plot=False)

    assert isinstance(result, PortfolioDecomposer)

def test_report_day_returns_daily_snapshot_matching_stats_row(portfolio_factory):
    p = portfolio_factory()

    result = p.report(day=p.t0, plot=False)

    assert isinstance(result, DailySnapshot)
    assert result.strat_cum_return == pytest.approx(p.stats.loc[p.t0]["strat_ret"])
    assert result.bench_cum_return == pytest.approx(p.stats.loc[p.t0]["bench_ret"])

def test_report_start_end_slicing_returns_narrower_tearsheet(portfolio_factory):
    p = portfolio_factory(n_days=20)

    full = p.report(plot=False)
    half_end = p.stats.index[len(p.stats) // 2]
    partial = p.report(end=half_end, plot=False)

    assert partial.strategy.total_days < full.strategy.total_days

def test_report_plot_false_creates_no_figures_and_prints_nothing(portfolio_factory):
    p = portfolio_factory()

    figs_before = len(plt.get_fignums())
    buf = io.StringIO()
    with redirect_stdout(buf):
        p.report(plot=False)

    assert len(plt.get_fignums()) == figs_before
    assert buf.getvalue() == ""

def test_report_plot_true_period_mode_shows_and_closes_two_figures(portfolio_factory, captured_figures):
    p = portfolio_factory()

    figs_before = len(plt.get_fignums())
    p.report(plot=True)

    assert len(captured_figures) == 2
    assert len(plt.get_fignums()) == figs_before

def test_report_plot_true_decompose_mode_skips_manual_chart_defers_to_decomposer(portfolio_factory, captured_figures):
    p = portfolio_factory()

    figs_before = len(plt.get_fignums())
    p.report(decompose=True, plot=True)

    # decompose=True suppresses the ad-hoc equity plot block entirely; PortfolioDecomposer.report()
    # (which itself renders 2 figures) is the sole source of figures in this branch
    assert len(captured_figures) == 2
    assert len(plt.get_fignums()) == figs_before

def test_report_day_plot_true_prints_snapshot(portfolio_factory):
    p = portfolio_factory()

    buf = io.StringIO()
    with redirect_stdout(buf):
        p.report(day=p.t0, plot=True)

    assert "Strategy Return" in buf.getvalue()
    assert "Benchmark Return" in buf.getvalue()

def test_report_day_plot_false_prints_nothing(portfolio_factory):
    p = portfolio_factory()

    buf = io.StringIO()
    with redirect_stdout(buf):
        p.report(day=p.t0, plot=False)

    assert buf.getvalue() == ""

# ---- savepath --------------------------------------------------------

def test_report_period_mode_savepath_saves_two_sibling_directories_and_closes_figures(portfolio_factory, captured_figures, tmp_path):
    p = portfolio_factory()

    figs_before = len(plt.get_fignums())
    p.report(plot=True, savepath=tmp_path)

    assert len(captured_figures) == 2
    assert len(plt.get_fignums()) == figs_before

    created = sorted(p.name for p in tmp_path.iterdir())
    assert len(created) == 2
    assert created[0].startswith("Portfolio_")
    assert created[1].startswith("Tearsheet_")

def test_report_decompose_mode_savepath_saves_decomposer_directory_only(portfolio_factory, captured_figures, tmp_path):
    p = portfolio_factory()

    figs_before = len(plt.get_fignums())
    p.report(decompose=True, plot=True, savepath=tmp_path)

    assert len(captured_figures) == 2
    assert len(plt.get_fignums()) == figs_before

    created = list(tmp_path.iterdir())
    assert len(created) == 1
    assert created[0].name.startswith("PortfolioDecomposer_")

def test_report_day_mode_savepath_saves_and_closes_figure(portfolio_factory, captured_figures, tmp_path):
    p = portfolio_factory()

    figs_before = len(plt.get_fignums())
    p.report(day=p.t0, plot=True, savepath=tmp_path)

    assert len(captured_figures) == 1
    assert len(plt.get_fignums()) == figs_before

    created = list(tmp_path.iterdir())
    assert len(created) == 1
    assert created[0].name.startswith("Portfolio_")
    assert (created[0] / "noise_area_and_leverage.png").exists()

# ---- __str__ --------------------------------------------------------

def test_str_contains_aum_sharpe_and_period(portfolio_factory):
    p = portfolio_factory(aum=123_000)

    text = str(p)

    assert "123,000" in text
    assert "Sharpe" in text
    assert str(p.t0) in text
    assert str(p.t1) in text

# ---- sharpe_curve --------------------------------------------------------

class TestSharpeCurve:
    def test_runs_shows_and_closes_one_figure(self, synthetic_ohlcv, cycling_strategy, captured_figures):
        df = synthetic_ohlcv(n_days=20)

        figs_before = len(plt.get_fignums())
        Portfolio.sharpe_curve(df=df, strategy_model=cycling_strategy, min_aum=1e4, max_aum=1e8, resolution=5)

        assert len(captured_figures) == 1
        assert len(plt.get_fignums()) == figs_before

    def test_legend_has_exactly_three_entries_in_order(self, synthetic_ohlcv, cycling_strategy, captured_figures):
        df = synthetic_ohlcv(n_days=20)
        Portfolio.sharpe_curve(df=df, strategy_model=cycling_strategy, min_aum=1e4, max_aum=1e8, resolution=5)

        ax = captured_figures[-1].axes[0]
        labels = [t.get_text() for t in ax.get_legend().get_texts()]

        assert len(labels) == 3
        assert labels[0] == "Sharpe"
        assert labels[1].startswith("Base (AUM=")
        assert labels[2] == "Risk Free"

    def test_base_legend_aum_format_one_decimal_no_plus_sign(self, synthetic_ohlcv, cycling_strategy, captured_figures):
        df = synthetic_ohlcv(n_days=20)
        Portfolio.sharpe_curve(df=df, strategy_model=cycling_strategy, min_aum=1e4, max_aum=1e8, resolution=5)

        ax = captured_figures[-1].axes[0]
        base_label = next(t.get_text() for t in ax.get_legend().get_texts() if t.get_text().startswith("Base"))
        inner = base_label.split("AUM=", 1)[1].rstrip(")")

        assert "+" not in inner
        if inner != "N/A":
            mantissa, exp = inner.split("e")
            assert len(mantissa.split(".")[1]) == 1
            assert exp.lstrip("-").isdigit()

    def test_no_minor_xticks(self, synthetic_ohlcv, cycling_strategy, captured_figures):
        df = synthetic_ohlcv(n_days=20)
        Portfolio.sharpe_curve(df=df, strategy_model=cycling_strategy, min_aum=1e4, max_aum=1e8, resolution=5)

        ax = captured_figures[-1].axes[0]
        assert len(ax.xaxis.get_minorticklocs()) == 0

    def test_base_aum_not_on_grid_gets_merged_in(self, synthetic_ohlcv, cycling_strategy, captured_figures):
        # 2e5 falls between resolution=5's sampled points (1,3,5,7,9 x 10^k), forcing the
        # "append base_aum and re-sort" branch rather than reusing an existing grid point
        df = synthetic_ohlcv(n_days=20)

        Portfolio.sharpe_curve(df=df, strategy_model=cycling_strategy, min_aum=1e4, max_aum=1e8, resolution=5, base_aum=2e5)

        ax = captured_figures[-1].axes[0]
        assert ax.get_legend() is not None

    def test_base_aum_out_of_range_is_clipped_not_erroring(self, synthetic_ohlcv, cycling_strategy, captured_figures):
        df = synthetic_ohlcv(n_days=20)

        Portfolio.sharpe_curve(df=df, strategy_model=cycling_strategy, min_aum=1e4, max_aum=1e8, resolution=5, base_aum=1e20)

        assert captured_figures[-1].axes[0].get_legend() is not None

    def test_axis_labels_and_title(self, synthetic_ohlcv, cycling_strategy, captured_figures):
        df = synthetic_ohlcv(n_days=20)
        Portfolio.sharpe_curve(df=df, strategy_model=cycling_strategy, min_aum=1e4, max_aum=1e8, resolution=5)

        ax = captured_figures[-1].axes[0]
        assert ax.get_xlabel() == "AUM ($, Piecewise-Linear-Scaled)"
        assert ax.get_ylabel() == "Sharpe Ratio"
        assert "Capacity: Sharpe vs AUM" in ax.get_title()

    def test_savepath_saves_and_closes_figure(self, synthetic_ohlcv, cycling_strategy, captured_figures, tmp_path):
        df = synthetic_ohlcv(n_days=20)

        figs_before = len(plt.get_fignums())
        Portfolio.sharpe_curve(df=df, strategy_model=cycling_strategy, min_aum=1e4, max_aum=1e8, resolution=5, savepath=tmp_path)

        assert len(captured_figures) == 1
        assert len(plt.get_fignums()) == figs_before

        created = list(tmp_path.iterdir())
        assert len(created) == 1
        assert created[0].name.startswith("Portfolio_")
        assert (created[0] / "sharpe_capacity_curve.png").exists()
