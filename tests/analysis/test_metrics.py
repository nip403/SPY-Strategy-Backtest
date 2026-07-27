from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_engine.analysis.metrics import (
    SeriesMetrics, TradeMetrics, RelativeMetrics,
    compute_series_metrics, compute_trade_metrics, compute_relative_metrics,
    format_value, dataclass_rows, merge_groups, render_sections,
)

# ---- compute_series_metrics --------------------------------------------------

def test_compute_series_metrics_hand_derived_small_series():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    drawdown = pd.Series([0.0, -0.05, -0.10, -0.03, 0.0])

    result = compute_series_metrics(returns, drawdown)

    assert result.total_days == 5
    assert result.cum_return == pytest.approx((1 + returns).prod() - 1)
    assert result.ann_return == pytest.approx((1 + result.cum_return) ** (252 / 5) - 1)
    assert result.avg_daily_return == pytest.approx(0.006)
    assert result.skew == pytest.approx(returns.skew())
    assert result.kurt == pytest.approx(returns.kurt())
    assert result.max_gain == pytest.approx(0.03)
    assert result.best_day == returns.index[2]
    assert result.max_loss == pytest.approx(-0.02)
    assert result.worst_day == returns.index[1]
    assert result.daily_win_rate == pytest.approx(0.6)
    assert result.ann_vol == pytest.approx(returns.std() * np.sqrt(252))

    assert result.max_drawdown == pytest.approx(-0.10)
    assert result.max_dd_days == 3  # indices 1,2,3 form one contiguous underwater run
    assert result.max_dd_recovery_days == 2  # trough at idx 2, recovers (dd~=0) at idx 4

    var_95 = np.percentile(returns, 5)
    assert result.var_95pct == pytest.approx(var_95)
    assert result.cvar == pytest.approx(returns[returns <= var_95].mean())

    downside = returns[returns < 0]
    ann_downside_vol = downside.std() * np.sqrt(252)
    assert result.sharpe_ratio == pytest.approx(result.avg_daily_return * 252 / result.ann_vol)
    assert result.sortino_ratio == pytest.approx(result.ann_return / ann_downside_vol)
    assert result.calmar_ratio == pytest.approx(result.ann_return / 0.10)

def test_compute_series_metrics_sharpe_nan_on_zero_vol_no_warning(assert_no_warnings):
    returns = pd.Series([0.5] * 10)  # 0.5 is exactly representable in binary float -> exact zero std, unlike e.g. 0.001
    drawdown = pd.Series([0.0] * 10)

    with assert_no_warnings():
        result = compute_series_metrics(returns, drawdown)

    assert np.isnan(result.sharpe_ratio)

def test_compute_series_metrics_calmar_nan_on_never_underwater_no_warning(assert_no_warnings):
    returns = pd.Series([0.01, 0.02, -0.01, 0.005])
    drawdown = pd.Series([0.0, 0.0, 0.0, 0.0])

    with assert_no_warnings():
        result = compute_series_metrics(returns, drawdown)

    assert np.isnan(result.calmar_ratio)
    assert result.max_dd_days == 0
    assert result.max_dd_recovery_days == 0
    assert result.max_drawdown == 0.0

# ---- compute_trade_metrics --------------------------------------------------

def test_compute_trade_metrics_hand_derived():
    trade_count = pd.Series([0, 1, 0, 1, 1])
    trade_wins = pd.Series([0, 1, 0, 0, 1])

    result = compute_trade_metrics(trade_count, trade_wins, cum_return=0.05)

    assert result.total_trades == 3
    assert result.win_rate == pytest.approx(2 / 3)
    assert result.trades_per_day == pytest.approx(3 / 5)
    assert result.return_per_trade == pytest.approx(0.05 / 3)

def test_compute_trade_metrics_zero_trades_returns_nan_no_warning(assert_no_warnings):
    trade_count = pd.Series([0, 0, 0])
    trade_wins = pd.Series([0, 0, 0])

    with assert_no_warnings():
        result = compute_trade_metrics(trade_count, trade_wins, cum_return=0.0)

    assert result.total_trades == 0
    assert np.isnan(result.win_rate)
    assert np.isnan(result.return_per_trade)

# ---- compute_relative_metrics -----------------------------------------------

def test_compute_relative_metrics_hand_derived():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    reference = pd.Series([0.005, -0.01, 0.02, -0.005, 0.015])

    result = compute_relative_metrics(returns, reference)

    beta = returns.cov(reference) / reference.var()
    alpha = (returns.mean() - beta * reference.mean()) * 252
    correlation = returns.cov(reference) / (returns.std() * reference.std())

    assert result.beta == pytest.approx(beta)
    assert result.alpha == pytest.approx(alpha)
    assert result.correlation == pytest.approx(correlation)
    assert result.r_squared == pytest.approx(correlation ** 2)

    tracking_error = (returns - reference).std() * np.sqrt(252)
    assert result.information_ratio == pytest.approx((returns.mean() - reference.mean()) * 252 / tracking_error)

    residual = returns - (reference * beta + alpha / 252)
    assert result.idiosyncratic_risk == pytest.approx(residual.std() * np.sqrt(252))

def test_pandas_corr_warns_on_zero_variance_input_confirming_the_regression_is_real():
    # sanity-check that the bug compute_relative_metrics works around is genuine: pandas'
    # built-in .corr() throws an unguarded RuntimeWarning (via its internal np.corrcoef call)
    # on exactly this input shape.
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    reference = pd.Series([0.01] * 5)

    with pytest.warns(RuntimeWarning):
        returns.corr(reference)

def test_compute_relative_metrics_zero_variance_reference_no_runtimewarning(assert_no_warnings):
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    reference = pd.Series([0.01] * 5)  # constant -> zero variance

    with assert_no_warnings():
        result = compute_relative_metrics(returns, reference)

    assert np.isnan(result.beta)
    assert np.isnan(result.correlation)
    assert np.isnan(result.r_squared)

def test_compute_relative_metrics_up_down_market_capture_hand_derived():
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    reference = pd.Series([0.01, -0.005, 0.02, -0.01, 0.005])

    result = compute_relative_metrics(returns, reference)

    up_mask = reference > 0
    down_mask = reference < 0

    assert result.up_market_capture == pytest.approx(returns[up_mask].mean() / reference[up_mask].mean())
    assert result.down_market_capture == pytest.approx(returns[down_mask].mean() / reference[down_mask].mean())

def test_compute_relative_metrics_tail_dependency_hand_derived():
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0, 0.01, 50))
    reference = pd.Series(rng.normal(0, 0.01, 50))

    result = compute_relative_metrics(returns, reference)

    ref_lower, ref_upper = reference <= reference.quantile(0.10), reference >= reference.quantile(0.90)
    ret_lower_q, ret_upper_q = returns.quantile(0.10), returns.quantile(0.90)

    expected_lower = ((returns <= ret_lower_q) & ref_lower).sum() / ref_lower.sum()
    expected_upper = ((returns >= ret_upper_q) & ref_upper).sum() / ref_upper.sum()

    assert result.lower_tail_dependency == pytest.approx(expected_lower)
    assert result.upper_tail_dependency == pytest.approx(expected_upper)

# ---- format_value -------------------------------------------------------

def test_format_value_none_and_nan_render_dash():
    assert format_value(None) == "-"
    assert format_value(float("nan")) == "-"

def test_format_value_pct_formatting():
    assert format_value(0.1234, pct=True) == "12.34%"

def test_format_value_int_comma_formatting():
    assert format_value(1234567) == "1,234,567"

def test_format_value_float_comma_formatting():
    assert format_value(1234567.891) == "1,234,567.89"

def test_format_value_suffix_appended():
    assert format_value(5, suffix=" Days") == "5 Days"

def test_format_value_fallback_str_for_other_types():
    ts = pd.Timestamp("2024-01-02")
    assert format_value(ts) == str(ts)

# ---- dataclass_rows / merge_groups / render_sections -------------------------

def test_dataclass_rows_groups_by_declared_section_order():
    strat = compute_series_metrics(pd.Series([0.01, -0.01, 0.02]), pd.Series([0.0, -0.01, 0.0]))
    groups = dataclass_rows([strat], SeriesMetrics)

    assert [title for title, _ in groups] == [None, "Return Profile", "Trades", "Drawdowns", "Risk", "Ratios"]

def test_dataclass_rows_none_instance_renders_dash():
    groups = dataclass_rows([None], TradeMetrics, default_section="Trades")
    _, values = groups[0][1][0]

    assert values == ["-"]

def test_merge_groups_combines_same_titled_sections_from_different_calls():
    a = [("Risk", [("A", ["1"])])]
    b = [("Risk", [("B", ["2"])]), ("Other", [("C", ["3"])])]

    merged = merge_groups(a, b)

    assert [title for title, _ in merged] == ["Risk", "Other"]
    assert merged[0][1] == [("A", ["1"]), ("B", ["2"])]

def test_merge_groups_combines_ratios_section_across_series_and_relative_metrics():
    returns = pd.Series([0.01, -0.01, 0.02, 0.01])
    reference = pd.Series([0.005, -0.005, 0.01, 0.005])

    series = compute_series_metrics(returns, pd.Series([0.0, -0.01, 0.0, 0.0]))
    relative = compute_relative_metrics(returns, reference)

    groups = merge_groups(
        dataclass_rows([series], SeriesMetrics),
        dataclass_rows([relative], RelativeMetrics, default_section="Relative"),
    )

    ratios_section = next(rows for title, rows in groups if title == "Ratios")
    labels = [label for label, _ in ratios_section]

    assert {"Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Information Ratio"} <= set(labels)

def test_render_sections_produces_dataframe_to_string_output():
    groups = [(None, [("Metric A", ["1.00", "2.00"])])]
    output = render_sections(["Col1", "Col2"], groups)

    assert "Col1" in output and "Col2" in output
    assert "Metric A" in output

def test_render_sections_show_header_false_omits_column_names():
    groups = [(None, [("Metric A", ["1.00"])])]

    with_header = render_sections(["Col1"], groups, show_header=True)
    without_header = render_sections(["Col1"], groups, show_header=False)

    assert "Col1" in with_header
    assert "Col1" not in without_header
