from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtest_engine.utils import (
    round_date, compute_drawdown, trade_stats, _safe_div,
    generate_toy_returns, generate_toy_equity,
)

# ---- round_date --------------------------------------------------------

@pytest.fixture
def business_day_index() -> pd.DatetimeIndex:
    # Tue Jan 2 -> Wed Jan 10 2024, business days only: 2,3,4,5,8,9,10
    return pd.date_range("2024-01-02", "2024-01-10", freq="B", tz="America/New_York")

def test_round_date_exact_match_returns_same_date(business_day_index):
    assert round_date(business_day_index, date(2024, 1, 4)) == date(2024, 1, 4)

def test_round_date_rounds_to_nearer_neighbour(business_day_index):
    # Sat Jan 6 sits 1 day after Fri Jan 5, 2 days before Mon Jan 8 -> nearer is Jan 5
    assert round_date(business_day_index, date(2024, 1, 6)) == date(2024, 1, 5)

def test_round_date_clamps_before_range(business_day_index):
    assert round_date(business_day_index, date(2023, 12, 25)) == date(2024, 1, 2)

def test_round_date_clamps_after_range(business_day_index):
    assert round_date(business_day_index, date(2024, 2, 1)) == date(2024, 1, 10)

# ---- compute_drawdown --------------------------------------------------

def test_compute_drawdown_hand_derived_values():
    equity = pd.Series([100.0, 110.0, 105.0, 120.0, 90.0])
    dd = compute_drawdown(equity)

    expected = [0.0, 0.0, (105 - 110) / 110, 0.0, (90 - 120) / 120]
    np.testing.assert_allclose(dd.to_numpy(), expected)

def test_compute_drawdown_monotonic_equity_is_all_zero():
    equity = pd.Series([100.0, 101.0, 105.0, 110.0])
    dd = compute_drawdown(equity)

    np.testing.assert_allclose(dd.to_numpy(), [0.0, 0.0, 0.0, 0.0])

# ---- trade_stats --------------------------------------------------------

def test_trade_stats_flags_close_and_win():
    position = pd.Series([0.0, 1.0, 1.0, 0.0])
    net_ret = pd.Series([0.0, 0.01, 0.02, 0.0])

    result = trade_stats(position, net_ret)

    assert result["trade_count"].tolist() == [0, 0, 0, 1]
    assert result["trade_wins"].tolist() == [0, 0, 0, 1]

def test_trade_stats_flags_close_and_loss():
    position = pd.Series([0.0, 1.0, 1.0, 0.0])
    net_ret = pd.Series([0.0, -0.01, -0.02, 0.0])

    result = trade_stats(position, net_ret)

    assert result["trade_count"].tolist() == [0, 0, 0, 1]
    assert result["trade_wins"].tolist() == [0, 0, 0, 0]

def test_trade_stats_flip_closes_old_and_opens_new():
    position = pd.Series([1.0, 1.0, -1.0, -1.0])
    net_ret = pd.Series([0.0, 0.01, -0.02, 0.01])

    result = trade_stats(position, net_ret)

    # long leg (bars 0-2, net_ret 0/0.01/-0.02) compounds to a loss, closed on the flip bar (idx 2)
    # short leg (bar 3, net_ret 0.01) is a win, closed at series end (idx 3)
    assert result["trade_count"].tolist() == [0, 0, 1, 1]
    assert result["trade_wins"].tolist() == [0, 0, 0, 1]

def test_trade_stats_all_flat_has_no_trades():
    position = pd.Series([0.0, 0.0, 0.0])
    net_ret = pd.Series([0.0, 0.0, 0.0])

    result = trade_stats(position, net_ret)

    assert result["trade_count"].sum() == 0
    assert result["trade_wins"].sum() == 0

# ---- _safe_div --------------------------------------------------------

def test_safe_div_zero_denominator_returns_nan():
    assert np.isnan(_safe_div(5.0, 0.0))

def test_safe_div_nonzero_denominator_returns_value():
    assert _safe_div(10.0, 4.0) == 2.5

# ---- generate_toy_returns --------------------------------------------------

def test_generate_toy_returns_deterministic_given_seed():
    a = generate_toy_returns(100, mean=0.0001, std=0.01, random_seed=42)
    b = generate_toy_returns(100, mean=0.0001, std=0.01, random_seed=42)

    np.testing.assert_array_equal(a, b)

def test_generate_toy_returns_restores_global_random_state_after_call():
    before = np.random.get_state()
    generate_toy_returns(50, mean=0.0, std=0.01, random_seed=123)
    after = np.random.get_state()

    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]

def test_generate_toy_returns_non_numeric_seed_falls_back_to_42():
    # int(random_seed) raises for a non-numeric seed; falls back to the hardcoded default of 42
    a = generate_toy_returns(50, mean=0.0, std=0.01, random_seed="not-a-number")
    b = generate_toy_returns(50, mean=0.0, std=0.01, random_seed=42)

    np.testing.assert_array_equal(a, b)

# ---- generate_toy_equity --------------------------------------------------

def test_generate_toy_equity_raises_without_return_or_sharpe(portfolio_factory):
    portfolio = portfolio_factory()

    with pytest.raises(ValueError):
        generate_toy_equity(portfolio=portfolio)

def test_generate_toy_equity_no_benchmark_scalar_returns_equity_valid_tuple(portfolio_factory):
    # regression: this branch used to return a bare equity object with no validity mask,
    # despite the docstring/type hint promising an (equity, valid) tuple like the with-benchmark
    # branch always gave - fixed to match the documented contract.
    portfolio = portfolio_factory()
    equity, valid = generate_toy_equity(portfolio=portfolio, sharpe=1.0, volatility=0.15, random_seed=1)

    assert isinstance(equity, pd.Series)
    assert equity.index.equals(portfolio.df.index)
    np.testing.assert_array_equal(valid, [True])

def test_generate_toy_equity_no_benchmark_vectorised_output_shape_and_labels(portfolio_factory):
    # regression: this branch used to build the DataFrame with no explicit `columns=`, giving
    # plain integer columns (0,1,2,...) instead of the "S:.../σ:.../β:..." labels the
    # with-benchmark branch always produced - fixed to label consistently.
    portfolio = portfolio_factory()
    equity, valid = generate_toy_equity(portfolio=portfolio, sharpe=np.array([0.5, 1.0, 1.5]), volatility=0.15, random_seed=1)

    assert isinstance(equity, pd.DataFrame)
    assert equity.shape == (len(portfolio.df), 3)
    assert list(equity.columns) == ["S:0.5|σ:0.150|β:0.0", "S:1.0|σ:0.150|β:0.0", "S:1.5|σ:0.150|β:0.0"]
    np.testing.assert_array_equal(valid, [True, True, True])

def test_generate_toy_equity_with_benchmark_returns_equity_valid_tuple(portfolio_factory):
    portfolio = portfolio_factory()
    benchmark = portfolio.df["close"]
    equity, valid = generate_toy_equity(portfolio=portfolio, sharpe=1.0, volatility=0.15, beta=0.5, benchmark=benchmark, random_seed=1)

    assert isinstance(equity, pd.Series)
    assert equity.index.equals(portfolio.df.index)
    assert valid.all()

def test_generate_toy_equity_impossible_beta_vol_combo_warns_and_drops(portfolio_factory):
    portfolio = portfolio_factory()
    benchmark = portfolio.df["close"]

    # beta=5 needs huge idiosyncratic variance headroom at low target vol -> impossible; beta=0 always valid
    with pytest.warns(UserWarning):
        equity, valid = generate_toy_equity(
            portfolio=portfolio, sharpe=np.array([1.0, 1.0]), volatility=0.05,
            beta=np.array([5.0, 0.0]), benchmark=benchmark, random_seed=1,
        )

    assert valid.tolist() == [False, True]
    assert equity.shape[1] == 1

def test_generate_toy_equity_all_invalid_combos_raises_valueerror(portfolio_factory):
    portfolio = portfolio_factory()
    benchmark = portfolio.df["close"]

    with pytest.raises(ValueError):
        generate_toy_equity(
            portfolio=portfolio, sharpe=np.array([1.0]), volatility=0.001,
            beta=np.array([50.0]), benchmark=benchmark, random_seed=1,
        )

def test_generate_toy_equity_expected_return_kwarg_path(portfolio_factory):
    # expected_return= is mutually exclusive with sharpe= (used when sharpe is left None)
    portfolio = portfolio_factory()
    equity, valid = generate_toy_equity(portfolio=portfolio, expected_return=0.1, volatility=0.15, random_seed=1)

    assert isinstance(equity, pd.Series)
    assert valid.all()

def test_generate_toy_equity_non_broadcastable_shapes_raises_valueerror(portfolio_factory):
    # expected_return= (unlike sharpe=) skips the sharpe*volatility multiplication, so this
    # reaches np.broadcast_arrays directly rather than raising numpy's raw error a step earlier
    portfolio = portfolio_factory()

    with pytest.raises(ValueError):
        generate_toy_equity(portfolio=portfolio, expected_return=np.array([0.1, 0.15, 0.2]), volatility=np.array([0.1, 0.2]), random_seed=1)
