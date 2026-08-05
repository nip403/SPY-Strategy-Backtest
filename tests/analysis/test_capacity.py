from __future__ import annotations

import io
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pytest

from backtest_engine.analysis.capacity import CapacityEstimator

@pytest.fixture
def estimator(portfolio_factory) -> CapacityEstimator:
    # CyclingStrategy is a blind state machine on zero-drift synthetic data, so its
    # exposure-weighted baseline return can legitimately land non-positive for some random seeds
    # (a real, expected outcome - not a bug). Skip rather than fail: the degenerate/terminate-early
    # path itself is covered separately by test_flat_market_warns_and_terminates_early.
    portfolio = portfolio_factory()
    est = CapacityEstimator(portfolio)

    if not hasattr(est, "window"):
        pytest.skip("non-positive baseline for this random seed - degenerate case covered separately")

    return est

# ---- construction / decay curve ---------------------------------------------

def test_decay_curve_zero_delay_is_total_gross_return(portfolio_factory):
    # decay_curve is a raw total (sum of gross_ret contributions), not normalised by exposure held
    portfolio = portfolio_factory()
    est = CapacityEstimator(portfolio)

    if not hasattr(est, "window"):
        pytest.skip("non-positive baseline for this random seed - degenerate case covered separately")

    df = portfolio.df
    exposure = df["position"].shift(1).fillna(0)
    expected = float((exposure * df["ret"]).sum())

    assert est.decay_curve[0] == pytest.approx(expected)

def test_decay_ratio_zero_delay_is_unity(estimator):
    assert estimator.decay_ratio[0] == pytest.approx(1.0)

def test_delays_and_curves_span_max_delay(portfolio_factory):
    portfolio = portfolio_factory()
    est = CapacityEstimator(portfolio, max_delay=10)

    if not hasattr(est, "decay_ratio"):
        pytest.skip("non-positive baseline for this random seed - degenerate case covered separately")

    assert list(est.delays) == list(range(11))
    assert len(est.decay_curve) == 11
    assert len(est.decay_ratio) == 11
    assert len(est.sharpe_curve) == 11

def test_window_bounded_by_max_delay(estimator):
    assert 0 <= estimator.window <= estimator.max_delay

def test_decay_threshold_of_zero_is_accepted(portfolio_factory):
    # bound is 0 <= decay_threshold < 1 (inclusive of 0), not the stricter open interval
    portfolio = portfolio_factory()
    est = CapacityEstimator(portfolio, decay_threshold=0)

    assert est.decay_threshold == 0

def test_invalid_decay_threshold_raises(portfolio_factory):
    portfolio = portfolio_factory()

    with pytest.raises(AssertionError):
        CapacityEstimator(portfolio, decay_threshold=1)

    with pytest.raises(AssertionError):
        CapacityEstimator(portfolio, decay_threshold=-0.1)

def test_entry_delay_exceeding_every_run_length_is_zero(estimator):
    # default portfolio_factory uses CyclingStrategy(period=47); max_delay=60 exceeds every run's
    # length, so no exposure survives at all - decay_curve is a raw total, so zero exposure
    # contributing means a total of exactly 0, not NaN. A blanket shift (rather than entry-only
    # delay) would never hit this at all, since it would just relocate the same trades later
    # instead of erasing them.
    assert estimator.decay_curve[-1] == 0.0
    assert estimator.decay_ratio[-1] == 0.0

# ---- capacity bound formula --------------------------------------------------

def test_capacity_bound_matches_formula(estimator):
    expected = estimator.window * estimator.avg_volume_per_min * estimator.participation_rate * estimator.avg_price
    assert estimator.capacity_bound == pytest.approx(expected)

def test_participation_rate_matches_realized_peak(portfolio_factory):
    # reverse-engineered from the portfolio's own realised fills, not a supplied assumption
    portfolio = portfolio_factory()
    est = CapacityEstimator(portfolio)

    if not hasattr(est, "window"):
        pytest.skip("non-positive baseline for this random seed - degenerate case covered separately")

    df = portfolio.df
    delta_position = df["position"].diff().abs().fillna(0)
    denom = df["close"] * df["volume"]
    expected = float((delta_position * portfolio.aum / denom).replace([np.inf, -np.inf], np.nan).fillna(0).max())

    assert est.participation_rate == pytest.approx(expected)

def test_participation_rate_is_not_a_constructor_input(portfolio_factory):
    portfolio = portfolio_factory()

    with pytest.raises(TypeError):
        CapacityEstimator(portfolio, participation_rate=0.05)

# ---- execution/cost-modelled Sharpe curve ------------------------------------

def test_modelled_sharpe_zero_delay_matches_hand_derived_population_std_sharpe(portfolio_factory):
    # delay=0 re-feeds the exact same position through the same execution/cost model instances
    # and a fresh cache, reproducing the same daily returns Portfolio.sharpe uses - but NOT the
    # same Sharpe value: sharpe() computes std via plain numpy (ddof=0, population), while
    # Portfolio.sharpe uses pandas' Series.std() (ddof=1, sample). Hand-derived here against the
    # population-std formula this code actually uses, not against Portfolio.sharpe directly.
    portfolio = portfolio_factory()
    est = CapacityEstimator(portfolio)

    if not hasattr(est, "window"):
        pytest.skip("non-positive baseline for this random seed - degenerate case covered separately")

    dates = portfolio.df.index.date
    eod_indices = np.append(np.flatnonzero(dates[:-1] != dates[1:]), len(dates) - 1)

    equity = np.cumprod(np.nan_to_num(1 + portfolio.df["net_ret"].to_numpy()))
    eod = equity[eod_indices]
    daily_ret = np.empty_like(eod)
    daily_ret[0] = eod[0] - 1
    daily_ret[1:] = np.diff(eod) / eod[:-1]

    expected = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252))  # ddof=0

    assert est.sharpe_curve[0] == pytest.approx(expected)

def test_modelled_sharpe_curve_matches_delays_shape(estimator):
    assert estimator.sharpe_curve.shape == estimator.delays.shape

# ---- degenerate / flat market -------------------------------------------

def test_flat_market_warns_and_terminates_early(portfolio_factory):
    # ret==0 everywhere makes decay_curve[0]==0 - with no positive baseline edge to measure decay
    # against, __init__ warns (exact wording) and returns early rather than computing a
    # misleading decay_ratio/window/capacity_bound.
    portfolio = portfolio_factory(ohlcv_kwargs={"minute_vol": 0.0})

    with pytest.warns(UserWarning, match="non-positive"):
        est = CapacityEstimator(portfolio)

    assert not hasattr(est, "window")
    assert not hasattr(est, "decay_ratio")
    assert not hasattr(est, "capacity_bound")

# ---- plotting -----------------------------------------------------------

def test_plot_shows_and_closes_figure(estimator, captured_figures):
    figs_before = len(plt.get_fignums())
    estimator.plot()

    assert len(captured_figures) == 2
    assert len(plt.get_fignums()) == figs_before

def test_plot_savepath_saves_and_closes_figure(estimator, captured_figures, tmp_path):
    figs_before = len(plt.get_fignums())
    estimator.plot(savepath=tmp_path)

    assert len(captured_figures) == 2
    assert len(plt.get_fignums()) == figs_before

    created = list(tmp_path.iterdir())
    assert len(created) == 1
    assert created[0].name.startswith("CapacityEstimator_")
    assert (created[0] / "alpha_decay_and_sharpe.png").exists()
    assert (created[0] / "economic_capacity.png").exists()

# ---- string / report ----------------------------------------------------

def test_str_contains_expected_labels(estimator):
    text = str(estimator)

    for label in ["Alpha Decay Window", "Participation Rate", "Avg. Volume / Min", "Est. Fill Capacity", "Sharpe @ Base", "Sharpe @ Window"]:
        assert label in text

def test_report_plots_and_prints(estimator, captured_figures):
    figs_before = len(plt.get_fignums())
    buf = io.StringIO()

    with redirect_stdout(buf):
        estimator.report()

    assert len(captured_figures) == 2
    assert len(plt.get_fignums()) == figs_before
    assert "Est. Fill Capacity" in buf.getvalue()
