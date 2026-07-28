from __future__ import annotations

import warnings
from contextlib import contextmanager

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from backtest_engine import Portfolio
from backtest_engine.components.base import BacktestContext, StrategyComponent

# ---- no-GUI guard ----------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _no_gui_backend():
    """Force a non-interactive backend and no-op plt.show() for the whole session."""

    matplotlib.use("Agg", force=True)
    mp = pytest.MonkeyPatch()  # session-scoped: the function-scoped `monkeypatch` fixture can't be used here
    mp.setattr(plt, "show", lambda *a, **k: None)
    yield
    mp.undo()

# ---- randomized, independently-repeated test data ---------------------------

@pytest.fixture(params=[11, 47, 89, 131, 197], ids=lambda s: f"seed{s}")
def random_seed(request) -> int:
    """Fixed, diverse seeds: every test that consumes synthetic data (directly or via
    portfolio_factory) runs once per seed against a genuinely different random dataset, each an
    independent pass/fail. Fixed rather than freshly randomized per collection so a failing case
    is reproducible by its pytest id (e.g. test_foo[seed47])."""

    return request.param

# ---- synthetic market data --------------------------------------------------

@pytest.fixture
def synthetic_ohlcv(random_seed):
    """Factory: build a minute-level OHLCV DataFrame shaped like data._preprocess()'s output
    (tz-aware America/New_York index, columns open/high/low/close/volume/time, 390 bars/day).
    Defaults to the randomized per-run seed; pass seed= explicitly to pin an exact dataset.

    Note: Portfolio._preprocess's "sigma"/"std" need a 14-trading-day rolling warmup, so pass
    n_days >= 20 whenever a test needs live (non-NaN) signals from the real strategy classes.
    """

    def _make(
        n_days: int = 20,
        *,
        seed: int | None = None,
        start: str = "2024-01-02",
        start_price: float = 400.0,
        minute_vol: float = 0.0005,
        avg_volume: int = 500_000,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(random_seed if seed is None else seed)

        days = pd.bdate_range(start, periods=n_days)
        minute_offsets = pd.to_timedelta(np.arange(390), unit="m") + pd.Timedelta(hours=9, minutes=30)
        index = pd.DatetimeIndex(np.repeat(days.values, 390)) + np.tile(minute_offsets.values, n_days)
        index = index.tz_localize("America/New_York")

        n = len(index)
        log_rets = rng.normal(0, minute_vol, size=n)
        close = start_price * np.exp(np.cumsum(log_rets))
        open_ = np.roll(close, 1)
        open_[0] = start_price

        spread = np.abs(rng.normal(0, minute_vol / 2, size=n))
        high = np.maximum(open_, close) * (1 + spread)
        low = np.minimum(open_, close) * (1 - spread)

        volume = rng.uniform(avg_volume / 390 * 0.5, avg_volume / 390 * 1.5, size=n)

        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)
        df["time"] = df.index.time

        return df

    return _make

# ---- deterministic strategy test double -------------------------------------

class CyclingStrategy(StrategyComponent):
    """Deterministic flat/long/flat/short state cycle, ignoring price data entirely.

    BaseStrategy's real noise-area signal rarely fires on tame synthetic random-walk data, so
    this exists wherever a test needs a *reliable*, known position series rather than exercising
    the actual noise-area logic (that belongs in components/test_strategy.py).
    """

    def __init__(self, period: int = 47, leverage: float = 1.0) -> None:
        self.period = period
        self.leverage = leverage

    def set(self, df: pd.DataFrame, ctx: BacktestContext) -> pd.DataFrame:
        phase = (np.arange(len(df)) // self.period) % 4  # 0 flat, 1 long, 2 flat, 3 short
        position = np.select([phase == 1, phase == 3], [self.leverage, -self.leverage], default=0.0)

        if not ctx.long_perm:
            position = np.where(position > 0, 0.0, position)
        if not ctx.short_perm:
            position = np.where(position < 0, 0.0, position)

        df["position"] = position

        return df

@pytest.fixture
def cycling_strategy() -> CyclingStrategy:
    return CyclingStrategy()

# ---- portfolio factory -------------------------------------------------

@pytest.fixture
def portfolio_factory(synthetic_ohlcv):
    """Callable returning a configured, already-backtested Portfolio on synthetic data.
    Defaults to CyclingStrategy (see above) rather than BaseStrategy, for the same reason.
    seed=None (default) defers to synthetic_ohlcv's own randomized per-run seed."""

    def _make(
        *,
        n_days: int = 20,
        seed: int | None = None,
        aum: float = 100_000,
        target_vol: float = 0.02,
        long_permissions: bool = True,
        short_permissions: bool = True,
        strategy=None,
        execution=None,
        cost_model=None,
        ohlcv_kwargs: dict | None = None,
    ) -> Portfolio:
        df = synthetic_ohlcv(n_days=n_days, seed=seed, **(ohlcv_kwargs or {}))

        return Portfolio(
            df,
            aum=aum,
            target_vol=target_vol,
            long_permissions=long_permissions,
            short_permissions=short_permissions,
            strategy_model=strategy or CyclingStrategy(),
            execution_model=execution,
            cost_model=cost_model,
        )

    return _make

# ---- cwd isolation for data.py cache tests ----------------------------------

@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch):
    """Isolate cache-file I/O in data.py (which uses a bare relative filename, not
    module-relative) into a throwaway directory."""

    monkeypatch.chdir(tmp_path)

    return tmp_path

# ---- misc --------------------------------------------------------------

@pytest.fixture
def fake_alpaca_config() -> dict:
    """Dummy credentials - never reads config.toml, which holds real live keys."""

    return {"key": "dummy-key", "secret": "dummy-secret"}

@pytest.fixture
def assert_no_warnings():
    """Context-manager fixture: fail if ANY warning (incl. an unguarded RuntimeWarning from e.g.
    pandas .corr()'s internal np.corrcoef) is emitted inside the `with` block."""

    @contextmanager
    def _ctx():
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            yield

    return _ctx
