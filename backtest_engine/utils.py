from __future__ import annotations

from scipy.optimize import newton
import pandas as pd
import numpy as np
import random
from datetime import date
import warnings
from typing import Callable, Optional, overload

def round_date(date_index: pd.DataFrame, dt: date) -> date:
    """
    Round a date to the nearest available trading date in an index.

    date_index : pd.DataFrame
        Datetime-indexed data containing the available trading dates.
    dt : date
        Target date to round.

    Returns date
        Nearest available trading date.
    """
    
    dates = pd.Index(date_index.date).unique()
    dt = pd.to_datetime(dt)
    dt = dt.tz_localize(date_index.tz).date() if dt.tz is None else dt.tz_convert(date_index.tz).date()
    
    pos = dates.searchsorted(dt)
    
    if not pos:
        return dates[0]
    
    if pos == len(dates):
        return dates[-1]
    
    before = dates[pos - 1]
    after = dates[pos]

    return before if (dt - before) <= (after - dt) else after

def compute_drawdown(equity: pd.Series) -> pd.Series:
    """
    Compute a running drawdown series from an equity curve.

    equity : pd.Series
        Cumulative equity/growth-factor series.

    Returns pd.Series
        Drawdown (<=0) at each point, relative to the running peak.
    """

    peak = np.maximum.accumulate(equity.to_numpy())

    return pd.Series((equity.to_numpy() - peak) / peak, index=equity.index)

def trade_stats(position: pd.Series, net_ret: pd.Series) -> pd.DataFrame:
    """
    Segment a position series into discrete trades and flag trade closes and wins.
    Trades are defined as runs of nonzero constant sign(position); sign flips signal closes.

    Note net_ret[t] = position[t-1] * ret[t] - cost(position[t] - position[t-1]).

    position : pd.Series
        Leverage series.
    net_ret : pd.Series
        Aligned net returns.

    Returns pd.DataFrame
        Index-aligned to inputs; columns:
            trade_count: 1 on the bar a trade closes, else 0 (sum/groupby for counts per period)
            trade_wins: 1 on the bar a *winning* trade closes, else 0
    """

    side = np.sign(position)
    prev_side = side.shift(fill_value=0)

    # id new trades
    new_trade = (side != prev_side) & (side != 0)
    entry_id = new_trade.cumsum().where(side != 0)

    # the trade a bar's net_ret is actually attributable to
    trade_id = entry_id.shift(1).where(prev_side != 0, entry_id)
    is_close = trade_id.notna() & (trade_id != trade_id.shift(-1))

    trade_returns = (1 + net_ret).groupby(trade_id).prod() - 1
    is_win = is_close & (trade_id.map(trade_returns) > 0)

    return pd.DataFrame({
        "trade_count": is_close.astype(int),
        "trade_wins": is_win.astype(int),
    }, index=position.index)
    
def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom else float("nan")

def generate_toy_returns(
    periods: int,
    *,
    mean: float |np.ndarray = 0,
    std: float | np.ndarray = 1,
    distribution: Optional[Callable] = None,
    random_seed: Optional[int] = 42,
    **kwargs,
) -> np.ndarray:
    """
    Generate synthetic returns from a specified distribution.
    Supports vectorised inputs to generate multiple return paths. 
    Uses global state management to support external distribution callables.
    Note: ndarray shapes must match if not a scalar.

    periods : int
        Number of return observations to generate.
    mean : float or np.ndarray = 0
        Target period minute mean(s).
    std : float or np.ndarray = 1
        Target period minute standard deviation(s).
    distribution : Callable = None
        Random distribution function that accepts loc, scale, and size arguments (as both float or np.ndarray), and returns an np.ndarray.
        Ideally an existing np/scipy distribution function or wrapper for compatibility with random_seed.
        Should rely on global np.random or builtin random modules to be deterministic, and should be managed by the user otherwise.
        Defaults to numpy normal distribution.
    random_seed : int = 42
        Random seed used for reproducible results. Set to None for non-deterministic output.
    **kwargs : dict
        Additional kw args to pass onto the distribution callable, if compatible. 

    Returns np.ndarray
        Array of simulated returns.
    """
    
    mean_arr, std_arr = np.broadcast_arrays(mean, std)
    
    if random_seed is not None:
        try:
            seed = int(random_seed)
        except: 
            seed = 42
            
        state = [np.random.get_state(), random.getstate()]
        
        random.seed(seed)
        np.random.seed(seed)

    try:
        return np.asarray((distribution or np.random.normal)(
            loc=mean_arr,
            scale=std_arr,
            size=(periods,) + mean_arr.shape, # support multidimensional ret/vol matrices
            **kwargs,
        ))

    finally: # reset initial state
        if random_seed is not None:
            np.random.set_state(state[0])
            random.setstate(state[1])
        
@overload
def generate_toy_equity(*, expected_return: float | np.ndarray = 0, sharpe: None = None, volatility: float | np.ndarray = 0.01, beta: float | np.ndarray = 0, portfolio: Portfolio, benchmark: Optional[pd.Series] = None, distribution: Optional[Callable] = None, random_seed: Optional[int] = 42) -> tuple[pd.Series | pd.DataFrame, np.ndarray]: ...

@overload
def generate_toy_equity(*, expected_return: None = None, sharpe: float | np.ndarray = 0, volatility: float | np.ndarray = 0.01, beta: float | np.ndarray = 0, portfolio: Portfolio, benchmark: Optional[pd.Series] = None, distribution: Optional[Callable] = None, random_seed: Optional[int] = 42) -> tuple[pd.Series | pd.DataFrame, np.ndarray]: ... 

def generate_toy_equity(
    *, 
    expected_return: Optional[float | np.ndarray] = None,
    sharpe: Optional[float | np.ndarray] = None,
    volatility: float | np.ndarray = 0.01,
    beta: float | np.ndarray = 0,
    portfolio: Portfolio,
    benchmark: Optional[pd.Series] = None,
    distribution: Optional[Callable] = None,
    random_seed: Optional[int] = 42,
    **kwargs,
    ) -> tuple[pd.Series | pd.DataFrame, np.ndarray]:
    """
    Generate synthetic intraday equity curve(s) based on target annual statistics, aligned to portfolio df index.
    Supports vectorised inputs to generate multiple return paths. 
    Note: np.ndarray shapes must match if not a scalar. Scalars are broadcasted to fill all potential inputs.
    
    Solves the CAPM model to create market-correlated returns:
    R_toy = alpha + beta * R_mkt + err

    expected_return : float = 0
        Target annualised expected return(s). Geometrically scaled to intraday.
    sharpe : float = 1
        Target annualised sharpe(s). Used mutually exclusively with expected_return (overrides if needed).
    volatility : float = 0.01
        Target annualised volatility/standard deviation(s).
    beta : float = 0
        Target beta(s) of the synthetic equity to benchmark returns, in expectation
        Ignored if benchmark is None.
    portfolio : Portfolio
        Portfolio used to determine the intraday index and starting AUM.
    benchmark : pd.Series = None
        Minute-indexed benchmark price series (e.g. SPY) used for calculating beta exposure. 
        If None, returns are purely random.
    distribution : Callable = None
        Random distribution function that accepts loc, scale, and size arguments, and returns an np.ndarray.
        Defaults to numpy normal distribution.
        See docstring in generate_toy_returns for more details.
    random_seed : int = 42
        Random seed used for reproducible results. Set to None for random output.
    **kwargs: dict
        Additional kw args to pass onto the distribution callable, if compatible. 

    Returns tuple[pd.Series | pd.DataFrame, np.ndarray]
        1. Synthetic equity curve aligned to the portfolio intraday index.
        2. 1d boolean np.ndarray mask to filter initial valid parameter sets
    """
    
    if sharpe is not None:
        expected_return = np.asarray(sharpe) * np.asarray(volatility)
    elif expected_return is not None:
        expected_return = np.asarray(expected_return)
    else:
        raise ValueError("Invalid params.")
    
    try:
        expected_return, volatility, beta = np.broadcast_arrays(expected_return, np.asarray(volatility), np.asarray(beta))
    except:
        raise ValueError("Shapes of expected_return/sharpe, volatility, and beta must be perfectly broadcastable (or scalar).")

    is_scalar = expected_return.ndim == 0 # return object check
    
    expected_return = np.atleast_1d(expected_return)
    volatility = np.atleast_1d(volatility)
    beta = np.atleast_1d(beta)

    periods_per_year = 252 * 390

    if benchmark is None or np.all(beta == 0):
        returns_matrix = generate_toy_returns(
            len(portfolio.df),
            mean=(1 + expected_return) ** (1 / periods_per_year) - 1,
            std=volatility / np.sqrt(periods_per_year),
            distribution=distribution,
            random_seed=random_seed,
            **kwargs,
        )
        
        equity = (1 + returns_matrix).cumprod(axis=0) * portfolio.aum

        labels = [
            f"{f"S:{r / v if v else 0:.1f}" if sharpe is not None else f"R:{r:.3f}"}|σ:{v:.3f}|β:{b:.1f}"
            for r, v, b in zip(expected_return, volatility, beta)
        ]

        result = pd.Series(equity[:, 0], index=portfolio.df.index, name=labels[0]) if is_scalar else pd.DataFrame(equity, index=portfolio.df.index, columns=labels)

        return result, valid
    
    benchmark = benchmark.reindex(portfolio.df.index).pct_change().fillna(0).to_numpy()

    # solve idiosyncratic vol so beta exposure + noise combine to the target annual vol
    # var(r_toy) = var(beta * r_mkt) + var(err) = beta^2 * var(r_mkt) + var(err)
    idiosyncratic_variance = volatility ** 2 - beta ** 2 * benchmark.var() * periods_per_year

    valid = idiosyncratic_variance >= 0
    
    if np.any(~valid):
        bad_vols = volatility[~valid]
        bad_betas = beta[~valid]
        min_vols = np.sqrt(bad_betas ** 2 * benchmark.var() * periods_per_year)
        
        pairs = [
            f"(beta={b:.2f}: target_vol={v:.4f}, min_implied_vol={m:.4f})" 
            for b, v, m in zip(bad_betas, bad_vols, min_vols)
        ]
        
        warnings.warn(f"Certain target volatilities are impossible given their specified beta exposures and will be dropped.\nInvalid pairs: {", ".join(pairs)}.")
        
        if not np.any(valid):
            raise ValueError("All given parameter combinations are impossible.")
        
    expected_return = expected_return[valid]
    volatility = volatility[valid]
    beta = beta[valid]
    idiosyncratic_variance = idiosyncratic_variance[valid]

    def solve_alpha() -> np.ndarray:
        """
        solve residual expected return x against benchmark s.t. pi^N_{t=1}(1 + x + beta * R_mkt_t) = 1 + R_toy
        Then, f(x) = ln(product) - ln(1 + R_toy) = 0, and root find.
        """
        
        target = np.log1p(expected_return) * len(benchmark) / periods_per_year
        base = 1 + beta * benchmark[:, None]
        
        # start with arithmetic return approximation
        x0 = (expected_return - beta * benchmark.mean() * periods_per_year) / periods_per_year
        
        def f(mu: np.ndarray) -> np.ndarray:
            return np.log(mu + base).sum(axis=0) - target
        
        def df(mu: np.ndarray) -> np.ndarray: # derivative
            return (1 / (base + mu)).sum(axis=0)
        
        return newton(
            func=f,
            x0=x0,
            fprime=df,
            tol=1e-14,
        )

    idiosyncratic_returns = generate_toy_returns(
        len(portfolio.df),
        mean = solve_alpha(),
        std = np.sqrt(idiosyncratic_variance / periods_per_year),
        distribution=distribution,
        random_seed=random_seed,
        **kwargs,
    )
    
    equity = (1 + beta * benchmark[:, None] + idiosyncratic_returns).cumprod(axis=0) * portfolio.aum

    labels = [
        f"{f"S:{r / v if v else 0:.1f}" if sharpe is not None else f"R:{r:.3f}"}|σ:{v:.3f}|β:{b:.1f}"
        for r, v, b in zip(expected_return, volatility, beta)
    ]
   
    return pd.Series(equity[:, 0], index=portfolio.df.index, name=labels[0]) if is_scalar else pd.DataFrame(equity, index=portfolio.df.index, columns=labels), valid
