from __future__ import annotations

from scipy.optimize import newton
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import random
from datetime import date, datetime
from pathlib import Path
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
    is_win = trade_id[is_close].map(trade_returns)
    
    return pd.DataFrame({
        "trade_count": is_close.astype(int),
        "trade_wins": is_win.astype(int),
    }, index=position.index)
    
def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom else float("nan")

def save_figures(figs: dict[str, plt.Figure], classname: str, savepath: str | Path) -> Path:
    """
    Save a batch of figures from one report()/plot() run to disk, then close them.

    figs : dict[str, plt.Figure]
        Maps a descriptive filename stem (no extension) to its figure.
    classname : str
        Name of the class that produced these figures, used as the output directory's prefix.
    savepath : str | Path
        Base directory under which a new f"{classname}_{timestamp}" directory is created.

    Returns Path
        The created directory containing the saved figures.
    """

    out_dir = Path(savepath) / f"{classname}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, fig in figs.items():
        fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return out_dir

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
def generate_toy_equity(*, expected_return: float | np.ndarray = 0, sharpe: None = None, volatility: float | np.ndarray = 0.01, beta: float | np.ndarray = 0, starting_book_value: int | float = 100_000, portfolio: Portfolio, benchmark: Optional[pd.Series] = None, distribution: Optional[Callable] = None, random_seed: Optional[int] = 42) -> tuple[pd.Series | pd.DataFrame, np.ndarray]: ...

@overload
def generate_toy_equity(*, expected_return: None = None, sharpe: float | np.ndarray = 0, volatility: float | np.ndarray = 0.01, beta: float | np.ndarray = 0, starting_book_value: int | float = 100_000, portfolio: Portfolio, benchmark: Optional[pd.Series] = None, distribution: Optional[Callable] = None, random_seed: Optional[int] = 42) -> tuple[pd.Series | pd.DataFrame, np.ndarray]: ... 

def generate_toy_equity(
    *, 
    expected_return: Optional[float | np.ndarray] = None,
    sharpe: Optional[float | np.ndarray] = None,
    volatility: float | np.ndarray = 0.01,
    beta: float | np.ndarray = 0,
    starting_book_value: int | float = 100_000,
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
    starting_book_value : int | float = 100_000
        Scale factor for book equity.
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
        
        equity = (1 + returns_matrix).cumprod(axis=0) * starting_book_value
        valid = np.ones(expected_return.shape, dtype=bool) # no idiosyncratic-variance constraint applies without a benchmark

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
    
    equity = (1 + beta * benchmark[:, None] + idiosyncratic_returns).cumprod(axis=0) * starting_book_value

    labels = [
        f"{f"S:{r / v if v else 0:.1f}" if sharpe is not None else f"R:{r:.3f}"}|σ:{v:.3f}|β:{b:.1f}"
        for r, v, b in zip(expected_return, volatility, beta)
    ]
   
    return pd.Series(equity[:, 0], index=portfolio.df.index, name=labels[0]) if is_scalar else pd.DataFrame(equity, index=portfolio.df.index, columns=labels), valid

def crash_period_estimators(benchmark: pd.Series, lookback_window: int = 21) -> None:
    """
    Debug function to eyeball different heuristics for determining crash periods.
    Determined the use of Centred Window Return in StrategyConnector._tradeoff_profile.
    
    benchmark : pd.Series
        Equity curve of the benchmark, intraday-indexed.
    lookback_window : int = 21
        Number of days used for lookback in rolling estimators.
        Defaults to 21 (1 trading mth).
    """
    
    benchmark = benchmark.resample("D").last().dropna()
    bench_ret = benchmark.pct_change().dropna()

    def spans(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Collapse a boolean daily mask into contiguous (start, end) date spans."""

        out, start = [], None

        for dt, flag in mask.items():
            if flag and start is None:
                start = dt
            elif not flag and start is not None:
                out.append((start, dt))
                start = None

        if start is not None:
            out.append((start, mask.index[-1]))

        return out

    def hmm_crash_mask(returns: pd.Series, n_iter: int = 25) -> pd.Series:
        """2-state Gaussian HMM (Baum-Welch EM); crash = lower-mean state."""

        x = returns.to_numpy()
        n = len(x)

        mu = np.array([x.mean() + x.std(), x.mean() - x.std()])
        var = np.array([x.var(), x.var() * 3])
        trans = np.array([[0.98, 0.02], [0.05, 0.95]])
        pi = np.array([0.9, 0.1])

        for _ in range(n_iter):
            b = np.clip(np.stack([
                np.exp(-0.5 * (x - mu[k]) ** 2 / var[k]) / np.sqrt(2 * np.pi * var[k])
                for k in range(2)
            ], axis=1), 1e-300, None)

            alpha, c = np.zeros((n, 2)), np.zeros(n)
            alpha[0] = pi * b[0]
            c[0] = alpha[0].sum()
            alpha[0] /= c[0]

            for t in range(1, n):
                alpha[t] = (alpha[t - 1] @ trans) * b[t]
                c[t] = alpha[t].sum()
                alpha[t] /= c[t]

            beta = np.zeros((n, 2))
            beta[-1] = 1

            for t in range(n - 2, -1, -1):
                beta[t] = (trans @ (b[t + 1] * beta[t + 1])) / c[t + 1]

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)

            xi_sum = np.zeros((2, 2))

            for t in range(n - 1):
                xi_sum += (alpha[t][:, None] * trans * (b[t + 1] * beta[t + 1])[None, :]) / c[t + 1]

            pi = gamma[0]
            trans = xi_sum / xi_sum.sum(axis=1, keepdims=True)

            for k in range(2):
                w = gamma[:, k]
                mu[k] = (w @ x) / w.sum()
                var[k] = max((w @ (x - mu[k]) ** 2) / w.sum(), 1e-8)

        crash_state = int(np.argmin(mu)) # crash = lower-mean regime

        return pd.Series(gamma[:, crash_state] > 0.5, index=returns.index)

    # crash-period estimators
    estimators = {}

    ewma_vol = bench_ret.ewm(span=lookback_window).std()
    estimators["EWMA Vol"] = ewma_vol > ewma_vol.quantile(0.90)

    centred_vol = bench_ret.rolling(lookback_window, center=True).std()
    estimators["Centred Window Vol"] = centred_vol > centred_vol.quantile(0.90)

    half_window = lookback_window // 2
    centred_ret = benchmark.shift(-half_window) / benchmark.shift(half_window) - 1 # net move over a centred window, not just dispersion
    estimators["Centred Window Return"] = centred_ret < centred_ret.quantile(0.10)
    
    dd = benchmark / benchmark.cummax() - 1.0
    worst = dd.rolling(2 * half_window + 1, center=True, min_periods=2 * half_window + 1).min()
    estimators["Centred Window Absolute"] = worst < -0.15  

    roll_mean = bench_ret.rolling(lookback_window).mean()
    roll_std = bench_ret.rolling(lookback_window).std()
    estimators["Rolling Z-Score"] = (bench_ret - roll_mean) / roll_std < -2

    estimators["Full-Sample Percentile"] = bench_ret < bench_ret.quantile(0.05)

    drawdown = compute_drawdown(benchmark / benchmark.iloc[0])
    estimators["Drawdown Episode"] = drawdown <= -0.10

    estimators["2-State HMM"] = hmm_crash_mask(bench_ret)

    masks = {name: mask.reindex(benchmark.index).fillna(False) for name, mask in estimators.items()}

    fig, (ax_main, ax_bars) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5]},
    )

    ax_main.plot(benchmark.index, benchmark.values, color="black", linewidth=1)
    ax_main.set_title("Benchmark (Full Sample) vs. Crash Period Estimators", loc="left", fontweight="bold")
    ax_main.set_ylabel("Benchmark Level")
    ax_main.margins(x=0)

    names = list(masks.keys())

    for i, name in enumerate(names):
        for start, end in spans(masks[name]):
            x0 = mdates.date2num(start)
            width = max(mdates.date2num(end) - x0, 0.8) # keep single-day spans visible

            ax_bars.barh(i, width, left=x0, height=0.6, color="firebrick")

    ax_bars.set_yticks(range(len(names)))
    ax_bars.set_yticklabels(names, fontsize=8)
    ax_bars.invert_yaxis()
    ax_bars.margins(x=0)
    ax_bars.set_xlabel("Date", fontweight="bold")

    plt.tight_layout()
    plt.show()
    plt.close(fig)
