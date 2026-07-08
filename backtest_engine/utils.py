from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import date
from typing import Callable, Optional

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

def generate_toy_returns(periods: int, *, mean: float = 0, std: float = 1, distribution: Optional[Callable] = None) -> np.ndarray:
    """
    Generate synthetic returns from a specified distribution.

    periods : int
        Number of return observations to generate.
    mean : float = 0
        Target minute mean.
    std : float = 1
        Target minute standard deviation.
    distribution : Callable = None
        Random distribution function that accepts loc, scale, and size arguments, and returns an np.ndarray.
        Defaults to numpy normal distribution.

    Returns np.ndarray
        Array of simulated returns.
    """
    
    return (distribution or np.random.normal)(loc=mean, scale=std, size=periods)

# annual exp ret and vol needed, assumes minute-intraday frequency
def generate_toy_equity(portfolio: Portfolio, *, expected_return: float = 0, volatility: float = 0.01, distribution: Optional[Callable] = None) -> pd.Series:
    """
    Generate a synthetic intraday equity curve based on desired annual statistics.

    portfolio : Portfolio
        Portfolio used to determine the intraday index and starting AUM.
    expected_return : float = 0
        Target nnualised expected return.
    volatility : float = 0.01
        Target annualised volatility used to scale intraday returns.
    distribution : Callable = None
        Random distribution function that accepts loc, scale, and size arguments, and returns an np.ndarray.
        Defaults to numpy normal distribution.

    Returns pd.Series
        Synthetic equity curve aligned to the portfolio intraday index.
    """
       
    returns = generate_toy_returns(
        len(portfolio.df), 
        mean = expected_return / (252 * 390), 
        std = volatility / np.sqrt(252 * 390), 
        distribution=distribution,
    )

    return pd.Series((1 + returns).cumprod() * portfolio.aum, index=portfolio.df.index, name="equity")