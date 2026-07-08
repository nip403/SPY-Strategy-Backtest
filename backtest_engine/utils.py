from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import date
from typing import Callable, Optional

def round_date(date_index: pd.DataFrame, dt: date) -> date:
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

def gen_toy_returns(periods: int, *, mean: float = 0, std: float = 1, distribution: Optional[Callable] = None) -> np.ndarray:
    return (distribution or np.random.normal)(loc=mean, scale=std, size=periods)

def gen_toy_equity(portfolio: Portfolio, *, mean: float = 0, std: float = 0.01, distribution: Optional[Callable] = None) -> pd.Series:
    returns = gen_toy_returns(len(portfolio.df), mean=mean, std=std, distribution=distribution)

    return pd.Series((1 + returns).cumprod() * portfolio.aum, index=portfolio.df.index, name="equity")