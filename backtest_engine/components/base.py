from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import pandas as pd
import numpy as np

@dataclass
class BacktestContext:
    """
    Snapshot of portfolio-level parameters shared with pluggable components.
    
    aum : float
        Portfolio capital used for position sizing and cost/capacity scaling.
    target_vol : float
        Target volatility used for position sizing.
    long_perm : bool
        Whether long positions are permitted.
    short_perm : bool
        Whether short positions are permitted.
    """

    aum: float
    target_vol: float
    long_perm: bool
    short_perm: bool

class StrategyComponent(Protocol):
    def set(self, df: pd.DataFrame, ctx: BacktestContext) -> pd.DataFrame:
        """
        Generate target strategy positions from trading signals.

        df : pd.DataFrame
            Preprocessed market data containing strategy indicators.
        ctx : BacktestContext
            Shared portfolio parameters for the backtest run.

        Returns pd.DataFrame
            DataFrame containing a "position" column of target exposure.
        """
        ...

class ExecutionComponent(Protocol):
    def fill(self, df: pd.DataFrame, ctx: BacktestContext) -> pd.DataFrame:
        """
        Constrain target positions into realistic fills.

        df : pd.DataFrame
            Market data containing the "position" column produced by a Strategy.
        ctx : BacktestContext
            Shared portfolio parameters for the backtest run.

        Returns pd.DataFrame
            DataFrame with "position" replaced by the capacity-constrained actual position.
        """
        ...

class CostComponent(Protocol):
    def compute(self, df: pd.DataFrame, ctx: BacktestContext) -> pd.DataFrame:
        """
        Calculate net returns from gross for the backtested AUM.

        df : pd.DataFrame
            Backtest data containing gross returns and final positions.
        ctx : BacktestContext
            Shared portfolio parameters for the backtest run.

        Returns pd.Series
            Net returns after costs, aligned to df's index.
        """
        ...

    def trim(self, index: pd.Index) -> None:
        """
        Realign self._cache to the final backtest index (after dropna).
        
        Override if a subclass caches more than a single DataFrame/Series.

        index : pd.Index
            Final trimmed index of the backtested portfolio dataframe.
        """

        self._cache = self._cache.loc[index]

    def returns_matrix(self, aum: np.ndarray) -> np.ndarray:
        """
        Generate vectorised net returns across multiple portfolio sizes without rerunning the backtest.

        aum : np.ndarray
            Portfolio capital values to evaluate.

        Returns np.ndarray
            Matrix of net returns with one column per AUM value.
        """
        ...
