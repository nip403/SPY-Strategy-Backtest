from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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

class StrategyComponent(ABC):
    @abstractmethod
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

class ExecutionComponent(ABC):
    @abstractmethod
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

    def trim(self, index: pd.Index) -> None:
        """
        Realign self._cache to the final backtest index (after dropna).

        Override if a subclass caches more than a single DataFrame/Series.

        index : pd.Index
            Final trimmed index of the backtested portfolio dataframe.
        """

        self._cache = self._cache.loc[index]

    def fill_matrix(self, aum: np.ndarray) -> np.ndarray:
        """
        Generate vectorised positions across multiple portfolio sizes.
        Default implementation assumes fills are independent of AUM.

        aum : np.ndarray
            Portfolio capital values to evaluate.

        Returns np.ndarray
            Matrix of positions with one column per AUM value.
        """

        return np.tile(self._cache.to_numpy()[:, None], (1, len(aum)))

class CostComponent(ABC):
    @abstractmethod
    def expense(self, df: pd.DataFrame, ctx: BacktestContext) -> pd.DataFrame:
        """
        Calculate net returns from gross for the backtested AUM.

        df : pd.DataFrame
            Backtest data containing gross returns and final positions.
        ctx : BacktestContext
            Shared portfolio parameters for the backtest run.

        Returns pd.DataFrame
            df with "net_ret" added: returns after costs.
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

    @abstractmethod
    def returns_matrix(self, aum: np.ndarray, position_matrix: np.ndarray, gross_matrix: np.ndarray) -> np.ndarray:
        """
        Generate vectorised net returns across multiple portfolio sizes.

        position_matrix and gross_matrix are supplied by Portfolio.returns_matrix (from ExecutionComponent.fill_matrix)
        Needed as capacity-constrained execution makes positions & therefore returns AUM-dependent.

        aum : np.ndarray
            Portfolio capital values to evaluate.
        position_matrix : np.ndarray
            Capacity-constrained positions, one column per AUM value.
        gross_matrix : np.ndarray
            Gross returns, one column per AUM value.

        Returns np.ndarray
            Matrix of net returns with one column per AUM value.
        """
        ...
