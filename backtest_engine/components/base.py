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
    max_leverage : float
        Maximum leverage backtest can deploy.
    """

    aum: float
    target_vol: float
    long_perm: bool
    short_perm: bool
    max_leverage: float

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
    def fill(self, df: pd.DataFrame, ctx: BacktestContext, cache: dict) -> pd.DataFrame:
        """
        Constrain target positions into realistic fills.
        Cache is owned by the Portfolio instance, and cache key/item management is handled within the component methods.

        df : pd.DataFrame
            Market data containing the "position" column produced by a Strategy.
        ctx : BacktestContext
            Shared portfolio parameters for the backtest run.
        cache : dict
            The portfolio's cache, to populate with whatever fill_matrix will later need.

        Returns pd.DataFrame
            DataFrame with "position" replaced by the capacity-constrained actual position.
        """
        ...

    def fill_matrix(self, aum: np.ndarray, cache: dict) -> np.ndarray:
        """
        Generate vectorised positions across multiple portfolio sizes.
        Default implementation assumes fills are independent of AUM.

        aum : np.ndarray
            Portfolio capital values to evaluate.
        cache : dict
            The Portfolio's cache, populated by fill().

        Returns np.ndarray
            Matrix of positions with one column per AUM value. 
            A read-only broadcast view as the underlying data is AUM invariant.
        """

        position = cache["position"].to_numpy()[:, None]

        return np.broadcast_to(position, (len(position), len(aum)))

class CostComponent(ABC):
    @abstractmethod
    def expense(self, df: pd.DataFrame, ctx: BacktestContext, cache: dict) -> pd.DataFrame:
        """
        Calculate net returns from gross for the backtested AUM.
        Cache is owned by the Portfolio instance, and cache key/item management is handled within the component methods.

        df : pd.DataFrame
            Backtest data containing gross returns and final positions.
        ctx : BacktestContext
            Shared portfolio parameters for the backtest run.
        cache : dict
            The portfolio's cache, to populate with whatever returns_matrix will later need.

        Returns pd.DataFrame
            df with "net_ret" added: returns after costs.
        """
        ...

    @abstractmethod
    def returns_matrix(self, aum: np.ndarray, position_matrix: np.ndarray, gross_matrix: np.ndarray, cache: dict) -> np.ndarray:
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
        cache : dict
            The Portfolio's cache, populated by expense().

        Returns np.ndarray
            Matrix of net returns with one column per AUM value.
        """
        ...
