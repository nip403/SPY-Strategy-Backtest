import pandas as pd
import numpy as np
from typing import Any
from ..core import Portfolio

class PortfolioRollingImmediateStop(Portfolio):
    def __init__(self, *, entry_window: int = 30, **kwargs: Any) -> None:
        """
        Initialise a rolling confirmation strategy variant with immediate stops.
        Rolling stops were also tested, but unambiguously performed worse.

        Entry signals require the boundary condition to persist for the confirmation window, while exits trigger immediately.

        entry_window : int = 30
            Number of observations required to confirm an entry signal.
        **kwargs: Any
            Base portfolio arrguments to pass on.
            
            df : pd.DataFrame
                1-minute intraday market data used for signal generation and execution.
            aum : float = 100_000
                Initial portfolio capital used for equity calculations.
            target_vol : float = 0.02
                Target volatility used for position sizing.
            long_permissions : bool = True
                Whether long positions are permitted.
            short_permissions : bool = True
                Whether short positions are permitted.
        """
        
        self.conf = entry_window
        
        super().__init__(**kwargs)

    def _set_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate positions using rolling entry confirmation and immediate exits.

        Requires entry conditions to persist before opening positions, while exit conditions are evaluated continuously.

        df : pd.DataFrame
            Preprocessed market data containing strategy indicators.

        Returns pd.DataFrame
            DataFrame containing generated portfolio positions.
        """
      
        # entry signal (rolling confirmation)
        raw_long_entry = df["close"] > df["upper_bound"]
        raw_short_entry = df["close"] < df["lower_bound"]

        long_entry = (raw_long_entry.rolling(window=self.conf).sum() == self.conf) & self.long_perm
        short_entry = (raw_short_entry.rolling(window=self.conf).sum() == self.conf) & self.short_perm
        
        # exit signal (triggers immediately upon crossing boundary)
        long_exit = df["close"] < df["long_stop"]
        short_exit = df["close"] > df["short_stop"]
        end_of_day = df.index.time == pd.Timestamp("15:59").time()

        # set positions & backtest
        df["position"] = np.nan
        
        df.loc[long_exit | short_exit | end_of_day, "position"] = 0
        df.loc[long_entry, "position"] = 1
        df.loc[short_entry, "position"] = -1

        df["position"] = df["position"].ffill().fillna(0) * (self.target_vol / df["std"]).clip(lower=-4, upper=4) 
        
        return df

class PortfolioRollingIntervalStop(Portfolio):
    def __init__(self, *, entry_window: int = 30, **kwargs: Any) -> None:
        """
        Initialise a rolling confirmation strategy with interval stops to isolate the confirmation effect.

        Entry signals require the boundary condition to persist for the confirmation window, while exits are evaluated as according to the base strategy.

        entry_window : int = 30
            Number of observations required to confirm an entry signal.
        **kwargs: Any
            Base portfolio arrguments to pass on.
            
            df : pd.DataFrame
                1-minute intraday market data used for signal generation and execution.
            aum : float = 100_000
                Initial portfolio capital used for equity calculations.
            target_vol : float = 0.02
                Target volatility used for position sizing.
            long_permissions : bool = True
                Whether long positions are permitted.
            short_permissions : bool = True
                Whether short positions are permitted.
        """
        
        self.conf = entry_window
        
        super().__init__(**kwargs)

    def _set_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate positions using rolling entry confirmation and interval exits.

        Requires entry conditions to persist, while exits are evaluated at fixed time intervals.

        df : pd.DataFrame
            Preprocessed market data containing strategy indicators.

        Returns pd.DataFrame
            DataFrame containing generated portfolio positions.
        """
        
        # entry signal (rolling confirmation)
        raw_long_entry = df["close"] > df["upper_bound"]
        raw_short_entry = df["close"] < df["lower_bound"]

        long_entry = (raw_long_entry.rolling(window=self.conf).sum() == self.conf) & self.long_perm
        short_entry = (raw_short_entry.rolling(window=self.conf).sum() == self.conf) & self.short_perm
        
        # exit signal (evaluated semi hourly)
        intervals = df.index.minute.isin([0, 30])
        
        long_exit = (df["close"] < df["long_stop"]) & intervals
        short_exit = (df["close"] > df["short_stop"]) & intervals
        end_of_day = df.index.time == pd.Timestamp("15:59").time()

        # set positions & backtest
        df["position"] = np.nan
        
        df.loc[long_exit | short_exit | end_of_day, "position"] = 0
        df.loc[long_entry, "position"] = 1
        df.loc[short_entry, "position"] = -1

        df["position"] = df["position"].ffill().fillna(0) * (self.target_vol / df["std"]).clip(lower=-4, upper=4) 
        
        return df