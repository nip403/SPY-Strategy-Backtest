import pandas as pd
import numpy as np
from typing import Any
from ..core import Portfolio

class PortfolioQuarterHourSample(Portfolio):
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialise a 15-minute sampled portfolio backtest - initial strategy samples every 30 mins.

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
    
        super().__init__(**kwargs)
        
    def _set_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate quarter-hour strategy positions from trading signals.

        Creates entries every 15 minutes, evaluates exits every 30 minutes, and scales exposure using volatility targeting.

        df : pd.DataFrame
            Preprocessed market data containing strategy indicators.

        Returns pd.DataFrame
            DataFrame containing generated portfolio positions.
        """    
        
        entry_intervals = df.index.minute.isin([0, 15, 30, 45])
        exit_intervals = df.index.minute.isin([0, 30])
        
        long_entry = (df["close"] > df["upper_bound"]) & entry_intervals & self.long_perm
        short_entry = (df["close"] < df["lower_bound"]) & entry_intervals & self.short_perm

        long_exit = (df["close"] < df["long_stop"]) & exit_intervals
        short_exit = (df["close"] > df["short_stop"]) & exit_intervals

        end_of_day = df.index.time == pd.Timestamp("15:59").time()

        df["position"] = np.nan

        df.loc[long_exit | short_exit | end_of_day, "position"] = 0
        df.loc[long_entry, "position"] = 1
        df.loc[short_entry, "position"] = -1

        df["position"] = df["position"].ffill().fillna(0) * (self.target_vol / df["std"]).clip(lower=-4, upper=4) 
    
        return df