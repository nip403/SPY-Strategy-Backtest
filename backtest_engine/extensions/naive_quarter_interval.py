import pandas as pd
import numpy as np
from ..core import Portfolio

class PortfolioQuarterHourSample(Portfolio):
    def __init__(self, df: pd.DataFrame, aum: float = 100_000, target_vol: float = 0.02, long_permissions: bool = True, short_permissions: bool = True) -> None:
        """
        Initialise a 15-minute sampled portfolio backtest - initial strategy samples every 30 mins.

        df : pd.DataFrame
            1-minute intraday market data used for signal generation and execution.
            Must contain required OHLCV fields and datetime index.
        aum : float = 100_000
            Initial portfolio value.
        target_vol : float = 0.02
            Target volatility used for position sizing, specified in the attached paper.
        long_permissions : bool = True
            Whether long trades are allowed.
        short_permissions : bool = True
            Whether short trades are allowed.
        """
    
        super().__init__(df=df, aum=aum, target_vol=target_vol, long_permissions=long_permissions, short_permissions=short_permissions)
        
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