import pandas as pd
import numpy as np
from typing import Optional
from ..core import Portfolio

class PortfolioQuarterHourSample(Portfolio):
    def __init__(self, df: pd.DataFrame, aum: float = 100_000, target_vol: float = 0.02, long_permissions: Optional[bool] = True, short_permissions: Optional[bool] = True) -> None:
        super().__init__(df=df, aum=aum, target_vol=target_vol, long_permissions=long_permissions, short_permissions=short_permissions)
        
    def _backtest(self, df: pd.DataFrame) -> pd.DataFrame:      
        df = self._preprocess(df)
        
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
        
        df["gross_ret"] = df["position"].shift(1).fillna(0) * df["ret"]
        df["net_ret"] = df["gross_ret"] - (df["position"].diff().abs().fillna(0) * self.frictions / df["close"])
        df["cum_ret"] = (1 + df["net_ret"].fillna(0)).cumprod()
        df["equity_curve"] = self.aum * df["cum_ret"]
        
        df["benchmark"] = (1 + df["ret"].fillna(0)).cumprod() * self.aum
                
        return df.dropna()