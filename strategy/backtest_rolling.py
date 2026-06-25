import pandas as pd
import numpy as np
from .backtest import Portfolio

class PortfolioRollingImmediateStop(Portfolio):
    def __init__(self, df: pd.DataFrame, aum: float = 100_000, target_vol: float = 0.02, entry_window: int = 30):
        self.conf = entry_window
        
        super().__init__(df=df, aum=aum, target_vol=target_vol)

    def _backtest(self, df: pd.DataFrame) -> pd.DataFrame:  
        # position sizing    
        closes = df.groupby(df.index.date)["close"].last()
        returns = closes.pct_change()
        
        df["mu"] = pd.Series(df.index.date, index=df.index).map(returns.rolling(window=14).mean().shift(1))
        df["std"] = pd.Series(df.index.date, index=df.index).map(returns.rolling(window=14).std().shift(1))
        df["ret"] = df["close"].pct_change()

        # noise area
        df["deviation"] = ((df["close"] / df["daily_open"]) - 1).abs()
        df["sigma"] = df.groupby("time")["deviation"].transform(lambda x: x.shift(1).rolling(14, min_periods=14).mean())

        df["upper_bound"] = df[["daily_open", "prev_close"]].max(axis=1) * (1 + df["sigma"])
        df["lower_bound"] = df[["daily_open", "prev_close"]].min(axis=1) * (1 - df["sigma"])

        df["long_stop"] = df[["upper_bound", "vwap"]].max(axis=1)
        df["short_stop"] = df[["lower_bound", "vwap"]].min(axis=1)
        
        # entry signal (rolling confirmation)
        raw_long_entry = df["close"] > df["upper_bound"]
        raw_short_entry = df["close"] < df["lower_bound"]

        long_entry = raw_long_entry.rolling(window=self.conf).sum() == self.conf
        short_entry = raw_short_entry.rolling(window=self.conf).sum() == self.conf
        
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
        
        held_position = df["position"].shift(1).fillna(0)
        df["gross_ret"] = held_position * df["ret"]

        # frictions
        df["net_ret"] = df["gross_ret"] - (df["position"].diff().abs().fillna(0) * self.frictions / df["close"])

        df["cum_ret"] = (1 + df["net_ret"].fillna(0)).cumprod()
        df["equity_curve"] = self.aum * df["cum_ret"]
        
        # visuals
        df["benchmark"] = (1 + df["ret"].fillna(0)).cumprod() * self.aum
                
        return df.dropna()
    
# isolate confirmation effect
class PortfolioRollingIntervalStop(Portfolio):
    def __init__(self, df: pd.DataFrame, aum: float = 100_000, target_vol: float = 0.02, entry_window: int = 30):
        self.conf = entry_window
        
        super().__init__(df=df, aum=aum, target_vol=target_vol)

    def _backtest(self, df: pd.DataFrame) -> pd.DataFrame:  
        # position sizing    
        closes = df.groupby(df.index.date)["close"].last()
        returns = closes.pct_change()
        
        df["mu"] = pd.Series(df.index.date, index=df.index).map(returns.rolling(window=14).mean().shift(1))
        df["std"] = pd.Series(df.index.date, index=df.index).map(returns.rolling(window=14).std().shift(1))
        df["ret"] = df["close"].pct_change()

        # noise area
        df["deviation"] = ((df["close"] / df["daily_open"]) - 1).abs()
        df["sigma"] = df.groupby("time")["deviation"].transform(lambda x: x.shift(1).rolling(14, min_periods=14).mean())

        df["upper_bound"] = df[["daily_open", "prev_close"]].max(axis=1) * (1 + df["sigma"])
        df["lower_bound"] = df[["daily_open", "prev_close"]].min(axis=1) * (1 - df["sigma"])

        # dynamic stop levels
        df["long_stop"] = df[["upper_bound", "vwap"]].max(axis=1)
        df["short_stop"] = df[["lower_bound", "vwap"]].min(axis=1)
        
        # entry signal (rolling confirmation)
        raw_long_entry = df["close"] > df["upper_bound"]
        raw_short_entry = df["close"] < df["lower_bound"]

        long_entry = raw_long_entry.rolling(window=self.conf).sum() == self.conf
        short_entry = raw_short_entry.rolling(window=self.conf).sum() == self.conf
        
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
        
        held_position = df["position"].shift(1).fillna(0)
        df["gross_ret"] = held_position * df["ret"]

        # frictions
        df["net_ret"] = df["gross_ret"] - (df["position"].diff().abs().fillna(0) * self.frictions / df["close"])

        df["cum_ret"] = (1 + df["net_ret"].fillna(0)).cumprod()
        df["equity_curve"] = self.aum * df["cum_ret"]
        
        # visuals
        df["benchmark"] = (1 + df["ret"].fillna(0)).cumprod() * self.aum
                
        return df.dropna()