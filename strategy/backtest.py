import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from datetime import datetime

class Portfolio:
    # per share frictions    
    commission = 0.0035
    slippage = 0.001
    
    def __init__(self, df: pd.DataFrame, aum: float = 100_000, target_vol: float = 0.02):
        self.aum = aum
        self.target_vol = target_vol
        self.frictions = self.commission + self.slippage
        
        self.df = self._backtest(self.df)
        
        self.t0 = self.df.index[0].date()
        self.t1 = self.df.index[-1].date()
        
    def _backtest(self, df: pd.DataFrame) -> pd.DataFrame:      
        # position sizing
        
        closes = df.groupby(df.index.date)["close"].last()
        returns = closes.pct_change()
        df["mu"] = pd.Series(df.index.date, index=df.index).map(returns.rolling(window=14).mean().shift(1))
        df["std"] = pd.Series(df.index.date, index=df.index).map(returns.rolling(window=14).std().shift(1))

        df["ret"] = df["close"].pct_change().shift(-1) # shift to represent return from open to close

        # noise area
        
        df["deviation"] = ((df["close"] / df["daily_open"]) - 1).abs() # "move"
        df["sigma"] = df.groupby("time")["deviation"].transform(lambda x: x.shift(1).rolling(14, min_periods=14).mean())

        df["upper_bound"] = df[["daily_open", "prev_close"]].max(axis=1) * (1 + df["sigma"])
        df["lower_bound"] = df[["daily_open", "prev_close"]].min(axis=1) * (1 - df["sigma"])

        df["long_stop"] = df[["upper_bound", "vwap"]].max(axis=1)
        df["short_stop"] = df[["lower_bound", "vwap"]].min(axis=1)
        
        # signal generation

        intervals = df.index.minute.isin([0, 30])
        long_entry = (df["close"] > df["upper_bound"]) & intervals
        short_entry = (df["close"] < df["lower_bound"]) & intervals

        long_exit = (df["close"] < df["long_stop"]) & intervals
        short_exit = (df["close"] > df["short_stop"]) & intervals

        end_of_day = df.index.time == pd.Timestamp("15:59").time()

        df["position"] = np.nan

        # set positions

        df.loc[long_exit | short_exit | end_of_day, "position"] = 0
        df.loc[long_entry, "position"] = 1
        df.loc[short_entry, "position"] = -1

        df["position"] = df["position"].ffill().fillna(0) * (self.target_vol / df["std"]).clip(lower=-4, upper=4) # multiply by aum // daily open
                
        # backtest (fractional shares assumed for vectorisation)

        df["gross_ret"] = df["position"] * df["ret"]

        # net of costs: cost% = delta(position) * leverage / cost/share

        df["net_ret"] = df["gross_ret"] - (df["position"].diff().abs().fillna(0) * self.frictions / df["close"])

        df["cum_ret"] = (1 + df["net_ret"]).fillna(0).cumprod()
        df["equity_curve"] = self.aum * df["cum_ret"]

        # for visuals

        df["benchmark"] = (1 + df["ret"]).cumprod() * self.aum
                
        return df.dropna()
    
    def daily_performance(self, date: datetime) -> dict:
        assert self.t0 <= date <= self.t1
        
        daily_aum = self.df["equity_curve"].groupby(self.df.index.date).last()
        daily_bench = self.df["benchmark"].groupby(self.df.index.date).last()

        plt.figure(figsize=(14, 7))

        plt.plot(daily_aum.index, daily_aum.values, color="blue", label="Strategy")
        plt.plot(daily_bench.index, daily_bench.values, color="red", label="SPY")

        plt.margins(x=0)
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))

        plt.legend()
        plt.show()
    
    #todo: suummary stats, plots, strategy customisation