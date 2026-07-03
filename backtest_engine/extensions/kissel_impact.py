import pandas as pd
import numpy as np
from typing import Optional
from ..core import Portfolio
 
# I-Star impact model - https://www.kissellresearch.com/post/i-star-market-impact-model
class PortfolioDynamicCost(Portfolio):
    DEFAULT_PARAMS = {
        "a1": 700, # outdated by over a decade
        "a2": 0.5,
        "a3": 1.0,
        "a4": 0.5,
        "b1": 0.85, # llm consensus
        "lookback": 20, # 1 trading mth
    }
    
    def __init__(self, df: pd.DataFrame, aum: float = 100_000, target_vol: float = 0.02, coeff_config: Optional[dict] = None, long_permissions: Optional[bool] = True, short_permissions: Optional[bool] = True) -> None:
        self._config = {**self.DEFAULT_PARAMS, **(coeff_config or {})}
        
        self.a1 = self._config["a1"]
        self.a2 = self._config["a2"]
        self.a3 = self._config["a3"]
        self.a4 = self._config["a4"]
        self.b1 = self._config["b1"]
        
        self.lookback_window = self._config["lookback"]
        
        super().__init__(df=df, aum=aum, target_vol=target_vol, long_permissions=long_permissions, short_permissions=short_permissions)
 
    def _backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._preprocess(df)
 
        # signal generation
        intervals = df.index.minute.isin([0, 30])
        long_entry = (df["close"] > df["upper_bound"]) & intervals & self.long_perm
        short_entry = (df["close"] < df["lower_bound"]) & intervals & self.short_perm
 
        long_exit = (df["close"] < df["long_stop"]) & intervals
        short_exit = (df["close"] > df["short_stop"]) & intervals
 
        end_of_day = df.index.time == pd.Timestamp("15:59").time()
 
        df["position"] = np.nan
 
        df.loc[long_exit | short_exit | end_of_day, "position"] = 0
        df.loc[long_entry, "position"] = 1
        df.loc[short_entry, "position"] = -1
 
        df["position"] = df["position"].ffill().fillna(0) * (self.target_vol / df["std"]).clip(lower=-4, upper=4)
 
        df["gross_ret"] = df["position"].shift(1).fillna(0) * df["ret"]
        
        # rolling average daily volume & ann. volatility 
        dates = pd.Series(df.index.date, index=df.index)
        buckets = df.groupby(df.index.date)
        adv = buckets["volume"].sum().rolling(self.lookback_window).mean().shift(1)
        ann_vol = buckets["close"].last().pct_change().rolling(self.lookback_window).std().shift(1) * 252 ** 0.5

        # I-star market impact
        delta_leverage = df["position"].diff().abs().fillna(0)
        shares = delta_leverage * self.aum / df["close"]
        participation_rate = shares / df["volume"]
        
        i_star = self.a1 * (shares / dates.map(adv)).fillna(0) ** self.a2 * dates.map(ann_vol).fillna(0) ** self.a3 # impact of trade (bps of trade value) on stock
        
        df["mkt_impact"] = (self.b1 * i_star * participation_rate ** self.a4 + (1 - self.b1) * i_star) / 10_000 * delta_leverage # impact cost expected, scaled with leverage
        
        df["net_ret"] = df["gross_ret"] - df["mkt_impact"] - self.COMMISSION * shares / self.aum
        df["cum_ret"] = (1 + df["net_ret"].fillna(0)).cumprod()
        df["equity_curve"] = self.aum * df["cum_ret"]
        df["benchmark"] = (1 + df["ret"].fillna(0)).cumprod() * self.aum
 
        return df.dropna()
