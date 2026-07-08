import pandas as pd
import numpy as np
from typing import Optional
from ..core import Portfolio

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
    
        self._cache = self._cache.loc[self.df.index]
    
    def _costing(self, df: pd.DataFrame) -> pd.Series:
        dates = pd.Series(df.index.date, index=df.index)
        buckets = df.groupby(df.index.date)
        
        # I* market impact model
        # build cache for net returns tensor, factor out aum for vectorised calcs in returns_matrix
        # isolate aum from i* and mi and cache for aum multiplication later 
        delta_leverage = df["position"].diff().abs().fillna(0).to_numpy()
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()
        
        adv = dates.map(buckets["volume"].sum().rolling(self.lookback_window).mean().shift(1)).fillna(0).to_numpy()
        ann_vol = dates.map(buckets["close"].last().pct_change().rolling(self.lookback_window).std().shift(1) * np.sqrt(252)).fillna(0).to_numpy()

        # common factors, failsafe set div by volume=0 scenarios to 0
        denom_adv = close * adv
        adv_term = np.divide(delta_leverage, denom_adv, out=np.zeros_like(delta_leverage), where=(denom_adv != 0))
        
        denom_vol = close * volume
        participation = np.divide(delta_leverage, denom_vol, out=np.zeros_like(delta_leverage), where=(denom_vol != 0))
        
        perm = self.a1 * (adv_term ** self.a2) * (ann_vol ** self.a3) * delta_leverage
        temp = perm * (participation ** self.a4)
        
        commission = np.divide(self.COMMISSION * delta_leverage, close, out=np.zeros_like(delta_leverage), where=(close != 0))
        
        gross = (df["position"].shift(1).fillna(0) * df["ret"].fillna(0)).to_numpy()
        
        self._cache = pd.DataFrame({
            "gross": gross,
            "perm": perm,
            "temp": temp,
            "commission": commission,
        },
        index=df.index
        )
        
        # info
        df["mkt_impact"] = np.nan_to_num((self.b1 * temp * (self.aum ** (self.a2 + self.a4)) + (1 - self.b1) * perm * (self.aum ** self.a2)) / 10_000, nan=0.0, posinf=0.0, neginf=0.0)

        return pd.Series(self.returns_matrix(np.array([self.aum]))[:, 0], index=df.index)
    
    def returns_matrix(self, aum: np.ndarray) -> np.ndarray:
        """
        1. shares = (delta_leverage x AUM) / close
        2. perm cost component = a1 x (shares / ADV)^a2 x vol^a3 
            = [a1 x (delta_leverage / (close x ADV))^a2 x vol^a3] x AUM^a2
            = baseline perm x AUM^a2
        3. temp cost component = perm cost x (shares / volume)^a4
            = [baseline perm x (delta_leverage / (close * volume))^a4] x AUM^(a2 + a4)
            = baseline temp x AUM^(a2 + a4)
        4. market impact cost = [b1 x baseline temp x AUM ^ (a2 + a4) + (1 - b1)] x [baseline perm x AUM^a2]
        """
        
        gross = self._cache["gross"].to_numpy()[:, None]
        perm = self._cache["perm"].to_numpy()[:, None]
        temp = self._cache["temp"].to_numpy()[:, None]
        commission = self._cache["commission"].to_numpy()[:, None]

        aum = aum[None, :]
        mkt = (self.b1 * temp * (aum ** (self.a2 + self.a4)) + (1 - self.b1) * perm * (aum ** self.a2)) / 10_000

        # handle division failures upstream - potentially a bad choice
        return np.nan_to_num(gross - mkt - commission, nan=0, posinf=0, neginf=0)
        