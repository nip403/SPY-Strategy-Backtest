import pandas as pd
import numpy as np
from typing import Optional, Any
from ..core import Portfolio
from .kissel_impact import PortfolioDynamicCost

class PortfolioCappedVolumeRollover(Portfolio):
    def __init__(self, *, participation_ceiling: float = 0.1, **kwargs: Any) -> None:
        """
        Initialise a standard portfolio that caps participation to a fixed share of contemporaneous volume to properly test capacity.
        Any unfilled portfolio of the desired trade size is carried forward until it is entirely filled or a new position target is hit.

        participation_ceiling: float = 0.1
            The maximum share of instantaneous volume the backtest is allowed to trade at any minute.
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
        
        self.participation_ceiling = participation_ceiling
    
        super().__init__(**kwargs)
        
    def _set_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate strategy positions from trading signals, constrained to a fixed % of minute volume using rollover. 
        Accumulates tradeable volume and continuously fills/rollover orders as liquidity and signals permit.

        df : pd.DataFrame
            Preprocessed market data containing strategy indicators.

        Returns pd.DataFrame
            DataFrame containing generated portfolio positions.
        """
        
        df = super()._set_positions(df)
        df["target"] = df["position"]
        
        # maximum leverage factor allowed
        df["capacity"] = (df["volume"] * df["close"] * self.participation_ceiling) / self.aum
        
        # split backtest into regimes (that have the same target); increment every time target changes
        mask = df["target"] != df["target"].shift(1)
        mask.iloc[0] = True
        df["regime"] = mask.cumsum()
        df["cum_cap"] = df.groupby("regime")["capacity"].cumsum() # cumulative capacity available in each regime
        
        # regime-/trade-level summary for path-dependent starts/ends
        regimes = pd.DataFrame({
            "target": df.groupby("regime")["target"].first(),
            "max_cap": df.groupby("regime")["cum_cap"].last(),
        })
        
        p_starts = np.zeros(len(regimes))
        p_current = 0
        
        for i in range(len(regimes)): # O(trades)
            p_starts[i] = p_current
            
            t, mc = regimes.iloc[i]
            
            # greedily resolving regime endpoints
            if t > p_current:
                p_current = min(t, p_current + mc)
            elif t < p_current:
                p_current = max(t, p_current - mc)
                
        regimes["p_start"] = p_starts
        df["p_start"] = df["regime"].map(regimes["p_start"])
        
        # matching and clipping minute by minute
        df["position"] = np.where(
            df["target"] > df["p_start"],
            np.minimum(df["target"], df["p_start"] + df["cum_cap"]),
            np.maximum(df["target"], df["p_start"] - df["cum_cap"]),
        )
        
        return df.drop(columns=["target", "capacity", "regime", "cum_cap", "p_start"])
    
class PortfolioCappedVolume(Portfolio):
    def __init__(self, *, participation_ceiling: float = 0.1, **kwargs: Any) -> None:
        """
        Initialise a standard portfolio that caps participation to a fixed share of contemporaneous volume to properly test capacity.
        Immediately kills remaining unfilled quantity using an Immediate-or-Cancel model.
        
        participation_ceiling: float = 0.1
            The maximum share of instantaneous volume the backtest is allowed to trade at any minute.
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
        
        self.participation_ceiling = participation_ceiling
    
        super().__init__(**kwargs)
    
    def _set_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate constrained strategy positions using an Immediate-or-Cancel model.
        Executes whatever partial fill is immediately available (capped to self.participation_ceiling) and kills the remaining.

        df : pd.DataFrame
            Preprocessed market data containing base strategy indicators.

        Returns pd.DataFrame
            DataFrame containing the physically constrained actual portfolio positions.
        """
        
        df = super()._set_positions(df)
        df["target"] = df["position"]
        
        # maximum leverage factor allowed
        df["capacity"] = (df["volume"] * df["close"] * self.participation_ceiling) / self.aum
        
        signal_mask = df["target"] != df["target"].shift(1)
        signal_mask.iloc[0] = True
        
        # signal events, update actual positions at each leverage-change event
        targets, caps = df.loc[signal_mask, ["target", "capacity"]].to_numpy().T
        
        p_current = 0
        actual_fills = np.zeros(len(targets))
        
        for i in range(len(targets)):
            t = targets[i]
            c = caps[i]
            
            if t > p_current:
                p_current = min(t, p_current + c)
            elif t < p_current:
                p_current = max(t, p_current - c)
                
            actual_fills[i] = p_current
        
        df["position"] = np.nan
        df.loc[signal_mask, "position"] = actual_fills
        df["position"] = df["position"].ffill()
        
        return df.drop(columns=["target", "capacity"])
