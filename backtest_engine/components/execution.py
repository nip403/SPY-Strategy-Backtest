from __future__ import annotations

import pandas as pd
import numpy as np
from .base import BacktestContext, ExecutionComponent

class NaiveExecution(ExecutionComponent):
    def fill(self, df: pd.DataFrame, ctx: BacktestContext) -> pd.DataFrame:
        """
        Naive execution filler, assumes the entire desired target trade is executed when the strategy demands.
        """
        
        return df

class CappedVolumeRolloverExecution(ExecutionComponent):
    def __init__(self, *, participation_ceiling: float = 0.1) -> None:
        """
        Caps participation to a fixed share of contemporaneous volume to properly test capacity.
        Any unfilled quantity of the desired trade size is rolled over until it is entirely filled or a new position target is hit.

        participation_ceiling : float = 0.1
            The maximum share of instantaneous volume the backtest is allowed to trade at any minute.
        """

        self.participation_ceiling = participation_ceiling

    def fill(self, df: pd.DataFrame, ctx: BacktestContext) -> pd.DataFrame:
        """
        Constrain target positions to a fixed % of minute volume using rollover.

        Accumulates tradeable volume and continuously fills orders as liquidity and signals allow.

        df : pd.DataFrame
            Market data containing the "position" column produced by a Strategy.
        ctx : BacktestContext
            Shared portfolio parameters for the backtest run.

        Returns pd.DataFrame
            DataFrame containing the physically constrained actual portfolio positions.
        """

        df["target"] = df["position"]

        # maximum leverage factor allowed
        df["capacity"] = (df["volume"] * df["close"] * self.participation_ceiling) / ctx.aum

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

        for i, (t, mc) in enumerate(regimes.itertuples(index=False, name=None)): # O(trades)
            p_starts[i] = p_current

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

class CappedVolumeExecution(ExecutionComponent):
    def __init__(self, *, participation_ceiling: float = 0.1) -> None:
        """
        An Immediate-or-Cancel fill model constrained to a fixed % of minute volume.

        participation_ceiling : float = 0.1
            The maximum share of instantaneous volume the backtest is allowed to trade at any minute.
        """

        self.participation_ceiling = participation_ceiling

    def fill(self, df: pd.DataFrame, ctx: BacktestContext) -> pd.DataFrame:
        """
        Constrain target positions using an Immediate-or-Cancel model.

        Executes whatever partial fill is immediately available (capped to
        self.participation_ceiling) and kills the remaining.

        df : pd.DataFrame
            Market data containing the "position" column produced by a Strategy.
        ctx : BacktestContext
            Shared portfolio parameters for the backtest run.

        Returns pd.DataFrame
            DataFrame containing the physically constrained actual portfolio positions.
        """

        df["target"] = df["position"]

        # maximum leverage factor allowed
        df["capacity"] = (df["volume"] * df["close"] * self.participation_ceiling) / ctx.aum

        signal_mask = df["target"] != df["target"].shift(1)
        signal_mask.iloc[0] = True

        # signal events, update actual positions at each leverage-change event
        events = df.loc[signal_mask, ["target", "capacity"]].to_numpy()

        p_current = 0
        actual_fills = np.zeros(len(events))

        for i, (t, c) in enumerate(events): # (targets, caps)
            if t > p_current:
                p_current = min(t, p_current + c)
            elif t < p_current:
                p_current = max(t, p_current - c)

            actual_fills[i] = p_current

        df["position"] = np.nan
        df.loc[signal_mask, "position"] = actual_fills
        df["position"] = df["position"].ffill()

        return df.drop(columns=["target", "capacity"])
