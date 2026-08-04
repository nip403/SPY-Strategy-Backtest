from __future__ import annotations

import pandas as pd
import numpy as np
from .base import BacktestContext, ExecutionComponent

class NaiveExecution(ExecutionComponent):
    def fill(self, df: pd.DataFrame, ctx: BacktestContext, cache: dict) -> pd.DataFrame:
        """
        Naive execution filler, assumes the entire desired target trade is executed when the strategy demands.
        """

        cache["position"] = df["position"]

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

    def fill(self, df: pd.DataFrame, ctx: BacktestContext, cache: dict) -> pd.DataFrame:
        """
        Constrain target positions to a fixed % of minute volume using rollover.
        Accumulates tradeable volume and continuously fills orders as liquidity and signals allow.

        Caches AUM-free components and wraps fill_matrix.

        df : pd.DataFrame
            Market data containing the "position" column produced by a Strategy.
        ctx : BacktestContext
            Shared portfolio parameters for the backtest run.
        cache : dict
            This Portfolio's cache.

        Returns pd.DataFrame
            DataFrame containing the physically constrained actual portfolio positions.
        """

        target = df["position"]
        close = cache.setdefault("close", df["close"])
        volume = cache.setdefault("volume", df["volume"])
        raw_capacity = volume * close * self.participation_ceiling # capacity, AUM not yet divided in

        # split backtest into regimes (that have the same target); increment every time target changes.
        prev = target.shift(1)
        mask = (target != prev) & ~(target.isna() & prev.isna())
        mask.iloc[0] = True

        cache["rollover"] = pd.DataFrame({"target": target, "raw_capacity": raw_capacity, "regime": mask.cumsum()})

        df["position"] = self.fill_matrix(np.array([ctx.aum]), cache)[:, 0]

        return df

    def fill_matrix(self, aum: np.ndarray, cache: dict) -> np.ndarray:
        """
        Vectorised rollover fill across multiple AUM values.
        Regime-level greedy resolution for AUM-created path dependency.

        aum : np.ndarray
            Portfolio capital values to evaluate.
        cache : dict
            This Portfolio's cache, as populated by fill().

        Returns np.ndarray
            Matrix of positions with one column per AUM value.
        """

        aum = np.asarray(aum, dtype=float)

        rollover = cache["rollover"]
        target = rollover["target"].to_numpy()
        raw_capacity = rollover["raw_capacity"].to_numpy()

        # 0-indexed regime ids
        raw_regime = rollover["regime"].to_numpy()
        regime = np.cumsum(np.concatenate([[True], raw_regime[1:] != raw_regime[:-1]])) - 1

        raw = pd.Series(raw_capacity).groupby(regime).cumsum().to_numpy()  # O(rows), one column
        cum_cap = np.divide(raw[:, None], aum[None, :], out=np.zeros((len(raw_capacity), len(aum))), where=(aum[None, :] > 0)) # cumulative capacity per regime

        # regime-level summary for path-dependent starts/ends
        regime_target = pd.Series(target).groupby(regime).first().to_numpy() # (R,)
        regime_maxcap = pd.DataFrame(cum_cap).groupby(regime).last().to_numpy() # (R, A)

        p_start = np.empty_like(regime_maxcap)
        p_current = np.zeros_like(aum)

        for i, t in enumerate(regime_target): # O(regimes), vectorised across AUM
            p_start[i] = p_current

            # greedily resolving regime endpoints
            p_current = np.where(
                t > p_current, np.minimum(t, p_current + regime_maxcap[i]),
                np.where(t < p_current, np.maximum(t, p_current - regime_maxcap[i]), p_current),
            )

        p_start_bar = p_start[regime] # broadcast regime-level starts back to bar level
        target_col = target[:, None]

        # matching and clipping minute by minute
        return p_start_bar + np.clip(target_col - p_start_bar, -cum_cap, cum_cap)

class CappedVolumeExecution(ExecutionComponent):
    def __init__(self, *, participation_ceiling: float = 0.1) -> None:
        """
        An Immediate-or-Cancel fill model constrained to a fixed % of minute volume.

        participation_ceiling : float = 0.1
            The maximum share of instantaneous volume the backtest is allowed to trade at any minute.
        """

        self.participation_ceiling = participation_ceiling

    def fill(self, df: pd.DataFrame, ctx: BacktestContext, cache: dict) -> pd.DataFrame:
        """
        Constrain target positions using an Immediate-or-Cancel model.

        Executes whatever partial fill is immediately available (capped to
        self.participation_ceiling) and kills the remaining.

        Caches AUM-free ingredients (target, raw per-minute capacity, signal mask) and delegates
        the actual fill to fill_matrix, so the single-AUM path and the vectorised AUM-sweep path
        can never drift out of sync.

        df : pd.DataFrame
            Market data containing the "position" column produced by a Strategy.
        ctx : BacktestContext
            Shared portfolio parameters for the backtest run.
        cache : dict
            This Portfolio's cache.

        Returns pd.DataFrame
            DataFrame containing the physically constrained actual portfolio positions.
        """

        target = df["position"]
        close = cache.setdefault("close", df["close"])
        volume = cache.setdefault("volume", df["volume"])
        raw_capacity = volume * close * self.participation_ceiling # capacity, AUM not yet divided in

        signal_mask = target != target.shift(1)
        signal_mask.iloc[0] = True

        cache["ioc"] = pd.DataFrame({"target": target, "raw_capacity": raw_capacity, "signal": signal_mask})

        df["position"] = self.fill_matrix(np.array([ctx.aum]), cache)[:, 0]

        return df

    def fill_matrix(self, aum: np.ndarray, cache: dict) -> np.ndarray:
        """
        Vectorised Immediate-or-Cancel fill across multiple AUM values without rerunning fill()
        per AUM.

        The only inherently sequential part is the greedy resolution at each signal-change event;
        that loop runs once over events (not over time, not over AUM) carrying a length-len(aum)
        vector of current positions. Unfilled bars are then forward-filled.

        aum : np.ndarray
            Portfolio capital values to evaluate.
        cache : dict
            This Portfolio's cache, as populated by fill().

        Returns np.ndarray
            Matrix of positions with one column per AUM value.
        """

        aum = np.asarray(aum, dtype=float)

        ioc = cache["ioc"]
        signal_mask = ioc["signal"].to_numpy()

        targets = ioc["target"].to_numpy()[signal_mask]
        raw_caps = ioc["raw_capacity"].to_numpy()[signal_mask, None]

        caps = np.divide(raw_caps, aum, out=np.zeros((len(raw_caps), len(aum))), where=(aum > 0))

        p_current = np.zeros_like(aum)
        fills = np.empty_like(caps)

        for i, t in enumerate(targets): # O(events), vectorised across AUM
            fills[i] = np.where(
                t > p_current, np.minimum(t, p_current + caps[i]),
                np.where(t < p_current, np.maximum(t, p_current - caps[i]), p_current),
            )
            p_current = fills[i]

        out = np.full((len(signal_mask), len(aum)), np.nan)
        out[signal_mask] = fills

        # match the portfolio's actual starting state
        return pd.DataFrame(out, index=ioc.index).ffill().fillna(0).to_numpy()
