from __future__ import annotations

import pandas as pd
import numpy as np
import numba
from .base import BacktestContext, StrategyComponent
from .numba_kernel import _resolve_positions

class BaseStrategy(StrategyComponent):
    def set(self, df: pd.DataFrame, ctx: BacktestContext) -> pd.DataFrame:
        """
        Default noise-area breakout strategy.
        Entries and exits are evaluated semi-hourly, and exposure is scaled to a volatility target.

        df : pd.DataFrame
            Preprocessed market data containing strategy indicators.
        ctx : BacktestContext
            Shared portfolio parameters for the backtest run.

        Returns pd.DataFrame
            DataFrame containing generated portfolio positions.
        """

        # signal generation
        intervals = df.index.minute.isin([0, 30])

        long_entry = (df["close"] > df["upper_bound"]) & intervals & ctx.long_perm
        short_entry = (df["close"] < df["lower_bound"]) & intervals & ctx.short_perm

        long_exit = (df["close"] < df["long_stop"]) & intervals
        short_exit = (df["close"] > df["short_stop"]) & intervals

        end_of_day = df.index.time == pd.Timestamp("15:59").time()

        df["position"] = _resolve_positions(np.asarray([long_entry, short_entry, long_exit, short_exit, end_of_day]).T) * (ctx.target_vol / df["std"]).clip(lower=-ctx.max_leverage, upper=ctx.max_leverage)

        return df

class RollingImmediateStopStrategy(StrategyComponent):
    def __init__(self, *, entry_window: int = 30) -> None:
        """
        Initialise a rolling confirmation strategy variant with immediate stops.

        Entry signals require the boundary condition to persist for the confirmation window and exits trigger immediately.

        entry_window : int = 30
            Number of observations required to confirm an entry signal.
        """

        self.conf = entry_window

    def set(self, df: pd.DataFrame, ctx: BacktestContext) -> pd.DataFrame:
        """
        Generate positions using rolling entry confirmation and immediate exits.

        df : pd.DataFrame
            Preprocessed market data containing strategy indicators.
        ctx : BacktestContext
            Shared portfolio parameters for the backtest run.

        Returns pd.DataFrame
            DataFrame containing generated portfolio positions.
        """

        # entry signal (rolling confirmation)
        raw_long_entry = df["close"] > df["upper_bound"]
        raw_short_entry = df["close"] < df["lower_bound"]

        long_entry = (raw_long_entry.rolling(window=self.conf).sum() == self.conf) & ctx.long_perm
        short_entry = (raw_short_entry.rolling(window=self.conf).sum() == self.conf) & ctx.short_perm

        # exit signal (triggers immediately upon crossing boundary)
        long_exit = df["close"] < df["long_stop"]
        short_exit = df["close"] > df["short_stop"]
        end_of_day = df.index.time == pd.Timestamp("15:59").time()

        df["position"] = _resolve_positions(np.asarray([long_entry, short_entry, long_exit, short_exit, end_of_day]).T) * (ctx.target_vol / df["std"]).clip(lower=-ctx.max_leverage, upper=ctx.max_leverage)

        return df

class RollingIntervalStopStrategy(StrategyComponent):
    def __init__(self, *, entry_window: int = 30) -> None:
        """
        Initialise a rolling confirmation strategy with interval stops to isolate the confirmation effect.

        Entry signals require the boundary condition to persist for the confirmation window.
        Exits are evaluated as according to the base strategy.

        entry_window : int = 30
            Number of observations required to confirm an entry signal.
        """

        self.conf = entry_window

    def set(self, df: pd.DataFrame, ctx: BacktestContext) -> pd.DataFrame:
        """
        Generate positions using rolling entry confirmation and interval exits.

        df : pd.DataFrame
            Preprocessed market data containing strategy indicators.
        ctx : BacktestContext
            Shared portfolio parameters for the backtest run.

        Returns pd.DataFrame
            DataFrame containing generated portfolio positions.
        """

        # entry signal (rolling confirmation)
        raw_long_entry = df["close"] > df["upper_bound"]
        raw_short_entry = df["close"] < df["lower_bound"]

        long_entry = (raw_long_entry.rolling(window=self.conf).sum() == self.conf) & ctx.long_perm
        short_entry = (raw_short_entry.rolling(window=self.conf).sum() == self.conf) & ctx.short_perm

        # exit signal (evaluated semi hourly)
        intervals = df.index.minute.isin([0, 30])

        long_exit = (df["close"] < df["long_stop"]) & intervals
        short_exit = (df["close"] > df["short_stop"]) & intervals
        end_of_day = df.index.time == pd.Timestamp("15:59").time()

        df["position"] = _resolve_positions(np.asarray([long_entry, short_entry, long_exit, short_exit, end_of_day]).T) * (ctx.target_vol / df["std"]).clip(lower=-ctx.max_leverage, upper=ctx.max_leverage)

        return df

class QuarterHourSampleStrategy(StrategyComponent):
    def set(self, df: pd.DataFrame, ctx: BacktestContext) -> pd.DataFrame:
        """
        Base strategy sampled at a 15 minute entry interval, while exits remain sampled every 30 minutes.

        df : pd.DataFrame
            Preprocessed market data containing strategy indicators.
        ctx : BacktestContext
            Shared portfolio parameters for the backtest run.

        Returns pd.DataFrame
            DataFrame containing generated portfolio positions.
        """

        entry_intervals = df.index.minute.isin([0, 15, 30, 45])
        exit_intervals = df.index.minute.isin([0, 30])

        long_entry = (df["close"] > df["upper_bound"]) & entry_intervals & ctx.long_perm
        short_entry = (df["close"] < df["lower_bound"]) & entry_intervals & ctx.short_perm

        long_exit = (df["close"] < df["long_stop"]) & exit_intervals
        short_exit = (df["close"] > df["short_stop"]) & exit_intervals

        end_of_day = df.index.time == pd.Timestamp("15:59").time()

        df["position"] = _resolve_positions(np.asarray([long_entry, short_entry, long_exit, short_exit, end_of_day]).T) * (ctx.target_vol / df["std"]).clip(lower=-ctx.max_leverage, upper=ctx.max_leverage)

        return df
