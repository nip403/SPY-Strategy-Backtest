from __future__ import annotations

import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import pandas as pd
import numpy as np
from datetime import time
from pathlib import Path
from typing import Optional
from scipy.interpolate import InterpolatedUnivariateSpline
from .base import AnalysisReport
from ..utils import save_figures, _safe_div

class CapacityEstimator(AnalysisReport):
    def __init__(self, portfolio: Portfolio, max_delay: int = 60, decay_threshold: float = 0.5) -> None:
        """
        Estimate the strategy's rough mechanical and economic capacity from how fast the edge decays.
        
        Mechanical Fill Capacity: (minutes the edge survives) x (volume / minute) x (participation rate) x (price)
            How much volume can physically be executed to capture the strategy's alpha?
            
        Economic Capacity: AUM at which mean net return crosses zero
            How much volume can be traded until the strategy is no longer worth executing?

        Costing the full target size over a single bar volume would explode cost.
        Trade regime transitions are therefore collapsed into one fill priced on the edge decay window volume and average price.

        participation rate (for fill bound) is not an input: it is the largest realised |delta position| x aum / (close x volume)

        The window is measured from the backtest by delaying entry by 0...max_delay minutes.
        The first delay which delayed return / base return <= decay_threshold (or flips sign), is taken as the window in minutes.

        The decay-window measurement above cannot be meaningfully used with rollover execution models.

        portfolio : Portfolio
            Backtested portfolio instance.
        max_delay : int = 60
            Maximum entry delay (in minutes/bars).
        decay_threshold : float = 0.5
            Fraction of the portfolio return at which the edge is considered to have been lost.
        """
        
        assert 0 <= decay_threshold < 1
        
        self.max_delay = max_delay
        self.decay_threshold = decay_threshold

        df = portfolio.df
        position = df["position"]
        ret = df["ret"]

        # entry-only delay: zero out the first d bars of each position run, exit timing remains the same
        side = np.sign(position)
        run_id = side.ne(side.shift(1, fill_value=0)).cumsum()
        runs = position.groupby(run_id).cumcount()

        self.delays = np.arange(max_delay + 1)
        self.decay_curve = np.array([float((position.where(runs >= d, 0).shift(1).fillna(0) * ret).sum()) for d in self.delays])

        if self.decay_curve[0] <= 0:
            warnings.warn("Base portfolio return is non-positive - no meaningful capacity can be derived. Terminating CapacityEstimator.")
            return

        self.decay_ratio = self.decay_curve / self.decay_curve[0]
        crossed = np.flatnonzero(self.decay_ratio <= decay_threshold)

        if len(crossed) == 0: # window is greater than provided max
            self.window = max_delay
        else:
            self.window = int(self.delays[crossed[0]])

        self.avg_price, self.avg_volume_per_min, self.participation_rate, self.capacity_bound = self._calc_fill_capacity(portfolio, df, position)
        self.economic_aum_grid, self.economic_curve, self.economic_capacity = self._calc_economic_capacity(portfolio)

        # precomputation for efficiency
        base_sub = df[["close", "volume", "ret"]].copy()
        pos, runs = position.to_numpy(), runs.to_numpy()
        dates = df.index.floor("D")
        eod_indices = np.append(np.flatnonzero(dates[:-1] != dates[1:]), len(dates) - 1)
        delayed_pos = np.where(runs[:, None] >= self.delays[None, :], pos[:, None], 0)

        def sharpe(delay: int) -> float:
            """
            Re-run the delayed position through the portfolio's execution/cost model to calculate sharpe.
            
            delay : int
                Delay in minutes.
            """
            sub = base_sub.copy(deep=False)
            sub["position"] = delayed_pos[:, delay]
            cache: dict = {}

            sub = portfolio.execution.fill(sub, portfolio.context, cache)
            sub["gross_ret"] = sub["position"].shift(1).fillna(0) * sub["ret"]
            sub = portfolio.cost_model.expense(sub, portfolio.context, cache)
            
            equity = np.cumprod(np.nan_to_num(1 + sub["net_ret"].to_numpy()))
            eod = equity[eod_indices]
            daily_ret = np.empty_like(eod)
            
            daily_ret[0] = eod[0] - 1
            daily_ret[1:] = np.diff(eod) / eod[:-1]

            return _safe_div(float(daily_ret.mean()) * 252, float(daily_ret.std()) * np.sqrt(252))

        self.sharpe_curve = np.array([sharpe(int(d)) for d in self.delays])

    def _calc_fill_capacity(self, portfolio: Portfolio, df: pd.DataFrame, position: pd.Series) -> tuple[float]:
        """
        Compute the mechanical fill capacity bound: (window) x (volume / minute) x (participation rate) x (price).

        portfolio : Portfolio
            Backtested portfolio instance.
        df : pd.DataFrame
            Market data with the strategy's "position" column.
        position : pd.Series
            The strategy's position series.

        Returns tuple[float]
            (avg_price, avg_volume_per_min, participation_rate, capacity_bound).
        """

        avg_price = float(df["close"].mean())
        avg_volume_per_min = float(df["volume"].mean())

        delta_position = position.diff().abs().fillna(0)
        realised_participation = (delta_position * portfolio.aum / (df["close"] * df["volume"])).replace([np.inf, -np.inf], np.nan).fillna(0)
        participation_rate = float(realised_participation.max())

        capacity_bound = self.window * avg_volume_per_min * participation_rate * avg_price

        return avg_price, avg_volume_per_min, participation_rate, capacity_bound

    def _eval_windows(self, df: pd.DataFrame) -> tuple[np.ndarray, dict]:
        """
        Collapse each trade regime transition into one fill, priced off the decay window's VWAP and total volume (instead of one bar).

        Normal transitions look forward, and the forced end-of-day flatten looks backward to prevent eating gaps.
        Both directions are bounded intraday and by trade regime.

        Trades fill at the window's volume-weighted centre of mass. 
        Overlapping fills (e.g. an entry + EOD close) are merged by np.maximum.accumulate into one net trade at the later bar (effectively cancelling).

        df : pd.DataFrame
            Market data with the strategy's raw "position" column already set.

        Returns tuple[np.ndarray, dict]
            (position array, cache with "close"/"volume" overridden at each fill bar).
        """

        target = df["position"].to_numpy()
        volume = df["volume"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)
        n = len(target)

        is_start = np.empty(n, dtype=bool)
        is_start[0] = True
        is_start[1:] = target[1:] != target[:-1]

        regime_starts = np.flatnonzero(is_start)
        regime_target = target[regime_starts]
        n_regimes = len(regime_starts)

        regime_length = np.diff(np.append(regime_starts, n)) # forward span
        prior_length = np.concatenate([[0], regime_length[:-1]]) # span before transition ends

        # (target=0) regime can span days so forward windows need their session-close cap too
        dates = df.index.floor("D")
        day_last = np.flatnonzero(np.append(dates[:-1] != dates[1:], True))
        session_close = day_last[np.searchsorted(day_last, regime_starts, side="left")]
        fwd_room = np.minimum(regime_length, session_close - regime_starts + 1)

        is_eod = df.index.time[regime_starts] == time(15, 59)

        w = np.maximum(np.where(is_eod, np.minimum(self.window, prior_length), np.minimum(self.window, fwd_room)), 1)

        start = np.where(is_eod, regime_starts - w + 1, regime_starts)
        end = np.where(is_eod, regime_starts + 1, regime_starts + w)

        cumvol = np.concatenate([[0.0], np.cumsum(volume)])
        cumpv = np.concatenate([[0.0], np.cumsum(close * volume)])

        total_volume = cumvol[end] - cumvol[start]
        vwap = np.divide(cumpv[end] - cumpv[start], total_volume, out=close[start], where=total_volume > 0)

        # fill bar
        mid = np.empty(n_regimes, dtype=np.int64)
        half_volume = total_volume / 2

        for i in range(n_regimes):
            s, e = start[i], end[i]
            local_cum = cumvol[s + 1:e + 1] - cumvol[s]
            j = np.searchsorted(local_cum, half_volume[i], side="left")
            mid[i] = s + min(j, e - s - 1)

        mid = np.maximum.accumulate(mid) # merge colliding fill pairs

        position = np.full(n, np.nan)
        position[mid] = regime_target
        position = pd.Series(position, index=df.index).ffill().fillna(0.0).to_numpy()

        adj_volume, adj_close = volume.copy(), close.copy()
        adj_volume[mid], adj_close[mid] = total_volume, vwap

        cache = {
            "close": pd.Series(adj_close, index=df.index),
            "volume": pd.Series(adj_volume, index=df.index),
        }

        return position, cache

    def _sweep_capacity(self, portfolio: Portfolio, resolution: int = 100) -> tuple[np.ndarray]:
        """
        Sweep AUM, costing the strategy's target position.
        Ignored the instance's execution model but uses the cost model.

        resolution : int = 100
            Number of log-spaced AUM points swept between $10k and 2x the fill capacity bound.
            Currently no functionality to enable configuration.

        Returns tuple[np.ndarray]
            (aum_grid, mean net return per AUM).
        """

        df = portfolio.strategy.set(portfolio.df.copy(), portfolio.context)
        df["position"], cache = self._eval_windows(df)

        df["gross_ret"] = df["position"].shift(1).fillna(0) * df["ret"]
        df = portfolio.cost_model.expense(df, portfolio.context, cache) # df's close/volume columns are untouched for ADV/ann_vol caching, setdefault leaves grouped close/volume untouched

        aum_grid = np.geomspace(1e4, max(self.capacity_bound, 1e5) * 2, resolution)

        position_matrix = np.broadcast_to(df["position"].to_numpy()[:, None], (len(df), len(aum_grid)))
        ret = df["ret"].to_numpy()[:, None]

        gross_matrix = np.empty(position_matrix.shape)
        gross_matrix[0, :] = 0
        gross_matrix[1:, :] = position_matrix[:-1] * ret[1:]

        net_matrix = portfolio.cost_model.returns_matrix(aum_grid, position_matrix, gross_matrix, cache) # costing done with grouped cached volume

        return aum_grid, np.mean(net_matrix, axis=0)

    def _calc_economic_capacity(self, portfolio: Portfolio) -> tuple[np.ndarray, float]:
        """
        Sweep the economic curve and locate the AUM at which it crosses zero.

        portfolio : Portfolio
            Backtested portfolio instance.

        Returns tuple[np.ndarray, np.ndarray, float]
            (aum_grid, mean net return per AUM, economic capacity).
        """

        aum_grid, means = self._sweep_capacity(portfolio)

        if np.all(means > 0):
            return aum_grid, means, float("inf")

        if np.all(means <= 0):
            return aum_grid, means, float(aum_grid[0])

        roots = InterpolatedUnivariateSpline(np.log10(aum_grid), means, k=3).roots()

        return aum_grid, means, float(10 ** roots[0]) if len(roots) else float("nan")

    def plot(self, *, savepath: Optional[str | Path] = None) -> None:
        """
        Plot the return decay & Sharpe curve, and mean net return against AUM (economic capacity sweep).

        savepath : Optional[str | Path] = None
            If given, this run's figures are saved under a new f"{ClassName}_{timestamp}"
            directory inside savepath, then closed. Still displayed either way.
        """

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 9.5))

        ax1.plot(self.delays, self.decay_ratio, linewidth=1, label="Delayed / Base Return Ratio")
        ax1.axhline(self.decay_threshold, color="gray", linestyle="--", linewidth=1, label=f"Decay Threshold ({self.decay_threshold:.0%})")
        ax1.axvline(self.window, color="red", linestyle=":", linewidth=1, label=f"Estimated Window ({self.window} min)")

        ax1.set_title("Returns Decay vs. Entry Delay", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Entry Delay (Minutes)", fontsize=10)
        ax1.set_ylabel("Decay Ratio", fontsize=10)
        ax1.margins(x=0)
        ax1.legend(loc="upper right", fontsize=9)
        ax1.set_xlim(self.delays.min(), self.delays.max())

        ax2.plot(self.delays, self.sharpe_curve, linewidth=1, label="Ann. Sharpe")
        ax2.axvline(self.window, color="red", linestyle=":", linewidth=1)

        ax2.set_title("Sharpe vs. Entry Delay", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Entry Delay (Minutes)", fontsize=10)
        ax2.set_ylabel("Annualised Sharpe", fontsize=10)
        ax2.margins(x=0)
        ax2.legend(loc="upper right", fontsize=9)
        ax2.set_xlim(self.delays.min(), self.delays.max())

        plt.tight_layout()
        plt.show()

        fig2, ax3 = plt.subplots(figsize=(12, 5))

        keep = self.economic_curve >= -2 * abs(self.economic_curve[0])

        pct_curve = self.economic_curve[keep] * 100
        max_abs = np.max(np.abs(pct_curve))
        exp = int(np.floor(np.log10(max_abs))) if max_abs > 0 else 0

        ax3.plot(self.economic_aum_grid[keep], pct_curve / 10 ** exp, linewidth=1, label="Mean Net Return")

        if np.isfinite(self.economic_capacity):
            ax3.axvline(self.economic_capacity, color="red", linestyle=":", linewidth=1, label=f"Economic Capacity (${self.economic_capacity:,.0f})")
            ax3.axhline(0, color="gray", linestyle="--", linewidth=1)

        ax3.set_xscale("log")
        ax3.xaxis.set_minor_locator(NullLocator())
        ax3.set_title("Mean Net Return vs. Strategy Size", fontsize=12, fontweight="bold")
        ax3.set_xlabel("AUM ($, log scale)", fontsize=10)
        ax3.set_ylabel(f"Mean Net Return (per bar, x1e{exp}%)", fontsize=10)
        ax3.margins(x=0)
        ax3.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        plt.show()

        if savepath is not None:
            save_figures({
                "alpha_decay_and_sharpe": fig,
                "economic_capacity": fig2,
            }, type(self).__name__, savepath)

        plt.close(fig)
        plt.close(fig2)

    def __str__(self) -> str:
        """
        Format the decay-derived capacity estimate, modelled Sharpe curve, and economic capacity as a short summary.
        """

        lines = [
            f"Alpha Decay Window: {self.window:,} min(s) (decays to {self.decay_threshold:.0%} of immediate-entry return)",
            f"Peak Participation Rate: {self.participation_rate:.1%}",
            f"Avg. Volume / Min: {self.avg_volume_per_min:,.0f} shares",
            f"Est. Fill Capacity: ${self.capacity_bound:,.0f}",
            f"Est. Economic Capacity: {f"${self.economic_capacity:,.0f}" if np.isfinite(self.economic_capacity) else "Above swept range (2x fill capacity)."}",
            f"Sharpe @ Base: {self.sharpe_curve[0]:,.2f}",
            f"Sharpe @ Window <{self.window} min>: {self.sharpe_curve[self.window]:,.2f}",
        ]

        return "\n".join(lines)
