from __future__ import annotations

import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from .base import AnalysisReport
from ..utils import save_figures, _safe_div

class CapacityEstimator(AnalysisReport):
    def __init__(self, portfolio: Portfolio, max_delay: int = 60, decay_threshold: float = 0.5) -> None:
        """
        Estimate a rough capacity bound from how fast the strategy's edge decays.

            capacity ~= (minutes the edge survives) x (volume / minute) x (participation rate) x (price)

        participation rate is not an input: it is the largest realised |delta position| x aum / (close x volume)
        across bars, i.e. whatever share of volume the portfolio's own execution model (or lack of one) already consumed.

        The window is measured from the backtest by delaying entry (not exit) by 0...max_delay minutes.
        The first delay which delayed return / base return <= decay_threshold (or flips sign), is taken as the window in minutes.

        Cannot be meaningfully used with rollover execution models.

        portfolio : Portfolio
            Backtested portfolio instance.
        max_delay : int = 60
            Maximum entry delay (in minutes/bars).
        decay_threshold : float = 0.5
            Fraction of the portfolio return at which the edge is considered to have been lost.
        """
        
        assert 0 <= decay_threshold < 1
        
        self.portfolio = portfolio
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

        self.avg_price = float(df["close"].mean())
        self.avg_volume_per_min = float(df["volume"].mean())

        delta_position = position.diff().abs().fillna(0)
        realised_participation = (delta_position * portfolio.aum / (df["close"] * df["volume"])).replace([np.inf, -np.inf], np.nan).fillna(0)
        self.participation_rate = float(realised_participation.max())

        self.capacity_bound = self.window * self.avg_volume_per_min * self.participation_rate * self.avg_price

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

    def plot(self, *, savepath: Optional[str | Path] = None) -> None:
        """
        Plot the return decay curve and the execution/cost-modelled Sharpe curve against entry delay.

        savepath : Optional[str | Path] = None
            If given, this run's figure is saved under a new f"{ClassName}_{timestamp}"
            directory inside savepath, then closed. Still displayed either way.
        """

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1]})

        ax1.plot(self.delays, self.decay_ratio, linewidth=1, label="Delayed / Base Return Ratio")
        ax1.axhline(self.decay_threshold, color="gray", linestyle="--", linewidth=1, label=f"Decay Threshold ({self.decay_threshold:.0%})")
        ax1.axvline(self.window, color="red", linestyle=":", linewidth=1, label=f"Estimated Window ({self.window} min)")

        ax1.set_title("Returns Decay vs. Entry Delay", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Decay Ratio", fontsize=10)
        ax1.margins(x=0)
        ax1.legend(loc="upper right", fontsize=9)

        ax2.plot(self.delays, self.sharpe_curve, linewidth=1, label="Ann. Sharpe")
        ax2.axvline(self.window, color="red", linestyle=":", linewidth=1)

        ax2.set_title("Sharpe vs. Entry Delay", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Entry Delay (Minutes)", fontsize=10)
        ax2.set_ylabel("Annualised Sharpe", fontsize=10)
        ax2.margins(x=0)
        ax2.legend(loc="upper right", fontsize=9)

        ax1.set_xlim(self.delays.min(), self.delays.max())

        plt.tight_layout()
        plt.show()

        if savepath is not None:
            save_figures({"alpha_decay_and_sharpe": fig}, type(self).__name__, savepath)

        plt.close(fig)

    def __str__(self) -> str:
        """
        Format the decay-derived capacity estimate and modelled Sharpe curve as a short summary.
        """

        lines = [
            f"Alpha Decay Window: {self.window:,} min(s) (decays to {self.decay_threshold:.0%} of immediate-entry return)",
            f"Peak Participation Rate: {self.participation_rate:.1%}",
            f"Avg. Volume / Min: {self.avg_volume_per_min:,.0f} shares",
            f"Est. Capacity Bound: ${self.capacity_bound:,.0f}",
            f"Sharpe @ Base: {self.sharpe_curve[0]:,.2f}",
            f"Sharpe @ Window <{self.window} min>: {self.sharpe_curve[self.window]:,.2f}",
        ]

        return "\n".join(lines)
