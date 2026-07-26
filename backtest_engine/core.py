from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import dates as mdates
import pandas as pd
import numpy as np
from datetime import date
from typing import Optional, Any

from .utils import round_date
from .analysis.tearsheet import Tearsheet
from .analysis.decomposition import PortfolioDecomposer
from .components.base import BacktestContext, StrategyComponent, ExecutionComponent, CostComponent
from .components.strategy import BaseStrategy
from .components.execution import NaiveExecution
from .components.cost_model import FlatCostModel

class Portfolio:
    def __init__(self, df: pd.DataFrame, aum: float = 100_000, target_vol: float = 0.02, long_permissions: bool = True, short_permissions: bool = True, *, strategy: Optional[StrategyComponent] = None, execution: Optional[ExecutionComponent] = None, cost_model: Optional[CostComponent] = None) -> None:
        """
        Initialise and run a complete portfolio backtest.
        Strategy is modelled from "Beat the Market" paper, attached in repo.

        Runs preprocessing, and uses composable components for signal generation, execution, and costing.

        Portfolio currently does not support capital withdrawals.

        df : pd.DataFrame
            1-minute intraday market data used for signal generation and execution.
            Must contain required OHLCV fields and datetime index.
        aum : float = 100_000
            Initial portfolio value.
        target_vol : float = 0.02
            Target volatility used for position sizing, specified in the attached paper.
        long_permissions : bool = True
            Whether long trades are allowed.
        short_permissions : bool = True
            Whether short trades are allowed.
        strategy : Optional[Strategy] = None
            Signal-generation component controlling entries, exits, and position sizing.
            Defaults to BaseStrategy (the paper's noise-area breakout).
        execution : Optional[Execution] = None
            Optional fill/capacity-constraint component applied to the strategy's target positions. 
            Defaults to None (unconstrained, immediate fills).
        cost_model : Optional[CostModel] = None
            Cost/market-impact component pricing net returns from gross returns during trades. 
            Defaults to FlatCostModel (naive per-share commission + slippage).
        """

        self.aum = aum
        self.target_vol = target_vol

        self.long_perm = long_permissions
        self.short_perm = short_permissions

        self.strategy = strategy or BaseStrategy()
        self.execution = execution or NaiveExecution()
        self.cost_model = cost_model or FlatCostModel()

        self.context = BacktestContext(
            aum=aum,
            target_vol=target_vol,
            long_perm=long_permissions,
            short_perm=short_permissions,
        )

        self.df = self._backtest(df.copy())

        self.t0 = self.df.index[0].date()
        self.t1 = self.df.index[-1].date()

        self.stats = self._aggregate()

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare raw market data for backtesting.

        Calculates daily/rolling statistics and noise bands used by the strategy.

        df : pd.DataFrame
            Cleaned intraday OHLCV market data.

        Returns pd.DataFrame
            DataFrame containing original data and derived indicators.
        """

        pv = df["volume"] * (df["high"] + df["low"] + df["close"]) / 3 
        df["vwap"] = pv.groupby(df.index.date).cumsum() / df["volume"].groupby(df.index.date).cumsum()
    
        df["daily_open"] = df.groupby(pd.Grouper(freq="D"))["open"].transform("first")
    
        closes = df.groupby(df.index.date)["close"].last()
        df["prev_close"] = pd.Series(df.index.date, index=df.index).map(closes.shift(1))

        # for position sizing
        returns = closes.pct_change()

        df["std"] = pd.Series(df.index.date, index=df.index).map(returns.rolling(window=14).std().shift(1))
        df["ret"] = df["close"].pct_change()

        # strategy specific prep - noise area
        df["deviation"] = ((df["close"] / df["daily_open"]) - 1).abs() # "move"
        df["sigma"] = df.groupby("time")["deviation"].transform(lambda x: x.shift(1).rolling(14, min_periods=14).mean())

        df["upper_bound"] = df[["daily_open", "prev_close"]].max(axis=1) * (1 + df["sigma"])
        df["lower_bound"] = df[["daily_open", "prev_close"]].min(axis=1) * (1 - df["sigma"])

        df["long_stop"] = df[["upper_bound", "vwap"]].max(axis=1)
        df["short_stop"] = df[["lower_bound", "vwap"]].min(axis=1)

        return df

    def _backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the complete backtest simulation.

        Applies preprocessing, signal generation, execution, and costing to get net returns and equity curves.

        Preprocessing needs to be subclassed, rather than composed.

        df : pd.DataFrame
            Raw market data used for backtesting.

        Returns pd.DataFrame
            Dataframe with backtested strategy and intermediate logic.
        """

        df = self._preprocess(df)
        
        df = self.strategy.set(df, self.context)
        df = self.execution.fill(df, self.context)

        df["gross_ret"] = df["position"].shift(1).fillna(0) * df["ret"]
        df = self.cost_model.expense(df, self.context)

        df["cum_ret"] = (1 + df["net_ret"].fillna(0)).cumprod()
        df["equity_curve"] = self.aum * df["cum_ret"]
        df["benchmark"] = (1 + df["ret"].fillna(0)).cumprod() * self.aum

        df = df.dropna(subset=["close", "volume", "ret", "position", "net_ret"])
        self.cost_model.trim(df.index)
        self.execution.trim(df.index)

        return df

    def returns_matrix(self, aum: np.ndarray) -> np.ndarray:
        """
        Generate returns across multiple portfolio sizes.

        Capacity-constrained execution makes positions AUM-dependent, so fills are recomputed per
        AUM (vectorised, via execution.fill_matrix) rather than reusing the single fill this
        instance was constructed with.

        aum : np.ndarray
            Portfolio capital values to evaluate.

        Returns np.ndarray
            Matrix of returns with one column per AUM value.
        """

        aum = np.asarray(aum, dtype=float)
        ret = self.df["ret"].to_numpy()[:, None]

        position_matrix = self.execution.fill_matrix(aum)
        gross_matrix = np.vstack([np.zeros((1, len(aum))), position_matrix[:-1]]) * ret

        return self.cost_model.returns_matrix(aum, position_matrix, gross_matrix)

    @classmethod
    def sharpe_curve(cls, df: pd.DataFrame, min_aum: int = 1e4, max_aum: int = 1e12 , base_aum: int = None, **kwargs: Any) -> None:
        """
        Plot strategy Sharpe ratio across different portfolio capacities.

        df : pd.DataFrame
            Raw market data used to initialise a portfolio instance.
        min_aum : int = 1e4 (10 k)
            Minimum portfolio size evaluated.
        max_aum : int = 1e12 (1 tr)
            Maximum portfolio size evaluated.
        base_aum : int = None
            Reference portfolio size used for comparison.
        **kwargs : Any
            Additional parameters passed to instantiate the class during construction.
            Can include strategy=/execution=/cost_model= to evaluate a specific composition.
        """

        base_aum = min_aum if base_aum is None else np.clip(base_aum, min_aum, max_aum)
        aum = np.logspace(np.log10(min_aum), np.log10(max_aum), num=int(np.log10(max_aum) - np.log10(min_aum)) * 9 + 1) # assumes 9 points per power of 10, even AUMs only when bounded by 1eX

        if not np.any(np.isclose(aum, base_aum, rtol=1e-9)):
            aum = np.sort(np.append(aum, base_aum))

        instance = cls(df=df, aum=base_aum if base_aum is not None and min_aum <= base_aum <= max_aum else min_aum, **kwargs)
        matrix = instance.returns_matrix(np.array(aum))

        means = np.nanmean(matrix, axis=0)
        stds = np.nanstd(matrix, axis=0)
        sharpes = np.where(stds != 0, (means / stds) * np.sqrt(252 * 390), 0) # annualise from minutes

        base_sharpe = sharpes[np.argmin(np.abs(aum - base_aum))]

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(aum, sharpes, color="blue", label="Sharpe")
        ax.axhline(base_sharpe, color="gray", linestyle="--", linewidth=1, label=f"Base Sharpe ({base_sharpe:.2f})")
        ax.axhline(base_sharpe * 0.5, color="red", linestyle="--", linewidth=1, label=f"50% Base ({base_sharpe * 0.5:.2f})")
        ax.axhline(0, color="black", linestyle="-", linewidth=1, label="Risk Free")

        ax.set_xscale("log")
        ax.set_xlim(min(aum), max(aum))
        ax.set_xlabel("AUM ($)")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title(f"{cls.__name__} Capacity: Sharpe vs AUM", fontweight="bold")

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend()

        plt.tight_layout()
        plt.show()

    def _aggregate(self) -> pd.DataFrame:
        """
        Aggregate intraday results into daily performance statistics for internal reporting.

        Returns pd.DataFrame
            Daily strategy and benchmark statistics.
        """

        strat_equity = self.df["equity_curve"].groupby(self.df.index.date).last()
        bench_equity = self.df["benchmark"].groupby(self.df.index.date).last()

        # daily returns
        strat_ret = strat_equity.pct_change()
        strat_ret.iloc[0] = (strat_equity.iloc[0] / self.aum) - 1

        bench_ret = bench_equity.pct_change()
        bench_ret.iloc[0] = (bench_equity.iloc[0] / self.aum) - 1

        # drawdown
        strat_peak = np.maximum.accumulate(strat_equity.values)
        strat_dd = pd.Series((strat_equity.values - strat_peak) / strat_peak, index=strat_equity.index)

        bench_peak = np.maximum.accumulate(bench_equity.values)
        bench_dd = pd.Series((bench_equity.values - bench_peak) / bench_peak, index=bench_equity.index)

        # trade count, only count entries and flips, and not position changes on the same side
        trade_count = (
            ((self.df["position"] != 0) & (np.sign(self.df["position"]) != np.sign(self.df["position"].shift(fill_value=0))))
            .groupby(self.df.index.date).sum().astype(int)
        )

        return pd.DataFrame({
            "strat_equity": strat_equity,
            "bench_equity": bench_equity,
            "strat_ret": strat_ret,
            "bench_ret": bench_ret,
            "strat_dd": strat_dd,
            "bench_dd": bench_dd,
            "trade_count": trade_count,
        })

    @property
    def sharpe(self) -> float:
        """
        Calculate the annualised strategy Sharpe ratio.
        Uses aggregated daily strategy returns without using a risk-free rate.

        Returns float
            Annualised Sharpe ratio.
        """

        return float((r := self.stats["strat_ret"]).mean() / r.std() * 252**0.5)

    def report(self, *, day: Optional[date] = None, start: Optional[date] = None, end: Optional[date] = None, plot: bool = True, decompose: bool = False) -> Tearsheet | PortfolioDecomposer:
        """
        Generate a portfolio performance report.
        Supports analysis for a single-day slice, period performance, and optional portfolio decomposition into long/short.

        day : date = None
            Single trading day to analyse, prioritised over provided start/end.
        start : date = None
            Start date for period analysis. Defaults to max range and defers to day if provided.
        end : date = None
            End date for period analysis. Defaults to max range and defers to day if provided.
        plot : bool = True
            Whether to display performance charts.
        decompose : bool = False
            Whether to return portfolio decomposition analysis instead of a Tearsheet.

        Returns Tearsheet | PortfolioDecomposer
            Performance report or decomposition object.
        """

        if day is not None:
            return self._daily_result(round_date(self.df.index, day), plot=plot)

        start = round_date(self.df.index, start) if start is not None else self.t0
        end = round_date(self.df.index, end) if end is not None else self.t1

        sliced = self.stats.loc[start:end, ["strat_equity", "bench_equity"]].copy()
        sliced *= self.aum / sliced.iloc[0].values

        strategy = sliced["strat_equity"]
        bench = sliced["bench_equity"]

        if plot and not decompose:
            plt.figure(figsize=(14, 8))

            plt.plot(strategy.index, strategy.values, color="blue", label="Strategy")
            plt.plot(bench.index, bench.values, color="red", label="SPY")

            plt.margins(x=0)
            plt.xlabel("Date")
            plt.ylabel("Equity")

            ax = plt.gca()
            ticks = pd.date_range(strategy.index[0], strategy.index[-1], periods=10)
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))

            plt.legend()
            plt.title(f"Strategy Performance ({start} - {end})", fontweight="bold")
            plt.show()

        sliced = self.stats[start:end].copy()
        sliced[["strat_equity", "bench_equity"]] *= self.aum / sliced[["strat_equity", "bench_equity"]].iloc[0].values

        return Tearsheet().report(sliced, plot_returns=plot) if not decompose else PortfolioDecomposer(self).report(start, end, plot=True)

    def _daily_result(self, dt: date, plot: bool) -> Tearsheet:
        """
        Internal (strategy-specific) method to generate an intraday analysis for a single trading day.

        dt : date
            Trading date to analyse.
        plot : bool
            Whether to display intraday charts.

        Returns Tearsheet
            Daily performance summary.
        """

        plot_df = self.df.loc[str(dt)] # date string slicing

        if plot:
            fig, (ax1, ax2) = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(14, 9),
                gridspec_kw={"height_ratios": [3, 1]},
                sharex=True
            )

            ax1.fill_between(
                plot_df.index,
                plot_df["lower_bound"],
                plot_df["upper_bound"],
                color="yellow",
                alpha=0.3,
                label="Noise Area"
            )

            ax1.plot(plot_df.index, plot_df["vwap"], color="red", linestyle=":", label="VWAP")
            ax1.plot(plot_df.index, plot_df["close"], color="black", label="SPY")

            ax1.set_title("Noise Area")
            ax1.set_ylabel("SPY", fontsize=12)
            ax1.legend(loc="upper left")
            ax1.margins(x=0)

            ax2.step(
                plot_df.index,
                plot_df["position"],
                color="blue",
                where="post",
                linewidth=1.5,
                label="Leverage Factor"
            )

            ax2.set_xlabel("Time")
            ax2.set_ylabel("Leverage")
            ax2.margins(x=0)

            ticks = list(pd.date_range(plot_df.index[0], plot_df.index[-1], freq="30min")) + [plot_df.index[-1]]

            ax2.set_xticks(ticks)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=plot_df.index.tz))
            ax2.set_xlim(ticks[0], ticks[-1])

            plt.suptitle(f"Strategy Performance ({dt})", fontweight="bold")
            plt.tight_layout()
            plt.show()

        t = Tearsheet()

        t.strat_cum_return = self.stats.loc[dt]["strat_ret"]
        t.bench_cum_return = self.stats.loc[dt]["bench_ret"] # includes overnight (i.e. from prev close)

        return t

    def __str__(self) -> str:
        """
        Return a short portfolio summary.
        Includes portfolio size, Sharpe ratio, and backtest period.

        Returns str
            Formatted portfolio description.
        """

        return f"{__class__.__name__}(AUM: ${self.aum:,.0f}, Sharpe: {self.sharpe:.2f}, Period: [{self.t0} - {self.t1}])"
