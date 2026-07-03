import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import dates as mdates
import pandas as pd
import numpy as np
from datetime import date
from typing import Optional
from .utils import round_date
from .analysis.tearsheet import Tearsheet
from .analysis.decomposition import PortfolioDecomposer

class Portfolio:
    # per share frictions, naive 
    COMMISSION = 0.0035
    SLIPPAGE = 0.001
    
    def __init__(self, df: pd.DataFrame, aum: float = 100_000, target_vol: float = 0.02, long_permissions: Optional[bool] = True, short_permissions: Optional[bool] = True) -> None:
        self.aum = aum
        self.target_vol = target_vol
        self.frictions = self.COMMISSION + self.SLIPPAGE
        
        self.long_perm = long_permissions
        self.short_perm = short_permissions
        
        self.df = self._backtest(df.copy())
        
        self.t0 = self.df.index[0].date()
        self.t1 = self.df.index[-1].date()
        
        self.stats = self._aggregate()
        
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:      
        # position sizing
        closes = df.groupby(df.index.date)["close"].last()
        returns = closes.pct_change()
        
        df["mu"] = pd.Series(df.index.date, index=df.index).map(returns.rolling(window=14).mean().shift(1))
        df["std"] = pd.Series(df.index.date, index=df.index).map(returns.rolling(window=14).std().shift(1))
        df["ret"] = df["close"].pct_change()

        # noise area
        df["deviation"] = ((df["close"] / df["daily_open"]) - 1).abs() # "move"
        df["sigma"] = df.groupby("time")["deviation"].transform(lambda x: x.shift(1).rolling(14, min_periods=14).mean())

        df["upper_bound"] = df[["daily_open", "prev_close"]].max(axis=1) * (1 + df["sigma"])
        df["lower_bound"] = df[["daily_open", "prev_close"]].min(axis=1) * (1 - df["sigma"])

        df["long_stop"] = df[["upper_bound", "vwap"]].max(axis=1)
        df["short_stop"] = df[["lower_bound", "vwap"]].min(axis=1)
        
        return df
        
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

        # set positions & backtest
        df.loc[long_exit | short_exit | end_of_day, "position"] = 0
        df.loc[long_entry, "position"] = 1
        df.loc[short_entry, "position"] = -1

        df["position"] = df["position"].ffill().fillna(0) * (self.target_vol / df["std"]).clip(lower=-4, upper=4) 
        
        df["gross_ret"] = df["position"].shift(1).fillna(0) * df["ret"]

        # frictions: # net of costs: cost% = delta(position) * leverage / cost/share, incurred at the minute position changes
        df["net_ret"] = df["gross_ret"] - (df["position"].diff().abs().fillna(0) * self.frictions / df["close"])
        df["cum_ret"] = (1 + df["net_ret"].fillna(0)).cumprod()
        df["equity_curve"] = self.aum * df["cum_ret"]
        
        # visuals
        df["benchmark"] = (1 + df["ret"].fillna(0)).cumprod() * self.aum
                
        return df.dropna()
    
    def _aggregate(self) -> pd.DataFrame:
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
        return float((r := self.stats["strat_ret"]).mean() / r.std() * 252**0.5)
    
    def result(self, *, date: Optional[date] = None, start: Optional[date] = None, end: Optional[date] = None, plot: Optional[bool] = True, decompose: Optional[bool] = False) -> Tearsheet | PortfolioDecomposer: 
        """
        date: prioritised, displays noise area, trades, and stats for a given date
        start/end: ranges for displaying backtest results. default to max range
        """
        
        if date is not None:
            return self._daily_result(round_date(self.df.index, date), plot=plot)
            
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
        
        return Tearsheet().generate(sliced, plot_returns=plot) if not decompose else PortfolioDecomposer(self).generate(start, end, plot=True)
        
    def _daily_result(self, dt: date, plot: bool) -> Tearsheet:
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

            plt.subplots_adjust(hspace=0)
            plt.suptitle(f"Strategy Performance ({dt})", fontweight="bold")
            plt.tight_layout()
            plt.show()
        
        t = Tearsheet()
        
        t.strat_cum_return = self.stats.loc[dt]["strat_ret"]
        t.bench_cum_return = self.stats.loc[dt]["bench_ret"] # includes overnight (i.e. from prev close)
        
        return t
        
    def __str__(self) -> str:
        return f"{__class__.__name__}(AUM: {self.aum}, Sharpe: {self.sharpe}, Period: [{self.t0} - {self.t1}])"