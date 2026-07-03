from __future__ import annotations

import statsmodels.api as sm
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from typing import Optional
from datetime import date
from .tearsheet import Tearsheet
    
class PortfolioDecomposer:
    def __init__(self, portfolio: Portfolio) -> None:
        self.portfolio = portfolio
        
        self.aum = self.portfolio.aum
        self.df = self.portfolio.df
        
        self.component_tearsheets = None
        self.decomposition = None
        
    def generate(self, start_date: date, end_date: date, plot: bool = True) -> PortfolioDecomposer:
        df = self.df.loc[str(start_date): str(end_date)]

        # long/short split
        positions = pd.concat(
            [df["position"].clip(lower=0), df["position"].clip(upper=0)],
            axis=1,
            keys=["long", "short"]
        )
        
        gross_ret = positions.shift(1).fillna(0).mul(df["ret"], axis=0)
        
        # cost split - distributed proportionally when trade flips
        delta = positions.diff().abs().fillna(0)
        
        costs = (
            delta
            .div(delta.sum(axis=1), axis=0).fillna(0) # proportion l/s of total flip/position change
            .mul(df["gross_ret"] - df["net_ret"], axis=0) # multiply by modelled cost
        )
        
        net_ret = gross_ret - costs
        
        # prepare tearsheets for each component
        def build(ret: pd.Series, pos: pd.Series) -> pd.DataFrame:
            equity = (1 + ret).cumprod().groupby(df.index.date).last()
            
            daily_ret = equity.pct_change().fillna(0)
            daily_ret.iloc[0] = equity.iloc[0] - 1
            
            peak = np.maximum.accumulate(equity.values)
            dd = pd.Series((equity.values - peak) / peak, index=equity.index)

            trade_count = (
                ((pos != 0) & (np.sign(pos) != np.sign(pos.shift(fill_value=0))))
                .groupby(df.index.date).sum().astype(int)
            )
            
            return pd.DataFrame({
                "strat_ret": daily_ret,
                "strat_dd": dd,
                "trade_count": trade_count
            })
            
        append_bench = lambda df: pd.concat([df, self.portfolio.stats.loc[start_date: end_date, ["bench_ret", "bench_dd"]]], axis=1)
           
        self.component_tearsheets = [ 
            Tearsheet().generate(self.portfolio.stats.loc[start_date: end_date], plot_returns=False), # strat
            Tearsheet().generate(append_bench(_l := build(net_ret["long"], positions["long"])), plot_returns=False), # long
            Tearsheet().generate(append_bench(_s := build(net_ret["short"], positions["short"])), plot_returns=False), # short
        ]
        
        self.decomposition = [] # NOTE: needs to be updated for new Portfolio children!! 
        
        if plot:
            self._plot_decomposition(
                pd.concat(
                    [
                        self.portfolio.stats.loc[start_date: end_date, ["strat_ret", "bench_ret"]], 
                        _l["strat_ret"].rename("long"),
                        _s["strat_ret"].rename("short"),
                    ],
                    axis=1,
                ).fillna(0).rename(
                    columns={
                        "strat_ret": "Strategy",
                        "bench_ret": "Benchmark",
                        "long": "Long-Only",
                        "short": "Short-Only",
                    }
                )
            )
            
        return self
    
    def _plot_decomposition(self, ret: pd.DataFrame) -> None:
            fig_size = (14, 10)
            
            fig, ax = plt.subplots(figsize=fig_size)
            
            colours = {
                "Strategy": "#1f77b4", 
                "Long-Only": "#2ca02c", 
                "Short-Only": "#d62728", 
                "Benchmark": "#ff7f0e"
            }
            
            for col in ret.columns:
                equity_curve = (1 + ret[col]).cumprod() * self.aum
                ax.plot(equity_curve.index, equity_curve.values, label=col, color=colours[col], linewidth=1)
                
            ax.set_title("Portfolio Decomposition Equity Curves", fontsize=12, fontweight="bold")
            ax.set_xlabel("Date", fontsize=10)
            ax.set_ylabel("Portfolio Value ($)", fontsize=10)
            ax.legend(loc="upper left", frameon=False)
            
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
            ax.set_xlim(ret.index.min(), ret.index.max())
            ax.margins(x=0)
            
            plt.tight_layout()
            plt.show()
            
            fig = plt.figure(figsize=fig_size)
            gs = fig.add_gridspec(2, 2, top=0.91, bottom=0.07, left=0.07, right=0.93, wspace=0.20, hspace=0.20)
            
            fig.suptitle("Daily Returns Distributions vs Benchmark (SymLog-Scaled)", fontsize=14, fontweight="bold", y=0.96)
            
            hist_configs = [
                {"label": "Strategy vs Benchmark", "series_col": "Strategy", "pos": gs[0, 0]},
                {"label": "Long-Only vs Benchmark", "series_col": "Long-Only", "pos": gs[0, 1]},
                {"label": "Short-Only vs Benchmark", "series_col": "Short-Only", "pos": gs[1, 0]}
            ]
            
            for config in hist_configs:
                ax_sub = fig.add_subplot(config["pos"])
                s_col = config["series_col"]
                
                s_data = ret[s_col]
                b_data = ret["Benchmark"]
                
                xlim = max(s_data.abs().std(), b_data.abs().std()) * 4.5
                bins_edges = np.linspace(-xlim, xlim, 51)
                
                ax_sub.hist(s_data.values, bins=bins_edges, color=colours[s_col], alpha=0.4, histtype="stepfilled", density=True, align="mid", label=s_col)
                ax_sub.hist(b_data.values, bins=bins_edges, color=colours["Benchmark"], alpha=0.25, histtype="stepfilled", density=True, align="mid", label="Benchmark")
                
                peak_s = np.histogram(s_data.values, bins=bins_edges, density=True)[0].max()
                peak_b = np.histogram(b_data.values, bins=bins_edges, density=True)[0].max()
                thresh = max(max(peak_s, peak_b) * 0.1, 1e-5)
                ax_sub.set_yscale("symlog", linthresh=thresh)
                
                ax_sub.set_title(config["label"], fontsize=11, fontweight="bold")
                ax_sub.set_xlim(-xlim, xlim)
                ax_sub.margins(x=0)
                ax_sub.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
                ax_sub.legend(loc="upper left", frameon=False, fontsize=9)
                
                stats_text = (
                    f"[{s_col}]\n"
                    f"Mean: {s_data.mean()*100:.3f}%\n"
                    f"Std:  {s_data.std()*100:.2f}%\n"
                    f"Skew: {s_data.skew():.2f}\n"
                    f"Kurt: {s_data.kurt():.2f}\n\n"
                    f"[Benchmark]\n"
                    f"Mean: {b_data.mean()*100:.3f}%\n"
                    f"Std:  {b_data.std()*100:.2f}%\n"
                    f"Skew: {b_data.skew():.2f}\n"
                    f"Kurt: {b_data.kurt():.2f}"
                )
                ax_sub.text(
                    0.96, 0.96, stats_text, 
                    transform=ax_sub.transAxes, 
                    fontsize=8, 
                    fontfamily="monospace",
                    horizontalalignment="right", 
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="lightgray")
                )
                
            ax_kde = fig.add_subplot(gs[1, 1])
            
            global_xlim = ret["Strategy"].abs().std() * 4.5
            xs = np.linspace(-global_xlim, global_xlim, 500)
            
            for col in ret.columns:
                kde = gaussian_kde(ret[col].values)
                ys = kde(xs)
                ax_kde.plot(xs, ys, label=col, color=colours[col], linewidth=1)
                
            ax_kde.set_title("Overlayed (Continuous, Gaussian KDE)", fontsize=11, fontweight="bold")
            ax_kde.set_xlabel("Daily Return", fontsize=10)
            ax_kde.set_ylabel("Probability Density", fontsize=10)
            ax_kde.set_xlim(-global_xlim, global_xlim)
            ax_kde.margins(x=0)
            ax_kde.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))
            ax_kde.legend(loc="upper right", frameon=False, fontsize=9)
            
            plt.show()
        
    def __str__(self) -> str:
        try:
            ts_strat, ts_long, ts_short = self.component_tearsheets
        
        except:
            return "Empty Portfolio"
            
        w_metric = 25
        w_col = 15
        
        header = (
            f"{"":<{w_metric}}"
            f"{"Strategy":>{w_col}}"
            f"{"Strat_Long":>{w_col}}"
            f"{"Short-Only":>{w_col}}"
            f"{"Benchmark":>{w_col}}"
        )
        lines = [header, "-" * (w_metric + w_col * 4)]
        
        for metric in Tearsheet._METRICS:
            vals_strat = ts_strat._data[metric]
            vals_long = ts_long._data[metric]
            vals_short = ts_short._data[metric]
            
            if len(vals_strat) == 0:
                continue
                
            s_strat = ts_strat._format_val(metric, vals_strat[0])
            s_long  = ts_long._format_val(metric, vals_long[0])
            s_short = ts_short._format_val(metric, vals_short[0])
            s_bench = ts_strat._format_val(metric, vals_strat[1]) if len(vals_strat) > 1 else "-"
            
            lines.append(
                f"{metric:<{w_metric}}"
                f"{s_strat:>{w_col}}"
                f"{s_long:>{w_col}}"
                f"{s_short:>{w_col}}"
                f"{s_bench:>{w_col}}"
            )
            
        return "\n".join(lines)