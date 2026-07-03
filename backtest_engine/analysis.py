from __future__ import annotations

import statsmodels.api as sm
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from typing import Optional
from datetime import date

def sharpe_curve(df: pd.DataFrame, portfolio: Portfolio, aum: Optional[list[int]] = None, base: Optional[int] = None) -> None:
    aum = aum or [i * (10 ** k) for k in range(4, 12) for i in range(1, 10)] + [1e12]
    sharpes = [portfolio(df, i).sharpe for i in aum]
    
    base = sharpes[aum.index(base) if base in aum else 0] # base: baseline/benchmark portfolio (in $ aum)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(aum, sharpes, color="blue", label="Sharpe")

    ax.axhline(base, color="gray", linestyle="--", linewidth=1, label=f"Base Sharpe ({base:.2f})")
    ax.axhline(base * 0.5, color="red", linestyle="--", linewidth=1, label=f"50% Base ({base * 0.5:.2f})")
    ax.axhline(0, color="black", linestyle="-", linewidth=1, label="Risk Free")

    ax.set_xscale("log")
    ax.set_xlim(min(aum), max(aum))

    ax.set_xlabel("AUM ($)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Strategy Capacity: Sharpe vs AUM", fontweight="bold")

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    
    plt.tight_layout()
    plt.show()

class Tearsheet:
    _METRICS = [
        "Total Days", 
        "Cum. Return", "Ann. Return", "Avg. Daily Return", "Ret. Skew", "Ret. Kurtosis",
        "Max Gain", "Best Day", "Max Loss", "Worst Day", "Win Rate", "Daily Win Rate",
        "Average Trades / Day", "Average Return / Trade",# "Average PnL / Share",
        "Alpha", "Beta", "Ann. Volatility", 
        "Max Drawdown", "Max DD Days", "95% VaR", 
        "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Information Ratio"
    ]
    
    _ATTRS = {
        "total_days": "Total Days",
        "cum_return": "Cum. Return",
        "ann_return": "Ann. Return",
        "avg_daily_return": "Avg. Daily Return",
        "skew": "Ret. Skew", 
        "kurt": "Ret. Kurtosis",
        "max_gain": "Max Gain",
        "best_day": "Best Day",
        "max_loss": "Max Loss",
        "worst_day": "Worst Day",
        "win_rate": "Win Rate",
        "daily_win_rate": "Daily Win Rate",
        "trades_per_day": "Average Trades / Day",
        "return_per_trade": "Average Return / Trade",
        #"pnl_per_share": "Average PnL / Share",
        "alpha": "Alpha",
        "beta": "Beta",
        "ann_vol": "Ann. Volatility",
        "max_drawdown": "Max Drawdown",
        "max_dd_days": "Max DD Days",
        "var_95pct": "95% VaR",
        "sharpe_ratio": "Sharpe Ratio",
        "sortino_ratio": "Sortino Ratio",
        "calmar_ratio": "Calmar Ratio",
        "information_ratio": "Information Ratio"
    }
    
    _PCT_METRICS = {
        "Cum. Return", "Ann. Return", "Avg. Daily Return", 
        "Max Gain", "Max Loss", "Win Rate", "Daily Win Rate",
        "Ann. Volatility", "Max Drawdown", "95% VaR"
    }
    
    def __init__(self) -> None:
        self._data = {metric: [] for metric in self._METRICS}
        
    def generate(self, df: pd.DataFrame, plot_returns: Optional[bool] = True) -> Tearsheet:
        """
        df: aggregate 
        """
        
        ret = df[["strat_ret", "bench_ret"]]
        dd = df[["strat_dd", "bench_dd"]]
        days = len(df.index)
        
        self.total_days = days
        
        self.cum_return = (1 + ret).prod().values - 1
        self.ann_return = (1 + np.array(self.cum_return)) ** (252 / days) - 1 # geometric
        self.avg_daily_return = ret.mean().values # arithmetic mean
        
        self.skew = ret.skew().values.tolist()
        self.kurt = ret.kurt().values.tolist()
        
        self.max_gain = ret.max().values 
        self.best_day = ret.idxmax().values
        self.max_loss = ret.min().values
        self.worst_day = ret.idxmin().values
        self.win_rate = [df.loc[df["strat_ret"] > 0, "trade_count"].sum() / df["trade_count"].sum(), None] # simplified calculation using agg data, not accurate
        self.daily_win_rate = (ret > 0).mean().values # daily, not per trade
        
        self.trades_per_day = [df["trade_count"].mean(), None]
        self.return_per_trade = [(self.cum_return[0] / df["trade_count"].sum()), None]
        
        # OLS, risk free not used
        model = sm.OLS(df["strat_ret"], sm.add_constant(df["bench_ret"])).fit()
        self.alpha = [model.params["const"] * 252, None]
        self.beta = [model.params["bench_ret"], None]
        
        self.ann_vol = (ret.std().values) * np.sqrt(252)
        
        self.max_drawdown = dd.min().values
        self.var_95pct = np.percentile(ret, 5, axis=0)
        
        underwater = dd < 0
        state_changes = (underwater != underwater.shift()).cumsum()
        self.max_dd_days = [
            f"{int(df[col].where(underwater[col]).groupby(state_changes[col]).size().max())} Days"
            for col in ["strat_dd", "bench_dd"]
        ]
        
        self.sharpe_ratio = np.array(self.avg_daily_return) * 252 / np.array(self.ann_vol)

        ann_downside_vol = ret.apply(lambda col: col[col < 0].std()).values * np.sqrt(252)
        self.sortino_ratio = (np.array(self.ann_return) / np.where(ann_downside_vol == 0, 1e-6, ann_downside_vol)).tolist()

        abs_max_dd = np.abs(np.array(self.max_drawdown))
        self.calmar_ratio = (np.array(self.ann_return) / np.where(abs_max_dd == 0, 1e-6, abs_max_dd)).tolist()

        tracking_error = (df["strat_ret"] - df["bench_ret"]).std() * np.sqrt(252)
        self.information_ratio = [(ret["strat_ret"].mean() - ret["bench_ret"].mean()) * 252 / tracking_error, None]  # edge relative to unit benchmark risk
        
        if plot_returns:
            self._plot_returns(ret)
        
        return self
    
    def _plot_returns(self, ret: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(12, 8)) 
        
        ax.hist(ret["strat_ret"], bins=51, color="blue", alpha=0.45, label="Strategy", edgecolor="darkblue", density=True)
        ax.hist(ret["bench_ret"], bins=51, color="orange", alpha=0.45, label="Benchmark", edgecolor="darkorange", density=True)
        
        ax.set_title("Daily Returns Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Daily Return", fontsize=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.legend(loc="upper left", frameon=False)
        xlim = max(ret["strat_ret"].abs().max(), ret["bench_ret"].abs().max()) * 1.05
        ax.set_xlim(-xlim, xlim)

        stats_text = (
            "--- Strategy ---\n"
            f"Mean: {ret["strat_ret"].mean()*100:.3f}%\n"
            f"Std:  {ret["strat_ret"].std()*100:.3f}%\n"
            f"Skew: {ret["strat_ret"].skew():.3f}\n"
            f"Kurt: {ret["strat_ret"].kurt():.3f}\n\n"
            "--- Benchmark ---\n"
            f"Mean: {ret["bench_ret"].mean()*100:.3f}%\n"
            f"Std:  {ret["bench_ret"].std()*100:.3f}%\n"
            f"Skew: {ret["bench_ret"].skew():.3f}\n"
            f"Kurt: {ret["bench_ret"].kurt():.3f}"
        )

        ax.text(
            0.97, 0.97, stats_text, 
            transform=ax.transAxes, 
            fontsize=12, 
            fontfamily="monospace",
            horizontalalignment="right", 
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.92, edgecolor="darkgray")
        )

        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.1f}%"))

        plt.tight_layout()
        plt.show()

    def __getattr__(self, name: str):
        if name in self._ATTRS:
            return self._data[self._ATTRS[name]]

        if name.startswith("strat_"):
            attr = name[6:]
            
            if attr in self._ATTRS:
                metric = self._ATTRS[attr]
                
                return self._data[metric][0] if len(self._data[metric]) > 0 else None

        elif name.startswith("bench_"):
            attr = name[6:]
            
            if attr in self._ATTRS:
                metric = self._ATTRS[attr]
                
                return self._data[metric][1] if len(self._data[metric]) > 1 else None

        raise AttributeError(f"'Tearsheet' object has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        if name in ["_METRICS", "_ATTRS", "_data"]:
            super().__setattr__(name, value)
            return

        if name.startswith("strat_"):
            attr = name[6:]
            
            if attr in self._ATTRS:
                if not len(self._data[self._ATTRS[attr]]):
                    self._data[self._ATTRS[attr]].append(value)
                else:
                    self._data[self._ATTRS[attr]][0] = value
                    
                return
                
        elif name.startswith("bench_"):
            attr = name[6:]
            
            if attr in self._ATTRS:
                match len(self._data[self._ATTRS[attr]]):
                    case 0:
                        self._data[self._ATTRS[attr]] = [None, value]
                    case 1:
                        self._data[self._ATTRS[attr]].append(value)
                    case 2:
                        self._data[self._ATTRS[attr]][1] = value
                
                return 
            
        if name in self._ATTRS:
            if isinstance(value, list):
                self._data[self._ATTRS[name]] = value
            
            elif isinstance(value, np.ndarray):
                self._data[self._ATTRS[name]] = value.tolist()
                
            else:
                self._data[self._ATTRS[name]] = [value]
            
        else:
            super().__setattr__(name, value)

    def _format_val(self, metric: str, val) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "-"

        if metric in self._PCT_METRICS:
            if isinstance(val, (int, float)):
                return f"{val * 100:.2f}%"
        
        if isinstance(val, (int, np.integer)):
            return f"{val:,}"
            
        if isinstance(val, float):
            return f"{val:.2f}"

        return str(val)

    def __str__(self) -> str:
        if all(len(v) == 0 for v in self._data.values()):
            return "Empty Tearsheet."

        w_metric = 25
        w_col = 15

        lines = [f"{"":<{w_metric}}{"Strategy":>{w_col}}{"Benchmark":>{w_col}}"]

        for metric in self._METRICS:
            vals = self._data[metric]
            
            if len(vals) == 0:
                continue
                
            elif len(vals) == 1:
                s = self._format_val(metric, vals[0])
                lines.append(f"{metric:<{w_metric}}{s:>{w_col}}")
                
            elif len(vals) == 2:
                s = self._format_val(metric, vals[0])
                b = self._format_val(metric, vals[1])
                lines.append(f"{metric:<{w_metric}}{s:>{w_col}}{b:>{w_col}}")

        return "\n".join(lines)
    
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