from __future__ import annotations

import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional

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