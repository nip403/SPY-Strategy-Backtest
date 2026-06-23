import statsmodels.api as sm
import pandas as pd
import numpy as np
from datetime import date

def round_date(date_index: pd.DataFrame, dt: date) -> date:
    dates = pd.Index(date_index.date).unique()
    dt = pd.to_datetime(dt)
    dt = dt.tz_localize(date_index.tz).date() if dt.tz is None else dt.tz_convert(date_index.tz).date()
    
    pos = dates.searchsorted(dt)
    
    if not pos:
        return dates[0]
    
    if pos == len(dates):
        return dates[-1]
    
    before = dates[pos - 1]
    after = dates[pos]
    
    return before if (dt - before) <= (after - dt) else after

class Tearsheet:
    def __init__(self):
        self._metrics = [
            "Total Days", "Average Trades / Day",
            "Cum. Return", "Ann. Return", "Avg. Daily Return",
            "Max Gain", "Best Day", "Max Loss", "Worst Day", "Win Rate", "Daily Win Rate",
            "Alpha", "Beta", "Ann. Volatility", 
            "Max Drawdown", "Max DD Days", "95% VaR", 
            "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Information Ratio"
        ]
        
        self._attrs = {
            "total_days": "Total Days",
            "trades_per_day": "Average Trades / Day",
            "cum_return": "Cum. Return",
            "ann_return": "Ann. Return",
            "avg_daily_return": "Avg. Daily Return",
            "max_gain": "Max Gain",
            "best_day": "Best Day",
            "max_loss": "Max Loss",
            "worst_day": "Worst Day",
            "win_rate": "Win Rate",
            "daily_win_rate": "Daily Win Rate",
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
        
        self._data = {metric: [] for metric in self._metrics}
        
    def generate(self, df: pd.DataFrame) -> "Tearsheet":
        equity = df[["strat_equity", "bench_equity"]]
        ret = df[["strat_ret", "bench_ret"]]
        dd = df[["strat_dd", "bench_dd"]]
        days = len(df.index)
        
        self.total_days = days
        self.trades_per_day = [df["trade_count"].mean(), None]
        
        self.cum_return = (1 + ret).prod().values - 1
        self.ann_return = (equity.iloc[-1] / equity.iloc[0]).values ** (252 / days) - 1
        self.avg_daily_return = ret.mean().values # arithmetic mean
        
        self.max_gain = ret.max().values 
        self.best_day = ret.idxmax().values
        self.max_loss = ret.min().values
        self.worst_day = ret.idxmin().values
        self.win_rate = [df.loc[df["strat_ret"] > 0, "trade_count"].sum() / df["trade_count"].sum(), None] # simplified calculation using agg data, not accurate
        self.daily_win_rate = (ret > 0).mean().values # daily, not per trade
        
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
        
        self.sharpe_ratio = np.array(self.ann_return) / np.array(self.ann_vol)

        ann_downside_vol = ret.apply(lambda col: col[col < 0].std()).values * np.sqrt(252)
        self.sortino_ratio = (np.array(self.ann_return) / np.where(ann_downside_vol == 0, 1e-6, ann_downside_vol)).tolist()

        abs_max_dd = np.abs(np.array(self.max_drawdown))
        self.calmar_ratio = (np.array(self.ann_return) / np.where(abs_max_dd == 0, 1e-6, abs_max_dd)).tolist()

        tracking_error = (df["strat_ret"] - df["bench_ret"]).std() * np.sqrt(252)
        self.information_ratio = [(self.ann_return[0] - self.ann_return[1]) / tracking_error if tracking_error != 0 else 0.0, None] # edge relative to unit benchmark risk
        
        return self

    def __getattr__(self, name: str):
        if name in self._attrs:
            return self._data[self._attrs[name]]

        if name.startswith("strat_"):
            attr = name[6:]
            
            if attr in self._attrs:
                metric = self._attrs[attr]
                
                return self._data[metric][0] if len(self._data[metric]) > 0 else None

        elif name.startswith("bench_"):
            attr = name[6:]
            
            if attr in self._attrs:
                metric = self._attrs[attr]
                
                return self._data[metric][1] if len(self._data[metric]) > 1 else None

        raise AttributeError(f"'Tearsheet' object has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        if name in ["_metrics", "_attrs", "_data"]:
            super().__setattr__(name, value)
            return

        if name.startswith("strat_"):
            attr = name[6:]
            
            if attr in self._attrs:
                if not len(self._data[self._attrs[attr]]):
                    self._data[self._attrs[attr]].append(value)
                else:
                    self._data[self._attrs[attr]][0] = value
                    
                return
                
        elif name.startswith("bench_"):
            attr = name[6:]
            
            if attr in self._attrs:
                match len(self._data[self._attrs[attr]]):
                    case 0:
                        self._data[self._attrs[attr]] = [None, value]
                    case 1:
                        self._data[self._attrs[attr]].append(value)
                    case 2:
                        self._data[self._attrs[attr]][1] = value
                
                return 
            
        if name in self._attrs:
            if isinstance(value, list):
                self._data[self._attrs[name]] = value
            
            elif isinstance(value, np.ndarray):
                self._data[self._attrs[name]] = value.tolist()
                
            else:
                self._data[self._attrs[name]] = [value]
            
        else:
            super().__setattr__(name, value)

    def _format_val(self, metric: str, val) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "-"
        
        pct_metrics = {
            "Cum. Return", "Ann. Return", "Avg. Daily Return", 
            "Max Gain", "Max Loss", "Win Rate", "Daily Win Rate"
            "Ann. Volatility", "Max Drawdown", "95% VaR"
        }

        if metric in pct_metrics:
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

        for metric in self._metrics:
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