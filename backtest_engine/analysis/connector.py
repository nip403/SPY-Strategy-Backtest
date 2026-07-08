from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(42) # for Monte Carlo

class StrategyConnector:
    def __init__(self, strategy_portfolio: Portfolio, book_equity: pd.Series, benchmark_equity: pd.Series, rebalance_period: int = 20) -> None:
        self.portfolio = strategy_portfolio
        self.book = book_equity # minute index, equity series, overall portfolio to incorporate strategy into
        self.bench = benchmark_equity # minute index, make sure spanned datetimes cover full book/strat period
        self.rebalance_period = rebalance_period
        
        # returns comparison df
        self.df = (
            pd.concat([
                self.portfolio.df["net_ret"], 
                self.book.pct_change(), 
                self.bench.pct_change(),
            ], axis=1, join="inner")
            .fillna(0)
        )
        
        self.df.columns = ["strat", "book", "bench"]
        self.daily = (1 + self.df).groupby(self.df.index.date).prod() - 1
        
        # create bundled portfolios - assumes regular rebalancing to given weight every self.rebalance_period days
        # NOTE: weights are BACKWARDS looking, set to base off book/strategy starting equity
        self._naive_w = self.portfolio.aum / (self.book.loc[self.df.index[0]] + self.portfolio.aum)
        
        self.daily["combined"] = self._mix_returns(np.array([self._naive_w]), self.daily["strat"].values[:, None]) # naive mix: target weight = initial starting capital proportions
        self._opt_w, self.daily["optimised"] = self._optimise_weight()
        
        self.metrics_df = pd.DataFrame(columns=self.daily.columns)
        
        self._generate_report()
        
    def _optimise_weight(self, depth: int = 3, points: int = 11) -> tuple[float, pd.Series]: # iterative 1d grid search
        lo, hi = 0, 1
        
        for _ in range(depth):
            weights = np.linspace(lo, hi, points)
            
            # w = aum / (aum + book). if weight = 1, book equity drops out, fallback to baseline strategy AUM
            aums = np.where(weights == 1, self.portfolio.aum, weights * self.book.loc[self.df.index[0]] / (1 - weights))
            
            daily = (pd.DataFrame(
                1 + self.portfolio.returns_matrix(aums), 
                index=self.portfolio.df.index,
            ).groupby(self.portfolio.df.index.date).prod() - 1).loc[self.daily.index].values
            
            mixed = self._mix_returns(weights, daily)
            
            std = np.nanstd(mixed, axis=0)
            sharpes = np.where(std != 0, (np.nanmean(mixed, axis=0) / std) * np.sqrt(252), 0)
            
            best_idx = np.argmax(sharpes)
            best_weight = weights[best_idx]
            
            if best_weight >= 0.9: # avoid blowing up strat aum since it scales hyperbolically with weight approaching 1
                return 1, self.daily["strat"]
            
            span = (hi - lo) / (points - 1)
            lo = max(0, best_weight - span)
            hi = min(1, best_weight + span)
    
        return best_weight, pd.Series(mixed[:, best_idx], index=self.daily.index)
    
    def _mix_returns(self, weights: np.ndarray, strat_daily_matrix: np.ndarray) -> np.ndarray:
        n, k = strat_daily_matrix.shape
        book_daily = self.daily["book"].values[:, None] # reshape to broadcast

        # component cumprods
        cum_s = np.empty((n + 1, k))
        cum_b = np.empty((n + 1, 1))
        
        cum_s[0, :] = cum_b[0, 0] = 1

        np.cumprod(1 + strat_daily_matrix, axis=0, out=cum_s[1:, :])
        np.cumprod(1 + book_daily, axis=0, out=cum_b[1:, :])

        rebals = (np.arange(n) // self.rebalance_period) * self.rebalance_period # rebalance windows, element = start index of block

        # broadcast weights
        p = weights * cum_s[1:, :] / cum_s[rebals, :] + (1 - weights) * cum_b[1:, :] / cum_b[rebals, :]

        # calc daily mixed returns 
        r = p / np.vstack([np.ones((1, k)), p[:-1, :]]) - 1
        r[rebals[1:], :] = p[rebals[1:], :] - 1 # overwrite on rebalance block boundaries

        return r 
    
    def _generate_report(self) -> None:
        s = self.daily[["strat", "combined", "optimised"]]
        b = self.daily["book"]

        # return, exposure, distribution
        self.metrics_df.loc["Expected Return"] = self.daily.mean() * 252
        self.metrics_df.loc["Volatility"] = self.daily.std() * np.sqrt(252)
        self.metrics_df.loc["Sharpe"] = np.where(self.daily.std() != 0, (self.daily.mean() / self.daily.std()) * np.sqrt(252), 0)
        self.metrics_df.loc["Beta"] = self.daily.cov().loc["bench"] / self.daily["bench"].var()

        self.metrics_df.loc["Skewness"] = self.daily.skew()
        self.metrics_df.loc["Kurtosis"] = self.daily.kurt()

        # dd & recovery
        equity = (1 + self.daily).cumprod()
        drawdowns = equity / equity.cummax() - 1
 
        self.metrics_df.loc["Max Drawdown"] = drawdowns.min()

        last_zero = (drawdowns < 0).cumsum().where(drawdowns >= 0).ffill().fillna(0)
        self.metrics_df.loc["Max DD Days"] = ((drawdowns < 0).cumsum() - last_zero).max()

        trough_rows = drawdowns.values.argmin(axis=0)
        rows = np.arange(len(drawdowns))[:, None]

        recovery_rows = np.where((rows >= trough_rows) & np.isclose(drawdowns.values, 0), rows, np.inf).min(axis=0)
        self.metrics_df.loc["Max DD Recovery Days"] = np.where(np.isinf(recovery_rows), len(drawdowns), recovery_rows) - trough_rows
        
        # correlation
        self.metrics_df.loc["Correlation to Book", s.columns] = s.corrwith(b)
        self.metrics_df.loc["Crash Correlation to Book", s.columns] = s[b <= b.quantile(0.10)].corrwith(b[b <= b.quantile(0.10)])
        self.metrics_df.loc["Intraday Correlation to Book", "strat"] = self.df["strat"].corr(self.df["book"])

        # alpha & incremental
        self.metrics_df.loc["Alpha"] = (self.daily.mean() - self.metrics_df.loc["Beta"] * self.daily["bench"].mean()) * 252
        self.metrics_df.loc["Idiosyncratic Risk"] = self.daily.sub(self.metrics_df.loc["Beta"] * self.daily["bench"] + (self.metrics_df.loc["Alpha"] / 252), axis=0).std() * np.sqrt(252) # capm residual vs bench
        
        b_beta = s.corrwith(b) * (s.std() / b.std())
        self.metrics_df.loc["Incremental Alpha", s.columns] = (s.mean() - b_beta * b.mean()) * 252 # incremental edge vs existing
        self.metrics_df.loc["Incremental Risk", s.columns] = s.sub(b_beta * b + (self.metrics_df.loc["Incremental Alpha", s.columns] / 252), axis=0).std() * np.sqrt(252)
        
        self.metrics_df.loc["Incremental Sharpe Test", "combined"] = "Pass" if self.metrics_df.loc["Sharpe", "combined"] > self.metrics_df.loc["Sharpe", "book"] * self.metrics_df.loc["Correlation to Book", "strat"] else "Fail"

        # risk & robustness
        self.metrics_df.loc["Lower Tail Dependency", s.columns] = ((s <= s.quantile(0.10)).mul(b <= b.quantile(0.10), axis=0)).sum() / (b <= b.quantile(0.10)).sum()
        self.metrics_df.loc["Upper Tail Dependency", s.columns] = ((s >= s.quantile(0.90)).mul(b >= b.quantile(0.90), axis=0)).sum() / (b >= b.quantile(0.90)).sum()

        self.metrics_df.loc["95% cVar (Historical)"] = self.daily[self.daily <= self.daily.quantile(0.05)].mean()
        self.metrics_df.loc["95% cVar (Monte Carlo)", s.columns] = np.sort(s.values[np.random.randint(0, len(self.daily), size=(10000, 252))], axis=1)[:, :12, :].mean(axis=(0, 1))
    
        # market capture
        for name, mask in {
            "Up-Market Capture": self.daily["bench"] > 0,
            "Down-Market Capture": self.daily["bench"] < 0,
        }.items():
            self.metrics_df.loc[name] = self.daily[mask].mean() / self.daily.loc[mask, "bench"].mean()

    
        active_returns = self.daily.sub(self.daily["bench"], axis=0)
        tracking_error = active_returns.std() * np.sqrt(252)
        self.metrics_df.loc["Information Ratio"] = np.where(tracking_error != 0, (active_returns.mean() * 252) / tracking_error, 0)
    
        downside_returns = self.daily.copy()
        downside_returns[downside_returns > 0] = 0
        downside_dev = downside_returns.std() * np.sqrt(252)
        
        self.metrics_df.loc["Sortino Ratio"] = np.where(downside_dev != 0, (self.daily.mean() * 252) / downside_dev, 0)
        self.metrics_df.loc["Calmar Ratio"] = np.where(self.metrics_df.loc["Max Drawdown"] != 0, (self.daily.mean() * 252) / abs(self.metrics_df.loc["Max Drawdown"]), 0)
    
        # info/weights
        self.metrics_df.loc["Strategy Weight", ["combined", "optimised"]] = [self._naive_w, self._opt_w]
        self.metrics_df.loc["Strategy AUM", ["combined", "optimised"]] = [self.book.loc[self.df.index[0]] * (self._naive_w / (1 - self._naive_w)), self.book.loc[self.df.index[0]] * (self._opt_w / (1 - self._opt_w))]

    def report(self, plot: bool = True) -> None:
        if plot:
            fig, axes = plt.subplots(
                nrows=4, 
                ncols=1, 
                figsize=(14, 18), 
                sharex=True, 
                gridspec_kw={"height_ratios": [3, 2, 1.5, 1.5]}
            )
            
            # cumulative equity
            equity = (1 + self.daily).cumprod()
            cols = {c: "Strategy" if c == "strat" else c.capitalize() for c in ["bench", "book", "strat", "combined", "optimised"]}
            
            for col, name in cols.items():
                axes[0].plot(equity.index, equity[col], label=name, linewidth=1, linestyle="--" if col == "bench" else "-")
                
            axes[0].set_title("Cumulative Portfolio Equity", loc="left", fontweight="bold")
            axes[0].set_ylabel("Growth Factor")
            axes[0].legend(loc="upper left", frameon=False)

            # rolling monthly sharpe
            sharpe = (self.daily.rolling(21).mean() / self.daily.rolling(21).std()) * np.sqrt(252)
            
            for col, name in cols.items():
                axes[1].plot(sharpe.index, sharpe[col], label=name, linewidth=1, linestyle="--" if col == "bench" else "-")
                
            axes[1].set_title("Rolling Monthly Sharpe", loc="left", fontweight="bold")
            axes[1].set_ylabel("Annualized Sharpe")
            axes[1].legend(loc="upper left")

            # rolling monthly correlation strat vs book
            roll_corr = self.daily["strat"].rolling(21).corr(self.daily["book"])
            
            axes[2].plot(roll_corr.index, roll_corr, linewidth=1)
            
            axes[2].axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
            axes[2].set_title("Rolling Monthly Correlation (Strategy vs Book)", loc="left", fontweight="bold")
            axes[2].set_ylabel("Correlation")
            
            # rolling monthly beta
            for col in ["strat", "combined", "optimised"]:
                roll_beta = self.daily[col].rolling(21).cov(self.daily["bench"]) / self.daily["bench"].rolling(21).var()
                axes[3].plot(roll_beta.index, roll_beta, label=f"Beta ({cols[col]} vs Bench)", linewidth=1)
            
            axes[3].axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
            axes[3].set_title("Rolling Monthly Betas", loc="left", fontweight="bold")
            axes[3].set_ylabel("Beta")
            axes[3].legend(loc="upper left", frameon=False)

            for ax in axes:
                ax.label_outer()
                ax.margins(x=0)

            plt.xlabel("Date", fontweight="bold")
            plt.tight_layout()
            plt.show()
            
        print(self)

    def __str__(self) -> str:
        cols = {
            "book": "Book",
            "strat": "Strategy",
            "combined": "Combined",
            "optimised": "Optimised",
            "bench": "Bench",
        }

        df = self.metrics_df.copy().rename(columns=cols)[["Book", "Strategy", "Combined", "Optimised", "Bench"]]

        pct_metrics = {
            "Expected Return",
            "Volatility",
            "Max Drawdown",
            "Alpha",
            "Incremental Alpha",
            "95% cVar (Historical)",
            "95% cVar (Monte Carlo)",
            "Strategy Weight",
        }
        
        int_metrics = {
            "Max DD Days",
            "Max DD Recovery Days",
        }
        
        def fmt(x, metric):
            if pd.isna(x):
                return "-"
            
            elif not isinstance(x, (int, float, np.number)):
                return x
            
            elif metric in pct_metrics:
                return f"{x:.2%}"
            
            elif metric in int_metrics:
                return f"{int(x)}"
            
            else:
                return f"{x:.2f}"

        return f"=== Portfolio Integration & Risk Report ===\n{df.apply(lambda col: [fmt(x, m) for m, x in col.items()])}"