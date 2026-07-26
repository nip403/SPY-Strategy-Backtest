from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

np.random.seed(42) # for Monte Carlo

warnings.filterwarnings("ignore", category=FutureWarning)

class StrategyConnector:
    def __init__(self, strategy_portfolio: Portfolio, book_equity: pd.Series, benchmark_equity: pd.Series, rebalance_period: int = 20) -> None:
        """
        Initialise portfolio integration analysis between an existing strategy and a master trading book.
        Combines strategy returns with an existing portfolio, evaluates naive and optimised allocations, and generates key metrics.

        Only evaluates the period intersected by strategy-book-benchmark indices.

        strategy_portfolio : Portfolio
            Strategy portfolio to integrate into the existing book.
        book_equity : pd.Series
            Minute-indexed existing book equity curve to be compared with strategy.
            Forward-filling daily equity should only affect intraday statistics produced in the report. 
        benchmark_equity : pd.Series
            Minute-indexed benchmark equity curve aligned with the book and strategy periods.
            No validation, but must span datetimes that cover the full strategy/book period.
        rebalance_period : int = 20
            Number of trading days between portfolio weight resets.
        """
        
        self.portfolio = strategy_portfolio
        self.book = book_equity 
        self.bench = benchmark_equity 
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
        
        self.initial_book_cap = self.book.loc[self.df.index[0]]
        
        self.df.columns = ["strat", "book", "bench"]
        self.daily = (1 + self.df).groupby(self.df.index.date).prod() - 1
        
        # create bundled portfolios - assumes regular rebalancing to given weight every self.rebalance_period days
        # NOTE: weights are BACKWARDS looking, set to base off book/strategy starting equity
        self._naive_w = 0.5
        self.daily["combined"] = self._mix_returns(np.array([self._naive_w]), self._get_exact_returns(self._naive_w))[:, 0] # naive mix: target weight = initial starting capital proportions
        
        self._opt_w, self.daily["optimised"] = self._optimise_weight()
        
        self.metrics_df = pd.DataFrame(columns=self.daily.columns, dtype=object)
        
        self._generate_report()
        
    def _optimise_weight(self, depth: int = 3, points: int = 11) -> tuple[float, pd.Series]:
        """
        Optimise strategy allocation weight by Sharpe ratio using an iterative 1D grid search.
        Searches over strategy weights and selects the allocation producing the highest annualised Sharpe ratio.
        
        Uses an AUM mean scaling approximation to project cost changes as strategy is rebalanced to avoid ridiculously high memory needs.
        
        Assumes trading book capital remains constant while adjusting weights for the strategy.

        depth : int = 3
            Number of refinement rounds performed around the best weight.
            Increases the search precision by one decimal place when using points = 11.
        points : int = 11
            Number of candidate weights tested per refinement round.
            Default points = 11 tests weights in 10% increments (0.0, 0.1, ..., 1.0) before refining the interval.

        Returns tuple[float, pd.Series]
            Optimised strategy weight and resulting mixed portfolio daily returns.
        """
        
        lo, hi = 0, 1
        
        for _ in range(depth):
            weights = np.linspace(lo, hi, points)
            
            approx_mixed = self._mix_returns(weights, np.tile(self.daily["strat"].values[:, None], (1, points))) 
            
            # average portfolio growth factor
            port_growth = np.cumprod(1 + approx_mixed, axis=0)
            port_growth = np.vstack([np.ones((1, points)), port_growth[:-1, :]]) # use capital on previous day to rebalance
            
            block_growth = port_growth[np.arange(0, len(self.daily), self.rebalance_period), :]
            mean_growth = block_growth.mean(axis=0)
            
            # scale by mean mixed growth per weight
            aums = weights * self.initial_book_cap * mean_growth
            
            daily = (pd.DataFrame(
                1 + self.portfolio.returns_matrix(aums),
                index=self.portfolio.df.index,
            ).groupby(self.portfolio.df.index.date).prod() - 1).loc[self.daily.index].values
            
            mixed = self._mix_returns(weights, daily)
            
            std = np.nanstd(mixed, axis=0)
            sharpes = np.where(std != 0, (np.nanmean(mixed, axis=0) / std) * np.sqrt(252), 0)
            
            best_idx = np.argmax(sharpes)
            best_weight = weights[best_idx]
            
            span = (hi - lo) / (points - 1)
            lo = max(0, best_weight - span)
            hi = min(1, best_weight + span)
            
        final_mixed = self._mix_returns(np.array([best_weight]), self._get_exact_returns(best_weight))[:, 0]
    
        return best_weight, pd.Series(final_mixed, index=self.daily.index)
    
    def _mix_returns(self, weights: np.ndarray, strat_daily_matrix: np.ndarray) -> np.ndarray:
        """
        Combine strategy and book returns according to provided weights, accounting for periodic rebalancing.
        
        Calculates portfolio returns from changing strategy allocations between rebalance points.

        weights : np.ndarray
            Strategy allocation weights for each return scenario.
        strat_daily_matrix : np.ndarray
            Matrix of strategy daily returns across allocation scenarios.

        Returns np.ndarray
            Matrix of mixed portfolio daily returns.
        """
    
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
    
    def _get_exact_returns(self, weight: float) -> np.ndarray:
        """
        Calculates the exact cost-adjusted daily returns for a rebalanced strategy integration.
        Recomputes strategy returns using exact AUM at each rebalance, scaling strategy capital with true mixed portfolio growth to maintain the target weight.

        weight : float
            Target strategy allocation.

        Returns np.ndarray
            Daily strategy return matrix of shape (n_days, 1).
        """
        
        approx_mixed = self._mix_returns(np.array([weight]), self.daily["strat"].values[:, None])[:, 0]
        
        # portfolio & book growth at the start of each rebalance period
        port_growth = pd.Series(approx_mixed).add(1).cumprod().shift(1).fillna(1).values
        block_growth = port_growth[np.arange(0, len(self.daily), self.rebalance_period)]
        block_idx = np.arange(len(self.daily)) // self.rebalance_period
        
        exact_aums = weight * self.initial_book_cap * block_growth
        
        daily_mixed = (pd.DataFrame(
            1 + self.portfolio.returns_matrix(exact_aums),
            index=self.portfolio.df.index,
        ).groupby(self.portfolio.df.index.date).prod() - 1).loc[self.daily.index].values
        
        return daily_mixed.reshape(len(self.daily), len(block_growth), 1)[np.arange(len(self.daily)), block_idx]

    def _generate_report(self) -> None:
        """
        Generate portfolio integration and risk metrics.
        """
    
        s = self.daily[["strat", "combined", "optimised"]]
        b = self.daily["book"]
        
        new_strats = ["combined", "optimised"]
        non_bench = ["book", "strat"] + new_strats

        # return, exposure, distribution
        self.metrics_df.loc["Expected Return"] = self.daily.mean() * 252
        self.metrics_df.loc["Volatility"] = self.daily.std() * np.sqrt(252)
        self.metrics_df.loc["Sharpe Ratio"] = np.where(self.daily.std() != 0, (self.daily.mean() / self.daily.std()) * np.sqrt(252), 0)
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
        self.metrics_df.loc["Intraday Correlation to Book", "strat"] = self.df[["strat", "book"]].groupby(self.df.index.date).corr().iloc[0::2, -1].mean() # average correlation intraday, probably not useful. very cursed slicing due to output of corr

        # alpha & incremental
        self.metrics_df.loc["Alpha"] = (self.daily.mean() - self.metrics_df.loc["Beta"] * self.daily["bench"].mean()) * 252
        self.metrics_df.loc["Idiosyncratic Risk"] = (self.daily - (self.daily["bench"].values[:, None] * self.metrics_df.loc["Beta"].values + self.metrics_df.loc["Alpha"].values / 252)).std() * np.sqrt(252) # capm residual vs bench
        
        b_beta = s.corrwith(b) * (s.std() / b.std())
    
        self.metrics_df.loc["Incremental Alpha", new_strats] = (s[new_strats].mean() - b_beta[new_strats] * b.mean()) * 252 # incremental edge vs book
        self.metrics_df.loc["Incremental Risk", new_strats] = (s[new_strats] - (b.values[:, None] * b_beta[new_strats].values + self.metrics_df.loc["Incremental Alpha", new_strats].values / 252)).std() * np.sqrt(252)
        
        self.metrics_df.loc["Incremental Sharpe (Marginal)", "strat"] = self.metrics_df.loc["Sharpe Ratio", "strat"] - self.metrics_df.loc["Correlation to Book", "strat"] * self.metrics_df.loc["Sharpe Ratio", "book"] # sharpe heuristic: SRnew > SRold * corr
        self.metrics_df.loc["Incremental Sharpe (Realised)", new_strats] = self.metrics_df.loc["Sharpe Ratio", new_strats] - self.metrics_df.loc["Sharpe Ratio", "book"]
        
        # risk & robustness
        self.metrics_df.loc["Lower Tail Dependency", s.columns] = ((s <= s.quantile(0.10)).mul(b <= b.quantile(0.10), axis=0)).sum() / (b <= b.quantile(0.10)).sum()
        self.metrics_df.loc["Upper Tail Dependency", s.columns] = ((s >= s.quantile(0.90)).mul(b >= b.quantile(0.90), axis=0)).sum() / (b >= b.quantile(0.90)).sum()

        self.metrics_df.loc["95% cVar (Historical)"] = self.daily[self.daily <= self.daily.quantile(0.05)].mean()
        self.metrics_df.loc["95% cVar (Monte Carlo)"] = np.sort(self.daily.values[np.random.randint(0, len(self.daily), size=(10000, 252))], axis=1)[:, :12, :].mean(axis=(0, 1))
    
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
        self.metrics_df.loc["Strategy Weight", new_strats] = [self._naive_w, self._opt_w]
        self.metrics_df.loc["Total AUM (Initial)", non_bench] = [self.initial_book_cap, self.portfolio.aum, self.initial_book_cap, self.initial_book_cap]
        self.metrics_df.loc["Total AUM (Final)", non_bench] = self.metrics_df.loc["Total AUM (Initial)"] * equity.iloc[-1]

        self.metrics_df.loc["Strategy AUM (Initial)", new_strats] = [self._naive_w * self.initial_book_cap, self._opt_w * self.initial_book_cap]
        self.metrics_df.loc["Strategy AUM (Final)", new_strats] = self.metrics_df.loc["Total AUM (Final)", new_strats] * np.array([self._naive_w, self._opt_w])
        
    def report(self, plot: bool = True) -> None:
        """
        Display portfolio integration performance and risk analysis.

        Plots cumulative equity, rolling Sharpe ratio, strategy correlation, and beta.

        plot : bool = True
            Whether to display performance charts before printing the report.
        """
    
        if plot:
            fig, axes = plt.subplots(
                nrows=4,
                ncols=1,
                figsize=(14, 12),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1, 1, 1]}
            )
            
            # cumulative equity
            equity = (1 + self.daily).cumprod()
            cols = {c: "Strategy" if c == "strat" else c.capitalize() for c in ["bench", "book", "strat", "combined", "optimised"]}
            
            for col, name in cols.items():
                axes[0].plot(equity.index, equity[col], label=name, linewidth=1, linestyle="--" if col == "bench" else "-")
                
            axes[0].set_title("Cumulative Portfolio Equity", loc="left", fontweight="bold")
            axes[0].set_ylabel("Growth Factor")
            axes[0].legend(loc="upper left")

            # rolling semi-annual sharpe
            sharpe = (self.daily.rolling(126).mean() / self.daily.rolling(126).std()) * np.sqrt(252)
            
            for col in ["optimised", "book"]:
                axes[1].plot(sharpe.index, sharpe[col], label=cols[col], linewidth=1)
                
            axes[1].set_title("Rolling Semi-Annual Sharpe", loc="left", fontweight="bold")
            axes[1].set_ylabel("Annualized Sharpe")
            axes[1].legend(loc="upper left")

            # rolling semi-annual correlation strat vs book
            roll_corr = self.daily["strat"].rolling(126).corr(self.daily["book"])
            
            axes[2].plot(roll_corr.index, roll_corr, linewidth=1)
            
            axes[2].axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
            axes[2].set_title("Rolling Semi-Annual Correlation (Strategy vs Book)", loc="left", fontweight="bold")
            axes[2].set_ylabel("Correlation")
            
            # rolling semi-annual beta
            for col in ["strat", "combined", "optimised"]:
                roll_beta = self.daily[col].rolling(126).cov(self.daily["bench"]) / self.daily["bench"].rolling(126).var()
                axes[3].plot(roll_beta.index, roll_beta, label=f"Beta ({cols[col]} vs Bench)", linewidth=1)
            
            axes[3].axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
            axes[3].set_title("Rolling Semi-Annual Betas", loc="left", fontweight="bold")
            axes[3].set_ylabel("Beta")
            axes[3].legend(loc="upper left")

            for ax in axes:
                ax.label_outer()
                ax.margins(x=0)

            plt.xlabel("Date", fontweight="bold")
            plt.tight_layout()
            plt.show()
            
        print(self)

    def __str__(self) -> str:
        """
        Format integration and risk metrics into a readable report table.

        Returns str
            Formatted portfolio integration and risk report.
        """
    
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
            "Idiosyncratic Risk",
            "Incremental Risk",
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

        return f"=== Portfolio Integration & Risk Report ===\n{df.apply(lambda col: [fmt(x, m) for m, x in col.items()]).to_string()}"