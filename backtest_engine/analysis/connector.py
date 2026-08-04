from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional, Any
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator
import pandas as pd
import numpy as np
import warnings
from .base import AnalysisReport
from .decomposition import PortfolioDecomposer
from .metrics import SeriesMetrics, RelativeMetrics, ConnectorExtras, compute_series_metrics, compute_relative_metrics, format_value, dataclass_rows, merge_groups, render_sections
from ..utils import compute_drawdown, save_figures

warnings.filterwarnings("ignore", category=FutureWarning)

class StrategyConnector(AnalysisReport):
    DEFAULT_PARAMS = {
        "rebalance_period": 21,
        "lookback_window": 21,
        "weight_intervals": 0.01,
    }
    
    def __init__(self, strategy_portfolio: Portfolio, book_equity: pd.Series, benchmark_equity: pd.Series, config: Optional[dict[str, Any]] = None) -> None:
        """
        Initialise portfolio integration analysis between an existing strategy and a master trading book.
        Combines strategy returns with an existing portfolio, evaluates naive and optimised allocations, and generates key metrics.

        The strategy is split into long and short legs:
            Long exposure is funded by withdrawing "weight" fraction of book capital and deploying it (i.e. reallocation).
            Short exposure is margin-funded: its notional is sized identically (the same weight). No margin limit or financing/borrow cost is modelled.
            If the provided strategy is short-only, optimal weight is determined by varying short-leg notional from 0-100% of book value.

        Only evaluates the period intersected by strategy-book-benchmark indices.
        Across all strategy-book mixing (e.g. rebalancing, _tradeoff_profile, optimising weight, etc.), models book with simple linear costing.

        strategy_portfolio : Portfolio
            Strategy portfolio to integrate into the existing book.
        book_equity : pd.Series
            Minute-indexed existing book equity curve to be compared with strategy.
            Forward-filling daily equity should only affect intraday statistics produced in the report.
        benchmark_equity : pd.Series
            Minute-indexed benchmark equity curve aligned with the book and strategy periods.
            No validation, but must span datetimes that cover the full strategy/book period.
        config : dict[str, Any] = None
            Overrides default config values.
            
            rebalance_period : int = 21
                Number of trading days between portfolio weight resets, default 1 month.
            lookback_window : int = 21
                Lookback window for rolling-z in self._tradeoff_profile, defaults to 1 trading month.
            weight_intervals : float = 0.01
                The interval to sweep strategy to book mix in self._tradeoff_profile, defaults to 1%.
                Rounded to the nearest basis point (4dp) to prevent floating point errors.
        """
        
        config = {**self.DEFAULT_PARAMS, **(config or {})}
        
        # for self._tradeoff_profile, consider adding switches (just override after init in current implementation)
        self.lookback_window = config["lookback_window"]
        self.weight_intervals = config["weight_intervals"]

        self.book = book_equity
        self.bench = benchmark_equity
        self.rebalance_period = config["rebalance_period"]

        # returns comparison df - long/short legs are internal only
        self.df = (
            pd.concat([
                strategy_portfolio.df["net_ret"],
                self.book.pct_change(),
                self.bench.pct_change(),
            ], axis=1, join="inner")
            .fillna(0)
        )

        self.initial_book_cap = self.book.loc[self.df.index[0]]

        self.df.columns = ["strat", "book", "bench"]

        valid_days = self.df.index.floor("D").unique()
        self.daily = ((1 + self.df).resample("D").prod() - 1).loc[valid_days]
        self.daily.index = self.daily.index.date

        # populated by _exact_leg_returns on first call: None = not yet probed, False = AUM-sensitive (e.g. DynamicCostModel), else the cached (n_days, 1) AUM-invariant result
        self._cached_legs = None

        # internal-only long/short daily legs
        daily_long, daily_short = self._exact_leg_returns(strategy_portfolio, np.array([strategy_portfolio.aum]))
        self._daily_long = pd.Series(daily_long[:, 0], index=self.daily.index)
        self._daily_short = pd.Series(daily_short[:, 0], index=self.daily.index)
        self._has_long_leg = bool((self._daily_long != 0).any())

        # create bundled portfolios - assumes regular rebalancing to given weight every self.rebalance_period days
        # NOTE: weights are BACKWARDS looking, set to base off book/strategy starting equity
        self._naive_w = 0.5
        naive_long, naive_short = self._exact_block_leg_returns(strategy_portfolio, np.array([self._naive_w]))
        self.daily["combined"] = self._mix_returns(np.array([self._naive_w]), naive_long, naive_short)[:, 0] # naive mix: target weight = initial starting capital proportions

        self._opt_w, self.daily["optimised"] = self._optimise_weight(strategy_portfolio)

        # reporting
        self.metrics: dict[str, SeriesMetrics] = {}
        self.relative: dict[str, RelativeMetrics] = {}
        self.incremental: dict[str, RelativeMetrics] = {}
        self.extras: ConnectorExtras | None = None
        
        self.tradeoff_series = self._tradeoff_profile(strategy_portfolio)

        self._generate_report(strategy_portfolio)

    def _optimise_weight(self, portfolio: Portfolio, depth: int = 3, points: int = 11) -> tuple[float, pd.Series]:
        """
        Optimise strategy allocation weight by Sharpe ratio using an iterative 1D grid search.
        Searches over the weight used to size both legs, and selects the allocation maximising sharpe.

        Uses an AUM mean scaling approximation to project cost changes as strategy is rebalanced to avoid ridiculously high memory needs..

        portfolio : Portfolio
            Strategy portfolio object.
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

            long_daily, short_daily = self._approx_aum_leg_returns(portfolio, weights)
            mixed = self._mix_returns(weights, long_daily, short_daily)

            std = np.nanstd(mixed, axis=0)
            sharpes = np.where(std != 0, (np.nanmean(mixed, axis=0) / std) * np.sqrt(252), 0)

            best_idx = np.argmax(sharpes)
            best_weight = weights[best_idx]

            span = (hi - lo) / (points - 1)
            lo = max(0, best_weight - span)
            hi = min(1, best_weight + span)

        final_long, final_short = self._exact_block_leg_returns(portfolio, np.array([best_weight]))
        final_mixed = self._mix_returns(np.array([best_weight]), final_long, final_short)[:, 0]

        return best_weight, pd.Series(final_mixed, index=self.daily.index)

    def _mix_returns(self, weights: np.ndarray, long_daily_matrix: np.ndarray, short_daily_matrix: np.ndarray) -> np.ndarray:
        """
        Combine long-leg, short-leg, and book returns according to specifed weights, accounting for periodic rebalancing.
        Long legs withdraw capital from the book, while short legs are margin-funded. 
        
        NOTE: No margin limit is assumed - short notional is sized at the same weight.

        weights : np.ndarray
            Strategy allocation weights for each return scenario.
        long_daily_matrix : np.ndarray
            Matrix of long-leg daily returns, either one column per scenario or a single column broadcast across all of them. 
            Ignored when self._has_long_leg is False.
        short_daily_matrix : np.ndarray
            Matrix of short-leg daily returns, either one column per scenario or a single column broadcast across all of them.

        Returns np.ndarray
            Matrix of mixed portfolio daily returns. Shape (n_days, scenarios)
        """

        n, k = short_daily_matrix.shape[0], len(weights)
        book_daily = self.daily["book"].values[:, None] # reshape to broadcast

        # component cumprods, each sized to its input's width 
        # aum invariant leg columns stay a single column and is broadcasted only at the combine step
        cum_s = np.empty((n + 1, short_daily_matrix.shape[1]))
        cum_b = np.empty((n + 1, 1))

        cum_s[0, :] = cum_b[0, 0] = 1

        np.cumprod(1 + short_daily_matrix, axis=0, out=cum_s[1:, :])
        np.cumprod(1 + book_daily, axis=0, out=cum_b[1:, :])

        rebals = (np.arange(n) // self.rebalance_period) * self.rebalance_period # rebalance windows, element = start index of block

        # short leg's dollar P&L, layered on top (margin-funded, additive) regardless of the long leg
        short_term = weights * (cum_s[1:, :] / cum_s[rebals, :] - 1)

        if self._has_long_leg:
            cum_l = np.empty((n + 1, long_daily_matrix.shape[1]))
            cum_l[0, :] = 1
            np.cumprod(1 + long_daily_matrix, axis=0, out=cum_l[1:, :])

            # book + long reallocation + short
            p = (
                weights * cum_l[1:, :] / cum_l[rebals, :]
                + (1 - weights) * cum_b[1:, :] / cum_b[rebals, :]
                + short_term
            )
        else:
            # no long leg = book keeps its full growth
            p = cum_b[1:, :] / cum_b[rebals, :] + short_term

        # calc daily mixed returns
        r = p / np.vstack([np.ones((1, k)), p[:-1, :]]) - 1
        r[rebals[1:], :] = p[rebals[1:], :] - 1 # overwrite on rebalance block boundaries

        return r

    def _exact_leg_returns(self, portfolio: Portfolio, aums: np.ndarray) -> tuple[np.ndarray]:
        """
        Calculates capacity-aware daily long/short leg returns at each AUM. 
        If the connected execution/cost-model is AUM-invariant, all future calls fall back to cache.
        
        portfolio : Portfolio
            Strategy portfolio object.
        aums : np.ndarray
            Portfolio capital values to evaluate, one per scenario.

        Returns tuple[np.ndarray, np.ndarray]
            (long_daily, short_daily), shape (n_days, len(aums)).
        """

        if self._cached_legs:
            long_daily, short_daily = self._cached_legs
            shape = (len(self.daily), len(aums))

            return np.broadcast_to(long_daily, shape), np.broadcast_to(short_daily, shape)

        long_daily, short_daily = self._compute_leg_returns(portfolio, aums)

        if self._cached_legs is False:
            return long_daily, short_daily

        probe_aum = aums[:1] + max(1, abs(float(aums[0])))
        probe_long, probe_short = self._compute_leg_returns(portfolio, probe_aum)

        if np.array_equal(long_daily[:, :1], probe_long) and np.array_equal(short_daily[:, :1], probe_short):
            self._cached_legs = (long_daily[:, :1], short_daily[:, :1])
        else:
            self._cached_legs = False

        return long_daily, short_daily

    def _compute_leg_returns(self, portfolio: Portfolio, aums: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        The actual capacity-aware position -> gross -> net -> daily-resample pipeline driving self._exact_leg_returns.
        
        portfolio : Portfolio
            Strategy portfolio object.
        aums : np.ndarray
            Portfolio capital values to evaluate, one per scenario.

        Returns tuple[np.ndarray, np.ndarray]
            (long_daily, short_daily), shape (n_days, len(aums)).
        """

        position_matrix, gross_matrix, net_matrix = portfolio.returns_matrix_components(aums)
        ret = portfolio.df["ret"].to_numpy()[:, None]

        legs = PortfolioDecomposer.split_long_short(position_matrix, ret, gross_matrix, net_matrix)

        def to_daily(net_ret_matrix: np.ndarray) -> np.ndarray:
            daily = pd.DataFrame(1 + net_ret_matrix, index=portfolio.df.index).resample("D").prod() - 1
            daily.index = daily.index.date

            return daily.loc[self.daily.index].values

        return to_daily(legs["long"]["net_ret"]), to_daily(legs["short"]["net_ret"])

    def _exact_block_leg_returns(self, portfolio: Portfolio, weights: np.ndarray) -> tuple[np.ndarray]:
        """
        Calculates the exact cost-adjusted daily long/short returns for rebalanced mix for all target weights.
        Recomputes leg returns using exact AUM at each rebalance, scaling strategy capital with mixed portfolio growth to maintain target weight.

        portfolio : Portfolio
            Strategy portfolio object.
        weights : np.ndarray
            Target strategy allocations to evaluate.

        Returns tuple[np.ndarray, np.ndarray]
            (long_daily, short_daily), each of shape (n_days, len(weights)).
        """
        
        # single shared column, not tiled to len(weights) - to avoid cumprod across multiple identical cols
        approx_mixed = self._mix_returns(
            weights,
            self._daily_long.values[:, None],
            self._daily_short.values[:, None],
        )

        # portfolio & book growth at the start of each rebalance period, per weight scenario
        port_growth = np.vstack([np.ones((1, len(weights))), np.cumprod(1 + approx_mixed, axis=0)[:-1, :]])
        block_starts = np.arange(0, len(self.daily), self.rebalance_period)
        block_growth = port_growth[block_starts, :] # shape (n_blocks, k)
        block_idx = np.arange(len(self.daily)) // self.rebalance_period

        exact_aums = (weights[None, :] * self.initial_book_cap * block_growth).ravel() # (block, weight) pairs, flattened for one batched capacity-cost call

        long_daily, short_daily = self._exact_leg_returns(portfolio, exact_aums)

        n_blocks = len(block_starts)
        pick = lambda m: m.reshape(len(self.daily), n_blocks, len(weights))[np.arange(len(self.daily)), block_idx, :]

        return pick(long_daily), pick(short_daily)

    def _approx_aum_leg_returns(self, portfolio: Portfolio, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Approximates capacity-aware daily long/short leg returns per weight using a single AUM per weight.
        Avoids blowing up memory by recomputing exact AUM at every block boundary like in self._exact_block_leg_returns.

        Scales at O(len(weights)) from O(n_blocks x len(weights)).

        portfolio : Portfolio
            Strategy portfolio object.
        weights : np.ndarray
            Target strategy allocations to evaluate, one per scenario.

        Returns tuple[np.ndarray, np.ndarray]
            (long_daily, short_daily), each of shape (n_days, len(weights)).
        """

        # single shared column, not tiled to len(weights) - _mix_returns broadcasts a width-1 leg
        # against `weights` internally, so this avoids cumprod-ing len(weights) identical columns
        approx_mixed = self._mix_returns(
            weights,
            self._daily_long.values[:, None],
            self._daily_short.values[:, None],
        )

        # average portfolio growth factor
        port_growth = np.cumprod(1 + approx_mixed, axis=0)
        port_growth = np.vstack([np.ones((1, len(weights))), port_growth[:-1, :]]) # use capital on previous day to rebalance

        block_growth = port_growth[np.arange(0, len(self.daily), self.rebalance_period), :]
        mean_growth = block_growth.mean(axis=0)

        # scale by mean mixed growth per weight
        aums = weights * self.initial_book_cap * mean_growth

        return self._exact_leg_returns(portfolio, aums)

    def _tradeoff_profile(self, portfolio: Portfolio) -> pd.DataFrame:
        """
        Return tradeoff statistics balancing different weights of strategy to book to plot.
        Stats describe the entire strategy + book mix across weight intervals.
        No setting for long/short only - ensure Portfolio has correct l/s permissions before connecting.

        Calm-period cost filters out market crashes simplistically using a centred window returning the bottom 10% of returns.
        This approach has many issues but is eyeballed to be the best indicator using utils.crash_period_estimators.

        portfolio : Portfolio
            Strategy portfolio object.

        Returns pd.DataFrame
            Columns (sharpe, drag, maxdd, dd_days, recovery, cvar, dd_relief_per_drag, cvar_relief_per_drag)
             = Sharpe, Drag (calm-period cost), Max Drawdown, Max DD Days, DD Recovery Days, 95% CVaR, (Max DD change vs. book) / Drag, (CVaR change vs. book) / Drag.
        """

        # crash detection
        bench = self.bench.resample("D").last().dropna()
        half_window = self.lookback_window // 2
        centred_ret = bench.shift(-half_window) / bench.shift(half_window) - 1 # net move over a centred window, not just dispersion
        
        crash = centred_ret < centred_ret.quantile(0.10)
        crash.index = crash.index.date # strip tz-aware stamps + dt
        crash = crash.reindex(self.daily.index, fill_value=False).to_numpy()

        # guarantee 10% intervals for reporting - rounded before dedup since linspace(0,1,101)[30]
        weights = np.unique(np.round(np.r_[
            np.linspace(0, 1, round(1 / self.weight_intervals) + 1),
            np.linspace(0, 1, 11),
        ], 4))

        long_daily, short_daily = self._approx_aum_leg_returns(portfolio, weights)
        returns_matrix = self._mix_returns(weights, long_daily, short_daily)

        equity = np.cumprod(1 + returns_matrix, axis=0)
        peak = np.maximum.accumulate(equity, axis=0)
        drawdown = (equity - peak) / peak

        std = np.nanstd(returns_matrix, axis=0)
        sharpe = np.where(std != 0, (np.nanmean(returns_matrix, axis=0) / std) * np.sqrt(252), 0)
        
        maxdd = drawdown.min(axis=0)

        underwater = (drawdown < 0).astype(int)
        running = np.cumsum(underwater, axis=0)
        reset_base = np.maximum.accumulate(np.where(underwater == 0, running, 0), axis=0)
        dd_days = (running - reset_base).max(axis=0)

        n_days = returns_matrix.shape[0]
        rows = np.arange(n_days)[:, None]
        trough_idx = np.argmin(drawdown, axis=0)
        recovered = np.where((rows >= trough_idx[None, :]) & np.isclose(drawdown, 0), rows, n_days)
        recovery = recovered.min(axis=0) - trough_idx

        # calm-period drag: annualised return given up vs. holding the book alone on non-crash days only
        drag = (self.daily["book"].to_numpy()[~crash].mean() - returns_matrix[~crash, :].mean(axis=0)) * 252

        var_95 = np.quantile(returns_matrix, 0.05, axis=0)
        cvar = np.nanmean(np.where(returns_matrix <= var_95[None, :], returns_matrix, np.nan), axis=0)

        # cost efficiency: crash-drawdown / tail-risk relief per unit of calm-period cost paid
        # drag > 0 & relief > 0: insurance premium
        # drag > 0 & relief < 0: unconditional loss
        # drag < 0 & relief > 0: free protection
        # drag < 0 & relief < 0: free cost, worse tail risk
        dd_relief = np.divide(maxdd - maxdd[0], drag, out=np.zeros_like(drag), where=drag != 0)
        cvar_relief = np.divide(cvar - cvar[0], drag, out=np.zeros_like(drag), where=drag != 0)

        return pd.DataFrame({
            "sharpe": sharpe,
            "drag": drag,
            "maxdd": maxdd,
            "dd_days": dd_days.astype(int),
            "recovery": recovery.astype(int),
            "cvar": cvar,
            "dd_relief_per_drag": dd_relief,
            "cvar_relief_per_drag": cvar_relief,
        }, index=weights)

    def _format_tradeoff_table(self) -> str:
        """
        Format self.tradeoff_series as a standalone table.
        Only shows 10% weight increments for readability.
        
        Returns str
            Formatted table.
        """

        sample = self.tradeoff_series.loc[np.round(np.linspace(0, 1, 11), 4)]
        fmt_vals = lambda arr, **kwargs: [format_value(i, **kwargs) for i in arr]
        
        rows = {
            "Sharpe": fmt_vals(sample["sharpe"]),
            "Drag (Calm-Period Cost)": fmt_vals(sample["drag"], pct=True),
            "Max Drawdown": fmt_vals(sample["maxdd"], pct=True),
            "Max DD Days": fmt_vals(sample["dd_days"], suffix=" Days"),
            "DD Recovery Days": fmt_vals(sample["recovery"], suffix=" Days"),
            "95% CVaR": fmt_vals(sample["cvar"], pct=True),
            "DD Relief / Drag": fmt_vals(sample["dd_relief_per_drag"], suffix=" x"),
            "CVaR Relief / Drag": fmt_vals(sample["cvar_relief_per_drag"], suffix=" x"),
        }

        cols = pd.Index([f"{w:.0%}" for w in sample.index], name="Strategy Weight")

        return pd.DataFrame(rows, index=cols).T.to_string()

    def _generate_report(self, portfolio: Portfolio) -> None:
        """
        Generate portfolio integration and risk metrics.
        
        portfolio : Portfolio
            Strategy portfolio object.
        """

        columns = list(self.daily.columns)  # ["strat", "book", "bench", "combined", "optimised"]
        new_strats = ["combined", "optimised"]
        incremental_cols = ["strat"] + new_strats
        equity = (1 + self.daily).cumprod()

        self.metrics = {col: compute_series_metrics(self.daily[col], compute_drawdown(equity[col])) for col in columns}
        self.relative = {col: compute_relative_metrics(self.daily[col], self.daily["bench"]) for col in columns}
        self.incremental = {col: compute_relative_metrics(self.daily[col], self.daily["book"]) for col in incremental_cols}  # vs book, not bench

        s = self.daily[incremental_cols]
        b = self.daily["book"]

        crash_correlation_to_book = s[b <= b.quantile(0.10)].corrwith(b[b <= b.quantile(0.10)]).to_dict()
        # .corr() isn't available on a Resampler, so this keeps groupby - Grouper(freq="D") is the
        # same fast binning resample uses internally, just without materialising df.index.date
        intraday_correlation_to_book = float(self.df[["strat", "book"]].groupby(pd.Grouper(freq="D")).corr().iloc[0::2, -1].mean()) # very cursed slicing due to output of corr

        incremental_sharpe_marginal = float(self.metrics["strat"].sharpe_ratio - self.incremental["strat"].correlation * self.metrics["book"].sharpe_ratio) # sharpe heuristic: SRnew > SRold * corr
        incremental_sharpe_realised = {col: float(self.metrics[col].sharpe_ratio - self.metrics["book"].sharpe_ratio) for col in new_strats}

        mc_sample = np.random.default_rng(42).integers(0, len(self.daily), size=(10000, 252))
        cvar_mc = np.sort(self.daily.values[mc_sample], axis=1)[:, :12, :].mean(axis=(0, 1))
        cvar_monte_carlo = dict(zip(columns, cvar_mc.tolist()))

        strategy_weight = {"combined": float(self._naive_w), "optimised": float(self._opt_w)}

        non_bench = ["book", "strat"] + new_strats
        book_growth_initial = {col: (portfolio.aum if col == "strat" else self.initial_book_cap) for col in non_bench}
        total_aum_final = {col: book_growth_initial[col] * float(equity[col].iloc[-1]) for col in non_bench}

        # true pre-return starting capital just for display (when available)
        book_loc = self.book.index.get_loc(self.df.index[0])
        display_book_cap = float(self.book.iloc[book_loc - 1]) if book_loc > 0 else self.initial_book_cap
        total_aum_initial = {col: portfolio.aum if col == "strat" else display_book_cap for col in non_bench}

        strategy_aum_initial = {"combined": self._naive_w * self.initial_book_cap, "optimised": self._opt_w * self.initial_book_cap}
        strategy_aum_final = {col: total_aum_final[col] * strategy_weight[col] for col in new_strats}

        self.extras = ConnectorExtras(
            crash_correlation_to_book=crash_correlation_to_book,
            intraday_correlation_to_book=intraday_correlation_to_book,
            incremental_sharpe_marginal=incremental_sharpe_marginal,
            incremental_sharpe_realised=incremental_sharpe_realised,
            cvar_monte_carlo=cvar_monte_carlo,
            strategy_weight=strategy_weight,
            total_aum_initial=total_aum_initial,
            total_aum_final=total_aum_final,
            strategy_aum_initial=strategy_aum_initial,
            strategy_aum_final=strategy_aum_final,
        )

    def plot(self, *, savepath: Optional[str | Path] = None) -> None:
        """
        Plot cumulative equity, rolling Sharpe ratio, strategy correlation, and beta.

        savepath : Optional[str | Path] = None
            If given, this run's figure is saved under a new f"{ClassName}_{timestamp}"
            directory inside savepath, then closed. Still displayed either way.
        """

        fig, axes = plt.subplots(
            nrows=4,
            ncols=1,
            figsize=(14, 12),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1, 1, 1]}
        )
        
        rename = {
            "strat": "Strategy",
            "bench": "Benchmark",
            "optimised": "Sharpe-Optimised"
        }
        cols = {c : rename.get(c, c.capitalize()) for c in ["bench", "book", "strat", "combined", "optimised"]}

        # cumulative equity
        equity = (1 + self.daily).cumprod()

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
        for col in ["strat", "book", "optimised"]:
            roll_beta = self.daily[col].rolling(126).cov(self.daily["bench"]) / self.daily["bench"].rolling(126).var()
            axes[3].plot(roll_beta.index, roll_beta, label=f"{cols[col]}", linewidth=1)

        axes[3].axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)
        axes[3].set_title("Rolling Semi-Annual Betas (vs Benchmark)", loc="left", fontweight="bold")
        axes[3].set_ylabel("Beta")
        axes[3].legend(loc="upper left")

        for ax in axes:
            ax.label_outer()
            ax.margins(x=0)

        plt.xlabel("Date", fontweight="bold")
        plt.tight_layout()
        plt.show()

        # tradeoff: sharpe / drag / 95% es
        fig2, axes2 = plt.subplots(nrows=3, ncols=1, figsize=(14, 9), sharex=True)

        axes2[0].plot(self.tradeoff_series.index, self.tradeoff_series["sharpe"], linewidth=1)
        axes2[0].set_title("Sharpe vs. Strategy Weight", loc="left", fontweight="bold")
        axes2[0].set_ylabel("Sharpe Ratio")

        axes2[1].plot(self.tradeoff_series.index, self.tradeoff_series["drag"], linewidth=1, color="firebrick")
        axes2[1].set_title("Calm-Period Drag vs. Strategy Weight", loc="left", fontweight="bold")
        axes2[1].set_ylabel("Drag (Ann.)")
        axes2[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

        axes2[2].plot(self.tradeoff_series.index, self.tradeoff_series["cvar"], linewidth=1, color="darkorange")
        axes2[2].set_title("95% CVaR vs. Strategy Weight", loc="left", fontweight="bold")
        axes2[2].set_ylabel("CVaR")
        axes2[2].yaxis.set_major_formatter(PercentFormatter(1, decimals=1))

        for ax in axes2:
            ax.axvline(self._opt_w, color="red", linestyle=":", linewidth=1, label="Sharpe-Optimised Weight")
            ax.label_outer()
            ax.margins(x=0)
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.xaxis.set_major_locator(MultipleLocator(0.1))

        axes2[0].legend(loc="upper right", fontsize=9)
        axes2[-1].set_xlabel("Strategy Weight", fontweight="bold")
        plt.tight_layout()
        plt.show()

        # tradeoff: max drawdown / dd days / recovery days
        fig3, axes3 = plt.subplots(nrows=3, ncols=1, figsize=(14, 9), sharex=True)

        axes3[0].plot(self.tradeoff_series.index, self.tradeoff_series["maxdd"], linewidth=1)
        axes3[0].set_title("Max Drawdown vs. Strategy Weight", loc="left", fontweight="bold")
        axes3[0].set_ylabel("Max Drawdown")
        axes3[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

        axes3[1].plot(self.tradeoff_series.index, self.tradeoff_series["dd_days"], linewidth=1, color="firebrick")
        axes3[1].set_title("Max Drawdown Days vs. Strategy Weight", loc="left", fontweight="bold")
        axes3[1].set_ylabel("Days")

        axes3[2].plot(self.tradeoff_series.index, self.tradeoff_series["recovery"], linewidth=1, color="darkorange")
        axes3[2].set_title("Drawdown Recovery Days vs. Strategy Weight", loc="left", fontweight="bold")
        axes3[2].set_ylabel("Days")

        for ax in axes3:
            ax.axvline(self._opt_w, color="red", linestyle=":", linewidth=1, label="Sharpe-Optimised Weight")
            ax.label_outer()
            ax.margins(x=0)
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.xaxis.set_major_locator(MultipleLocator(0.1))

        axes3[0].legend(loc="upper right", fontsize=9)
        axes3[-1].set_xlabel("Strategy Weight", fontweight="bold")
        plt.tight_layout()
        plt.show()

        if savepath is not None:
            save_figures({
                "integration_overview": fig,
                "tradeoff_sharpe_drag_es": fig2,
                "tradeoff_drawdown_profile": fig3,
            }, type(self).__name__, savepath)

        plt.close(fig)
        plt.close(fig2)
        plt.close(fig3)

    def __str__(self) -> str:
        """
        Return integration and risk metrics.
        """

        if not self.metrics:
            return "Empty Report"

        headers = ["Strategy", "Book", "Bench", "Combined", "Optimised"]
        col_keys = ["strat", "book", "bench", "combined", "optimised"]

        cvar_mc_row = ("95% cVar (Monte Carlo)", [format_value(self.extras.cvar_monte_carlo.get(c), pct=True) for c in col_keys])

        main_groups = merge_groups(
            dataclass_rows([self.metrics[c] for c in col_keys], SeriesMetrics),
            dataclass_rows([self.relative[c] for c in col_keys], RelativeMetrics, default_section="Relative (vs Benchmark)"),
            [("Risk", [cvar_mc_row])],
        )

        # incremental: strategy/combined/optimised vs book, aligned to the main table's columns
        # correlation/market capture dropped (redundant with beta/r-squared), info ratio and tail dependency dropped (not incremental)
        incr_instances = [self.incremental.get("strat"), None, None, self.incremental.get("combined"), self.incremental.get("optimised")]
        incr_groups = dataclass_rows(incr_instances, RelativeMetrics, default_section="Incremental (vs Book)")
        keep_labels = {"Alpha", "Beta", "R-Squared", "Idiosyncratic Risk"}

        incremental_rows = [
            row
            for title, rows in incr_groups
            if title == "Incremental (vs Book)"
            for row in rows
            if row[0] in keep_labels
        ]

        incremental_rows.append(("Incremental Sharpe (Marginal)", [format_value(self.extras.incremental_sharpe_marginal), "-", "-", "-", "-"]))
        incremental_rows.append(("Incremental Sharpe (Realised)", ["-", "-", "-", format_value(self.extras.incremental_sharpe_realised.get("combined")), format_value(self.extras.incremental_sharpe_realised.get("optimised"))]))

        # remaining connector-only fields, aligned to the main table columns
        other_rows = []
        skip_fields = {"incremental_sharpe_marginal", "incremental_sharpe_realised", "cvar_monte_carlo"}

        for f in dataclasses.fields(ConnectorExtras):
            if f.name in skip_fields:
                continue

            value = getattr(self.extras, f.name)
            label = f.metadata.get("label", f.name)
            pct = f.metadata.get("pct", False)

            values = [format_value(value.get(c), pct=pct) for c in col_keys] if isinstance(value, dict) else [format_value(value, pct=pct)] + ["-"] * (len(col_keys) - 1)
            other_rows.append((label, values))

        all_groups = main_groups + [("Incremental (vs Book)", incremental_rows), ("Other", other_rows)]

        tradeoff_table = self._format_tradeoff_table()

        return f"{render_sections(headers, all_groups)}\n\n> Strategy/Book Weight Tradeoff Profile\n{tradeoff_table}"
