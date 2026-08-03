from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional, Any
import matplotlib.pyplot as plt
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
        "sweep_intervals": 0.01,
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
        """
        
        config = {**self.DEFAULT_PARAMS, **(config or {})}
        
        # for self._tradeoff_profile, consider adding switches (just override after init in current implementation)
        self.lookback_window = config["lookback_window"]
        self.sweep_intervals = config["sweep_intervals"]

        self.portfolio = strategy_portfolio
        self.book = book_equity
        self.bench = benchmark_equity
        self.rebalance_period = config["rebalance_period"]

        # returns comparison df - long/short legs are internal only
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

        valid_days = self.df.index.floor("D").unique()
        self.daily = ((1 + self.df).resample("D").prod() - 1).loc[valid_days]
        self.daily.index = self.daily.index.date

        # internal-only long/short daily legs
        daily_long, daily_short = self._exact_leg_returns(np.array([self.portfolio.aum]))
        self._daily_long = pd.Series(daily_long[:, 0], index=self.daily.index)
        self._daily_short = pd.Series(daily_short[:, 0], index=self.daily.index)
        self._has_long_leg = bool((self._daily_long != 0).any())

        # create bundled portfolios - assumes regular rebalancing to given weight every self.rebalance_period days
        # NOTE: weights are BACKWARDS looking, set to base off book/strategy starting equity
        self._naive_w = 0.5
        naive_long, naive_short = self._exact_block_leg_returns(self._naive_w)
        self.daily["combined"] = self._mix_returns(np.array([self._naive_w]), naive_long, naive_short)[:, 0] # naive mix: target weight = initial starting capital proportions

        self._opt_w, self.daily["optimised"] = self._optimise_weight()

        # reporting
        self.metrics: dict[str, SeriesMetrics] = {}
        self.relative: dict[str, RelativeMetrics] = {}
        self.incremental: dict[str, RelativeMetrics] = {}
        self.extras: ConnectorExtras | None = None
        
        self.tradeoff_series = self._tradeoff_profile()

        self._generate_report()

    def _optimise_weight(self, depth: int = 3, points: int = 11) -> tuple[float, pd.Series]:
        """
        Optimise strategy allocation weight by Sharpe ratio using an iterative 1D grid search.
        Searches over the weight used to size both legs, and selects the allocation maximising sharpe.

        Uses an AUM mean scaling approximation to project cost changes as strategy is rebalanced to avoid ridiculously high memory needs..

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

            approx_mixed = self._mix_returns(
                weights,
                np.tile(self._daily_long.values[:, None], (1, points)),
                np.tile(self._daily_short.values[:, None], (1, points)),
            )

            # average portfolio growth factor
            port_growth = np.cumprod(1 + approx_mixed, axis=0)
            port_growth = np.vstack([np.ones((1, points)), port_growth[:-1, :]]) # use capital on previous day to rebalance

            block_growth = port_growth[np.arange(0, len(self.daily), self.rebalance_period), :]
            mean_growth = block_growth.mean(axis=0)

            # scale by mean mixed growth per weight
            aums = weights * self.initial_book_cap * mean_growth

            long_daily, short_daily = self._exact_leg_returns(aums)
            mixed = self._mix_returns(weights, long_daily, short_daily)

            std = np.nanstd(mixed, axis=0)
            sharpes = np.where(std != 0, (np.nanmean(mixed, axis=0) / std) * np.sqrt(252), 0)

            best_idx = np.argmax(sharpes)
            best_weight = weights[best_idx]

            span = (hi - lo) / (points - 1)
            lo = max(0, best_weight - span)
            hi = min(1, best_weight + span)

        final_long, final_short = self._exact_block_leg_returns(best_weight)
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
            Matrix of long-leg daily returns across allocation scenarios. Ignored when self._has_long_leg is False.
        short_daily_matrix : np.ndarray
            Matrix of short-leg daily returns across allocation scenarios.

        Returns np.ndarray
            Matrix of mixed portfolio daily returns. Shape (n_days, scenarios)
        """

        n, k = short_daily_matrix.shape
        book_daily = self.daily["book"].values[:, None] # reshape to broadcast

        # component cumprods
        cum_s = np.empty((n + 1, k))
        cum_b = np.empty((n + 1, 1))

        cum_s[0, :] = cum_b[0, 0] = 1

        np.cumprod(1 + short_daily_matrix, axis=0, out=cum_s[1:, :])
        np.cumprod(1 + book_daily, axis=0, out=cum_b[1:, :])

        rebals = (np.arange(n) // self.rebalance_period) * self.rebalance_period # rebalance windows, element = start index of block

        # short leg's dollar P&L, layered on top (margin-funded, additive) regardless of the long leg
        short_term = weights * (cum_s[1:, :] / cum_s[rebals, :] - 1)

        if self._has_long_leg:
            cum_l = np.empty((n + 1, k))
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

    def _exact_leg_returns(self, aums: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates capacity-aware daily long/short leg returns at each AUM.

        aums : np.ndarray
            Portfolio capital values to evaluate, one per scenario.

        Returns tuple[np.ndarray, np.ndarray]
            (long_daily, short_daily), shape (n_days, len(aums)).
        """

        position_matrix, gross_matrix, net_matrix = self.portfolio.returns_matrix_components(aums)
        ret = self.portfolio.df["ret"].to_numpy()[:, None]

        legs = PortfolioDecomposer.split_long_short(position_matrix, ret, gross_matrix, net_matrix)

        def to_daily(net_ret_matrix: np.ndarray) -> np.ndarray:
            daily = pd.DataFrame(1 + net_ret_matrix, index=self.portfolio.df.index).resample("D").prod() - 1
            daily.index = daily.index.date # match self.daily's date-object index so .loc below aligns correctly

            return daily.loc[self.daily.index].values

        return to_daily(legs["long"]["net_ret"]), to_daily(legs["short"]["net_ret"])

    def _exact_block_leg_returns(self, weight: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the exact cost-adjusted daily long/short leg returns for a rebalanced
        strategy integration. Recomputes leg returns using exact AUM at each rebalance, scaling
        strategy capital with true mixed portfolio growth to maintain the target weight.

        weight : float
            Target strategy allocation.

        Returns tuple[np.ndarray, np.ndarray]
            (long_daily, short_daily), each of shape (n_days, 1).
        """

        approx_mixed = self._mix_returns(np.array([weight]), self._daily_long.values[:, None], self._daily_short.values[:, None])[:, 0]

        # portfolio & book growth at the start of each rebalance period
        port_growth = pd.Series(approx_mixed).add(1).cumprod().shift(1).fillna(1).values
        block_growth = port_growth[np.arange(0, len(self.daily), self.rebalance_period)]
        block_idx = np.arange(len(self.daily)) // self.rebalance_period

        exact_aums = weight * self.initial_book_cap * block_growth

        long_daily, short_daily = self._exact_leg_returns(exact_aums)

        pick = lambda m: m.reshape(len(self.daily), len(block_growth), 1)[np.arange(len(self.daily)), block_idx]

        return pick(long_daily), pick(short_daily)
    
    def _tradeoff_profile(self) -> pd.DataFrame:
        """
        Return tradeoff statistics balancing different weights of strategy to book to plot.
        Stats describe the entire strategy + book mix across weight intervals.
        No setting for long/short only - ensure Portfolio has correct l/s permissions before connecting.
        
        Calm-period cost filters out market crashes simplistically using a centred window returning the bottom 10% of returns.
        This approach has many issues but is eyeballed to be the best indicator using utils.crash_period_estimators.
    
        Returns pd.DataFrame
            Columns (sharpe, drag, maxdd, dd_days, recovery, es)
             = Sharpe, Drag (calm-period cost), Max Drawdown, Max DD. Days, Exp. Shortfall, DD. Recovery Days, Expected Shortfall.  
        
        
        steps:
        Have init call and save the tradeoff profile, 
        and print a separate table with the tradeoff stats in str. 
        Generate more figures in plot: one for sharpe/drag/es, and one for maxdd, days, and recovery.
        """
        
        bench = self.benchmark.resample("D").last().dropna()
        half_window = self.lookback_window // 2
        centred_ret = bench.shift(-half_window) / bench.shift(half_window) - 1 # net move over a centred window, not just dispersion
        crash = centred_ret < centred_ret.quantile(0.10)
        
        scenarios = np.linspace(0, 1, 1 / self.sweep_intervals)
        #returns_matrix = self._mix_returns(scenarios, self._daily_long, self._daily_short)
        
        
        #out = pd.DataFrame([], columns=scenarios)
        
        return #pd.DataFrame(masks)

    def _generate_report(self) -> None:
        """
        Generate portfolio integration and risk metrics.
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
        book_growth_initial = {col: (self.portfolio.aum if col == "strat" else self.initial_book_cap) for col in non_bench}
        total_aum_final = {col: book_growth_initial[col] * float(equity[col].iloc[-1]) for col in non_bench}

        # true pre-return starting capital just for display (when available)
        book_loc = self.book.index.get_loc(self.df.index[0])
        display_book_cap = float(self.book.iloc[book_loc - 1]) if book_loc > 0 else self.initial_book_cap
        total_aum_initial = {col: self.portfolio.aum if col == "strat" else display_book_cap for col in non_bench}

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

        if savepath is not None:
            save_figures({"integration_overview": fig}, type(self).__name__, savepath)

        plt.close(fig)

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
        
        return render_sections(headers, all_groups)
