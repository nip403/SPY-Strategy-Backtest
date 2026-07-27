from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from .base import AnalysisReport
from .metrics import SeriesMetrics, RelativeMetrics, ConnectorExtras, compute_series_metrics, compute_relative_metrics, format_value, dataclass_rows, merge_groups, render_sections
from ..utils import compute_drawdown, save_figures

warnings.filterwarnings("ignore", category=FutureWarning)

class StrategyConnector(AnalysisReport):
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

        self.metrics: dict[str, SeriesMetrics] = {}
        self.relative: dict[str, RelativeMetrics] = {}
        self.incremental: dict[str, RelativeMetrics] = {}
        self.extras: ConnectorExtras | None = None

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
        intraday_correlation_to_book = float(self.df[["strat", "book"]].groupby(self.df.index.date).corr().iloc[0::2, -1].mean()) # very cursed slicing due to output of corr

        incremental_sharpe_marginal = float(self.metrics["strat"].sharpe_ratio - self.incremental["strat"].correlation * self.metrics["book"].sharpe_ratio) # sharpe heuristic: SRnew > SRold * corr
        incremental_sharpe_realised = {col: float(self.metrics[col].sharpe_ratio - self.metrics["book"].sharpe_ratio) for col in new_strats}

        mc_sample = np.random.default_rng(42).integers(0, len(self.daily), size=(10000, 252))
        cvar_mc = np.sort(self.daily.values[mc_sample], axis=1)[:, :12, :].mean(axis=(0, 1))
        cvar_monte_carlo = dict(zip(columns, cvar_mc.tolist()))

        strategy_weight = {"combined": float(self._naive_w), "optimised": float(self._opt_w)}

        non_bench = ["book", "strat"] + new_strats
        book_growth_initial = {col: (self.portfolio.aum if col == "strat" else self.initial_book_cap) for col in non_bench}
        total_aum_final = {col: book_growth_initial[col] * float(equity[col].iloc[-1]) for col in non_bench}

        # true pre-return starting capital for display; self.initial_book_cap already has the book's first-bar
        # return compounded in (no anchor row precedes it), so look one bar further back in the book's raw history
        # when available (falls back when the book has no lead-in data before the analysis window, e.g. synthetic data)
        book_loc = self.book.index.get_loc(self.df.index[0])
        display_book_cap = float(self.book.iloc[book_loc - 1]) if book_loc > 0 else self.initial_book_cap
        total_aum_initial = {col: (self.portfolio.aum if col == "strat" else display_book_cap) for col in non_bench}

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
