from __future__ import annotations

from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from datetime import date
from pathlib import Path
from typing import Optional
from .base import AnalysisReport
from .tearsheet import Tearsheet
from .metrics import SeriesMetrics, TradeMetrics, RelativeMetrics, dataclass_rows, merge_groups, render_sections
from ..utils import trade_stats, compute_drawdown, save_figures

class PortfolioDecomposer(AnalysisReport):
    def __init__(self, portfolio: Portfolio, start_date: date, end_date: date) -> None:
        """
        Perform long/short decomposition of a backtested portfolio.
        Transaction costs are assigned proportionally, and tearsheets are generated for each component.

        portfolio : Portfolio
            Backtested portfolio instance to decompose.
        start_date : date
            Start date of target backtest period.
        end_date : date
            End date of target backtest period.
        """

        self.aum = portfolio.aum
        df = portfolio.df.loc[str(start_date): str(end_date)]

        # long/short split
        legs = self.split_long_short(*(col[:, None] for col in df[["position", "ret", "gross_ret", "net_ret"]].to_numpy().T))

        positions = {leg: pd.Series(legs[leg]["position"][:, 0], index=df.index) for leg in ["long", "short"]}
        net_ret = {leg: pd.Series(legs[leg]["net_ret"][:, 0], index=df.index) for leg in ["long", "short"]}

        # prepare tearsheets for each component
        valid_days = df.index.floor("D").unique() # resample("D") bins every calendar day incl. weekends; restrict to days actually present

        def build(ret: pd.Series, pos: pd.Series) -> pd.DataFrame:
            equity = (1 + ret).cumprod().resample("D").last().loc[valid_days]
            equity.index = equity.index.date

            daily_ret = equity.pct_change().fillna(0)
            daily_ret.iloc[0] = equity.iloc[0] - 1

            dd = compute_drawdown(equity)

            trades = trade_stats(pos, ret)[["trade_count", "trade_wins"]].resample("D").sum().loc[valid_days].astype(int)
            trades.index = trades.index.date

            return pd.DataFrame({
                "strat_ret": daily_ret,
                "strat_dd": dd,
                "trade_count": trades["trade_count"],
                "trade_wins": trades["trade_wins"],
            })

        append_bench = lambda leg_df: pd.concat([leg_df, portfolio.stats.loc[start_date: end_date, ["bench_ret", "bench_dd"]]], axis=1)

        long_df = append_bench(build(net_ret["long"], positions["long"]))
        short_df = append_bench(build(net_ret["short"], positions["short"]))

        self.components: dict[str, Tearsheet] = {
            "strategy": Tearsheet(portfolio.stats.loc[start_date: end_date]),
            "long": Tearsheet(long_df),
            "short": Tearsheet(short_df),
        }

        self._plot_data = pd.concat(
            [
                portfolio.stats.loc[start_date: end_date, ["strat_ret", "bench_ret"]],
                long_df["strat_ret"].rename("long"),
                short_df["strat_ret"].rename("short"),
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
    
    @classmethod
    def split_long_short(cls, position: np.ndarray, ret: np.ndarray, gross: np.ndarray, net: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        """
        Decompose position/gross/net return matrices into independent long and short legs.

        Costs are attributed proportionally to each leg's share of that bar's position change
            
        position, ret, gross, net : np.ndarray
            (n_bars, n_scenarios) arrays sharing Portfolio.df's "position"/"ret"/"gross_ret"/"net_ret" series (or their matrix equivalents from Portfolio.returns_matrix_components). 
            Ret may be (n_bars, 1) to broadcast across scenarios.

        Returns dict[str, dict[str, np.ndarray]]
            {"long": {"position": ..., "net_ret": ...}, "short": {...}}, each (n_bars, n_scenarios).
        """

        pos_long, pos_short = np.clip(position, 0, None), np.clip(position, None, 0)

        gross_long = np.vstack([np.zeros((1, pos_long.shape[1])), pos_long[:-1]]) * ret
        gross_short = np.vstack([np.zeros((1, pos_short.shape[1])), pos_short[:-1]]) * ret

        delta_long = np.abs(np.diff(pos_long, axis=0, prepend=pos_long[:1]))
        delta_short = np.abs(np.diff(pos_short, axis=0, prepend=pos_short[:1]))
        
        total_delta = delta_long + delta_short
        cost = gross - net
        
        cost_long = np.divide(delta_long, total_delta, out=np.zeros_like(cost), where=(total_delta != 0)) * cost
        cost_short = np.divide(delta_short, total_delta, out=np.zeros_like(cost), where=(total_delta != 0)) * cost

        return {
            "long": {"position": pos_long, "net_ret": gross_long - cost_long},
            "short": {"position": pos_short, "net_ret": gross_short - cost_short},
        }

    def plot(self, *, savepath: Optional[str | Path] = None) -> None:
        """
        Plot portfolio component performance distributions.
        Displays cumulative equity curves and returns histograms.

        The KDE plot converts each return histogram into a smooth density curve.
        This allows multiple distributions to be compared on a single chart without too much visual clutter.

        savepath : Optional[str | Path] = None
            If given, this run's figures are saved under a new f"{ClassName}_{timestamp}"
            directory inside savepath, then closed. Still displayed either way.
        """

        ret = self._plot_data

        # equity curve
        fig1, ax = plt.subplots(figsize=(14, 10))

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
        ax.legend(loc="upper left")

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.set_xlim(ret.index.min(), ret.index.max())
        ax.margins(x=0)

        plt.tight_layout()
        plt.show()

        # Histogram
        fig2 = plt.figure(figsize=(14, 10))

        gs = fig2.add_gridspec(2, 2, top=0.91, bottom=0.07, left=0.07, right=0.93, wspace=0.20, hspace=0.20)

        fig2.suptitle("Daily Returns Distributions vs Benchmark (SymLog-Scaled)", fontsize=14, fontweight="bold", y=0.96)

        hist_configs = [
            {"label": "Strategy vs Benchmark", "series_col": "Strategy", "pos": gs[0, 0], "metrics": self.components["strategy"].strategy},
            {"label": "Long-Only vs Benchmark", "series_col": "Long-Only", "pos": gs[0, 1], "metrics": self.components["long"].strategy},
            {"label": "Short-Only vs Benchmark", "series_col": "Short-Only", "pos": gs[1, 0], "metrics": self.components["short"].strategy},
        ]

        bench_metrics = self.components["strategy"].benchmark

        for config in hist_configs:
            ax_sub = fig2.add_subplot(config["pos"])

            xlim = max(ret[config["series_col"]].abs().std(), ret["Benchmark"].abs().std()) * 4.5
            bins = np.linspace(-xlim, xlim, 51)

            n_series, _, _ = ax_sub.hist(ret[config["series_col"]].values, bins=bins, color=colours[config["series_col"]], alpha=0.4, histtype="stepfilled", density=True, align="mid", label=config["series_col"])
            n_bench, _, _ = ax_sub.hist(ret["Benchmark"].values, bins=bins, color=colours["Benchmark"], alpha=0.25, histtype="stepfilled", density=True, align="mid", label="Benchmark")

            ax_sub.set_yscale("symlog", linthresh=max(max(n_series.max(), n_bench.max()) * 0.1, 1e-5))

            ax_sub.set_title(config["label"], fontsize=11, fontweight="bold")
            ax_sub.set_xlim(-xlim, xlim)
            ax_sub.margins(x=0)
            ax_sub.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x * 100:.1f}%"))
            ax_sub.legend(loc="upper left", fontsize=9)

            m = config["metrics"]
            stats_text = (
                f"[{config['series_col']}]\n"
                f"Mean: {m.avg_daily_return * 100:.3f}%\n"
                f"Std:  {m.ann_vol / np.sqrt(252) * 100:.2f}%\n"
                f"Skew: {m.skew:.2f}\n"
                f"Kurt: {m.kurt:.2f}\n\n"
                f"[Benchmark]\n"
                f"Mean: {bench_metrics.avg_daily_return * 100:.3f}%\n"
                f"Std:  {bench_metrics.ann_vol / np.sqrt(252) * 100:.2f}%\n"
                f"Skew: {bench_metrics.skew:.2f}\n"
                f"Kurt: {bench_metrics.kurt:.2f}"
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

        ax_kde = fig2.add_subplot(gs[1, 1])
        global_xlim = ret["Strategy"].abs().std() * 4.5

        xs = np.linspace(-global_xlim, global_xlim, 200)

        for col in ret.columns:
            kde = gaussian_kde(ret[col].values)
            ax_kde.plot(xs, kde(xs), label=col, color=colours[col], linewidth=1)

        ax_kde.set_title("Overlayed (Continuous, Gaussian KDE)", fontsize=11, fontweight="bold")
        ax_kde.set_xlabel("Daily Return", fontsize=10)
        ax_kde.set_ylabel("Probability Density", fontsize=10)
        ax_kde.set_xlim(-global_xlim, global_xlim)
        ax_kde.margins(x=0)
        ax_kde.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x * 100:.1f}%"))
        ax_kde.legend(loc="upper right", fontsize=9)

        plt.show()

        if savepath is not None:
            save_figures({"equity_curves": fig1, "returns_distributions": fig2}, type(self).__name__, savepath)

        plt.close(fig1)
        plt.close(fig2)

    def __str__(self) -> str:
        """
        Format decomposition tearsheets as a comparison table.
        Reports strategy, long-only, short-only, and benchmark metrics.
        """

        strat = self.components["strategy"]
        long_ = self.components["long"]
        short = self.components["short"]

        headers = ["Strategy", "Long-Only", "Short-Only", "Benchmark"]

        groups = merge_groups(
            dataclass_rows([strat.strategy, long_.strategy, short.strategy, strat.benchmark], SeriesMetrics),
            dataclass_rows([strat.trades, long_.trades, short.trades, None], TradeMetrics, default_section="Trades"),
            dataclass_rows([strat.relative, long_.relative, short.relative, None], RelativeMetrics, default_section="Relative (vs Benchmark)"),
        )

        return render_sections(headers, groups)
