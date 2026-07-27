from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from .base import AnalysisReport
from .metrics import SeriesMetrics, TradeMetrics, RelativeMetrics, compute_series_metrics, compute_trade_metrics, compute_relative_metrics, dataclass_rows, merge_groups, render_sections

class Tearsheet(AnalysisReport):
    def __init__(self, df: pd.DataFrame) -> None:
        """
        Calculate performance and risk metrics from portfolio returns.

        df : pd.DataFrame
            Portfolio statistics containing strategy and benchmark returns,
            drawdowns, and trade counts.
        """

        self._ret = df[["strat_ret", "bench_ret"]]

        self.strategy = compute_series_metrics(df["strat_ret"], df["strat_dd"])
        self.benchmark = compute_series_metrics(df["bench_ret"], df["bench_dd"])
        self.trades = compute_trade_metrics(df["trade_count"], df["trade_wins"], self.strategy.cum_return)
        self.relative = compute_relative_metrics(df["strat_ret"], df["bench_ret"])

    def plot(self) -> list[plt.Figure]:
        """
        Plot daily return distributions for strategy and benchmark.

        Returns list[plt.Figure]
            The generated figure.
        """

        ret = self._ret
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.hist(ret["strat_ret"], bins=51, color="blue", alpha=0.45, label="Strategy", edgecolor="none", density=True)
        ax.hist(ret["bench_ret"], bins=51, color="orange", alpha=0.45, label="Benchmark", edgecolor="none", density=True)

        ax.set_title("Daily Returns Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Daily Return", fontsize=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.legend(loc="upper left")
        xlim = max(ret["strat_ret"].abs().max(), ret["bench_ret"].abs().max()) * 1.05
        ax.set_xlim(-xlim, xlim)

        stats_text = (
            "--- Strategy ---\n"
            f"Mean: {self.strategy.avg_daily_return * 100:.3f}%\n"
            f"Std:  {ret['strat_ret'].std() * 100:.3f}%\n"
            f"Skew: {self.strategy.skew:.3f}\n"
            f"Kurt: {self.strategy.kurt:.3f}\n\n"
            "--- Benchmark ---\n"
            f"Mean: {self.benchmark.avg_daily_return * 100:.3f}%\n"
            f"Std:  {ret['bench_ret'].std() * 100:.3f}%\n"
            f"Skew: {self.benchmark.skew:.3f}\n"
            f"Kurt: {self.benchmark.kurt:.3f}"
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

        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.1f}%"))

        plt.tight_layout()
        plt.show()

        return [fig]

    def __str__(self) -> str:
        """
        Return a formatted text tearsheet summary.
        """

        groups = merge_groups(
            dataclass_rows([self.strategy, self.benchmark], SeriesMetrics),
            dataclass_rows([self.trades, None], TradeMetrics, default_section="Trades"),
            dataclass_rows([self.relative, None], RelativeMetrics, default_section="Relative (vs Benchmark)"),
        )

        return render_sections(["Strategy", "Benchmark"], groups)
