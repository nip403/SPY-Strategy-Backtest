from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd
import numpy as np
from ..utils import _safe_div

@dataclass
class SeriesMetrics:
    """
    Performance/risk statistics for a single return series.
    Decoupled from trades or a comparison series, both of which are separate dataclasses.
    """

    total_days: int = field(metadata={"label": "Total Days"})
    cum_return: float = field(metadata={"label": "Cum. Return", "pct": True})
    ann_return: float = field(metadata={"label": "Ann. Return", "pct": True})  # geometric CAGR
    avg_daily_return: float = field(metadata={"label": "Avg. Daily Return", "pct": True})  # arithmetic, not annualised
    skew: float = field(metadata={"label": "Ret. Skew"})
    kurt: float = field(metadata={"label": "Ret. Kurtosis"})
    max_gain: float = field(metadata={"label": "Max Gain", "pct": True})
    best_day: Any = field(metadata={"label": "Best Day"})
    max_loss: float = field(metadata={"label": "Max Loss", "pct": True})
    worst_day: Any = field(metadata={"label": "Worst Day"})
    daily_win_rate: float = field(metadata={"label": "Daily Win Rate", "pct": True})
    ann_vol: float = field(metadata={"label": "Ann. Volatility", "pct": True})
    max_drawdown: float = field(metadata={"label": "Max Drawdown", "pct": True})
    max_dd_days: int = field(metadata={"label": "Max DD Days", "suffix": " Days"})
    var_95pct: float = field(metadata={"label": "95% VaR", "pct": True})
    cvar: float = field(metadata={"label": "Expected Shortfall"})
    sharpe_ratio: float = field(metadata={"label": "Sharpe Ratio"})
    sortino_ratio: float = field(metadata={"label": "Sortino Ratio"})
    calmar_ratio: float = field(metadata={"label": "Calmar Ratio"})

@dataclass
class TradeMetrics:
    """
    Discrete-trade statistics.
    """

    win_rate: float = field(metadata={"label": "Win Rate", "pct": True})
    trades_per_day: float = field(metadata={"label": "Average Trades / Day"})
    return_per_trade: float = field(metadata={"label": "Average Return / Trade"})
    total_trades: int = field(metadata={"label": "Total Trades"})

@dataclass
class RelativeMetrics:
    """
    Comparison statistics comparing return series against a reference series.
    """

    alpha: float = field(metadata={"label": "Alpha"})
    beta: float = field(metadata={"label": "Beta"})
    r_squared: float = field(metadata={"label": "R-Squared"})
    information_ratio: float = field(metadata={"label": "Information Ratio"})
    idiosyncratic_risk: float = field(metadata={"label": "Idiosyncratic Risk", "pct": True})  # residual vol after removing beta*reference + alpha/252

@dataclass
class DailySnapshot:
    """Single-day cumulative-return snapshot."""

    strat_cum_return: float
    bench_cum_return: float

    def __str__(self) -> str:
        return f"Strategy: {format_value(self.strat_cum_return, pct=True)}   Benchmark: {format_value(self.bench_cum_return, pct=True)}"

@dataclass
class ConnectorExtras:
    """
    StrategyConnector fields not covered by Tearsheet/SeriesMetrics/RelativeMetrics.
    Fields are keyed by column name.
    """

    max_dd_recovery_days: dict[str, int] = field(metadata={"label": "Max DD Recovery Days"})
    correlation_to_book: dict[str, float] = field(metadata={"label": "Correlation to Book"})
    crash_correlation_to_book: dict[str, float] = field(metadata={"label": "Crash Correlation to Book"})
    intraday_correlation_to_book: float = field(metadata={"label": "Intraday Correlation to Book"})
    incremental_sharpe_marginal: float = field(metadata={"label": "Incremental Sharpe (Marginal)"})
    incremental_sharpe_realised: dict[str, float] = field(metadata={"label": "Incremental Sharpe (Realised)"})
    lower_tail_dependency: dict[str, float] = field(metadata={"label": "Lower Tail Dependency"})
    upper_tail_dependency: dict[str, float] = field(metadata={"label": "Upper Tail Dependency"})
    cvar_monte_carlo: dict[str, float] = field(metadata={"label": "95% cVar (Monte Carlo)", "pct": True})
    up_market_capture: dict[str, float] = field(metadata={"label": "Up-Market Capture"})
    down_market_capture: dict[str, float] = field(metadata={"label": "Down-Market Capture"})
    strategy_weight: dict[str, float] = field(metadata={"label": "Strategy Weight", "pct": True})
    total_aum_initial: dict[str, float] = field(metadata={"label": "Total AUM (Initial)"})
    total_aum_final: dict[str, float] = field(metadata={"label": "Total AUM (Final)"})
    strategy_aum_initial: dict[str, float] = field(metadata={"label": "Strategy AUM (Initial)"})
    strategy_aum_final: dict[str, float] = field(metadata={"label": "Strategy AUM (Final)"})

def compute_series_metrics(returns: pd.Series, drawdown: pd.Series) -> SeriesMetrics:
    """
    Compute performance/risk statistics for a single return series.

    returns : pd.Series
        Daily return series.
    drawdown : pd.Series
        Aligned drawdown series (see compute_drawdown).

    Returns SeriesMetrics
        Populated statistics.
    """

    days = len(returns)
    cum_return = float((1 + returns).prod() - 1)
    ann_return = (1 + cum_return) ** (252 / days) - 1 if days else float("nan")  # geometric
    avg_daily_return = float(returns.mean())
    ann_vol = float(returns.std()) * np.sqrt(252)
    max_dd = float(drawdown.min())

    underwater = drawdown < 0
    state_changes = (underwater != underwater.shift()).cumsum()
    max_dd_days = int(drawdown.where(underwater).groupby(state_changes).count().max())

    var_95 = float(np.percentile(returns, 5))
    cvar = float(np.nanmean(np.where(returns <= var_95, returns, np.nan)))

    ann_downside_vol = float(returns[returns < 0].std()) * np.sqrt(252)

    return SeriesMetrics(
        total_days=days,
        cum_return=cum_return,
        ann_return=ann_return,
        avg_daily_return=avg_daily_return,
        skew=float(returns.skew()),
        kurt=float(returns.kurt()),
        max_gain=float(returns.max()),
        best_day=returns.idxmax(),
        max_loss=float(returns.min()),
        worst_day=returns.idxmin(),
        daily_win_rate=float((returns > 0).mean()),
        ann_vol=ann_vol,
        max_drawdown=max_dd,
        max_dd_days=max_dd_days,
        var_95pct=var_95,
        cvar=cvar,
        sharpe_ratio=_safe_div(avg_daily_return * 252, ann_vol),
        sortino_ratio=_safe_div(ann_return, ann_downside_vol),
        calmar_ratio=_safe_div(ann_return, abs(max_dd)),
    )

def compute_trade_metrics(trade_count: pd.Series, trade_wins: pd.Series, cum_return: float) -> TradeMetrics:
    """
    Compute discrete-trade statistics.

    trade_count : pd.Series
        1 on bars where a trade closes, else 0.
    trade_wins : pd.Series
        1 on bars where a winning trade closes, else 0.
    cum_return : float
        Total compounded return over the period, used for return-per-trade.

    Returns TradeMetrics
        Populated statistics.
    """

    total_trades = int(trade_count.sum())

    return TradeMetrics(
        win_rate=_safe_div(float(trade_wins.sum()), total_trades),
        trades_per_day=float(trade_count.mean()),
        return_per_trade=_safe_div(cum_return, total_trades),
        total_trades=total_trades,
    )

def compute_relative_metrics(returns: pd.Series, reference: pd.Series) -> RelativeMetrics:
    """
    Compute comparison statistics for a return series against a reference series.

    returns : pd.Series
        Return series being evaluated.
    reference : pd.Series
        Reference/benchmark return series to compare against.

    Returns RelativeMetrics
        Populated statistics.
    """

    beta = _safe_div(returns.cov(reference), reference.var())
    alpha = float((returns.mean() - beta * reference.mean()) * 252)
    correlation = _safe_div(returns.cov(reference), returns.std() * reference.std())  # avoids pandas .corr()'s unguarded internal np.corrcoef division
    r_squared = float(correlation ** 2)

    tracking_error = (returns - reference).std() * np.sqrt(252)
    information_ratio = _safe_div((returns.mean() - reference.mean()) * 252, tracking_error)

    residual = returns - (reference * beta + alpha / 252)
    idiosyncratic_risk = float(residual.std()) * np.sqrt(252)

    return RelativeMetrics(
        alpha=alpha,
        beta=float(beta),
        r_squared=r_squared,
        information_ratio=information_ratio,
        idiosyncratic_risk=idiosyncratic_risk,
    )
    
##### Print/Output Pipeline #####

def format_value(value: Any, *, pct: bool = False, suffix: str = "") -> str:
    """
    Format a single metric value for display.

    value : Any
        Value to format (float, int, Timestamp, None, or NaN).
    pct : bool = False
        Whether to render as a percentage.
    suffix : str = ""
        Literal text appended after formatting (e.g. " Days").

    Returns str
        Formatted display string. "-" for None/NaN.
    """

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"

    if pct and isinstance(value, (int, float)):
        return f"{value * 100:.2f}%"

    if isinstance(value, (int, np.integer)):
        return f"{value:,}{suffix}"

    if isinstance(value, float):
        return f"{value:.2f}{suffix}"

    return f"{value}{suffix}"

def metric_rows(instance: Any, *, columns: int, at: int) -> list[tuple[str, bool, str, list[Any]]]:
    """
    Turn one dataclass instance's fields into row tuples for render_table.
    Values are placed at position "at" across "columns" total columns (other positions left as None).

    instance : Any
        A dataclass instance (SeriesMetrics, TradeMetrics, or RelativeMetrics).
    columns : int
        Total number of columns in the target table.
    at : int
        Column index this instance's values should populate.

    Returns list[tuple[str, bool, str, list[Any]]]
        (label, pct, suffix, values) rows for render_table.
    """

    rows = []

    for f in dataclasses.fields(instance):
        values = [None] * columns
        values[at] = getattr(instance, f.name)
        rows.append((f.metadata.get("label", f.name), f.metadata.get("pct", False), f.metadata.get("suffix", ""), values))

    return rows

def paired_rows(strategy: Any, benchmark: Optional[Any]) -> list[tuple[str, bool, str, list[Any]]]:
    """
    Turn a strategy/benchmark pair of the *same* dataclass type into 2-column row tuples.
    Used for SeriesMetrics.

    strategy : Any
        Strategy-side dataclass instance.
    benchmark : Optional[Any]
        Benchmark-side dataclass instance (or None to show "-").

    Returns list[tuple[str, bool, str, list[Any]]]
        (label, pct, suffix, [strategy_value, benchmark_value]) rows.
    """

    rows = []

    for f in dataclasses.fields(strategy):
        s_val = getattr(strategy, f.name)
        b_val = getattr(benchmark, f.name) if benchmark is not None else None
        rows.append((f.metadata.get("label", f.name), f.metadata.get("pct", False), f.metadata.get("suffix", ""), [s_val, b_val]))

    return rows

def render_table(headers: list[str], rows: list[tuple[str, bool, str, list[Any]]], *, title: Optional[str] = None) -> str:
    """
    Render a fixed-width text table shared by every AnalysisReport subclass.

    headers : list[str]
        Column header labels.
    rows : list[tuple[str, bool, str, list[Any]]]
        (label, pct, suffix, values) rows, values has len(headers) entries (None for N/A cells).
    title : Optional[str] = None
        Optional title line prepended above the header row.

    Returns str
        Formatted table.
    """

    w_metric, w_col = 25, 15

    lines = [title] if title else []
    lines.append(f"{'':<{w_metric}}" + "".join(f"{h:>{w_col}}" for h in headers))

    for label, pct, suffix, values in rows:
        line = f"{label:<{w_metric}}"

        for value in values:
            line += f"{format_value(value, pct=pct, suffix=suffix):>{w_col}}"

        lines.append(line)

    return "\n".join(lines)
