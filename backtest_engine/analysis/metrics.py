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

    cum_return: float = field(metadata={"label": "Cum. Return", "pct": True, "section": "Return Profile"})
    ann_return: float = field(metadata={"label": "Ann. Return", "pct": True, "section": "Return Profile"})  # geometric CAGR
    avg_daily_return: float = field(metadata={"label": "Avg. Daily Return", "pct": True, "section": "Return Profile"})  # arithmetic, not annualised
    skew: float = field(metadata={"label": "Ret. Skew", "section": "Return Profile"})
    kurt: float = field(metadata={"label": "Ret. Kurtosis", "section": "Return Profile"})

    max_gain: float = field(metadata={"label": "Max Gain", "pct": True, "section": "Trades"})
    max_loss: float = field(metadata={"label": "Max Loss", "pct": True, "section": "Trades"})
    best_day: Any = field(metadata={"label": "Best Day", "section": "Trades"})
    worst_day: Any = field(metadata={"label": "Worst Day", "section": "Trades"})
    daily_win_rate: float = field(metadata={"label": "Daily Win Rate", "pct": True, "section": "Trades"})

    max_drawdown: float = field(metadata={"label": "Max Drawdown", "pct": True, "section": "Drawdowns"})
    max_dd_days: int = field(metadata={"label": "Max DD Days", "suffix": " Days", "section": "Drawdowns"})
    max_dd_recovery_days: int = field(metadata={"label": "Max DD Recovery Days", "suffix": " Days", "section": "Drawdowns"})

    ann_vol: float = field(metadata={"label": "Ann. Volatility", "pct": True, "section": "Risk"})
    var_95pct: float = field(metadata={"label": "95% VaR", "pct": True, "section": "Risk"})
    cvar: float = field(metadata={"label": "Expected Shortfall", "pct": True, "section": "Risk"})

    sharpe_ratio: float = field(metadata={"label": "Sharpe Ratio", "section": "Ratios"})
    sortino_ratio: float = field(metadata={"label": "Sortino Ratio", "section": "Ratios"})
    calmar_ratio: float = field(metadata={"label": "Calmar Ratio", "section": "Ratios"})

@dataclass
class TradeMetrics:
    """
    Discrete-trade statistics.
    """

    win_rate: float = field(metadata={"label": "Win Rate", "pct": True})
    trades_per_day: float = field(metadata={"label": "Average Trades / Day"})
    return_per_trade: float = field(metadata={"label": "Average Return / Trade", "pct": True})
    total_trades: int = field(metadata={"label": "Total Trades"})

@dataclass
class RelativeMetrics:
    """
    Comparison statistics for a return series against a reference series.
    """

    alpha: float = field(metadata={"label": "Alpha"})
    beta: float = field(metadata={"label": "Beta"})
    r_squared: float = field(metadata={"label": "R-Squared"})
    information_ratio: float = field(metadata={"label": "Information Ratio", "section": "Ratios"})
    idiosyncratic_risk: float = field(metadata={"label": "Idiosyncratic Risk", "pct": True})  # residual vol after removing beta*reference + alpha/252

    correlation: float = field(metadata={"label": "Correlation"})
    up_market_capture: float = field(metadata={"label": "Up-Market Capture"})
    down_market_capture: float = field(metadata={"label": "Down-Market Capture"})
    lower_tail_dependency: float = field(metadata={"label": "Lower Tail Dependency", "section": "Risk"})
    upper_tail_dependency: float = field(metadata={"label": "Upper Tail Dependency", "section": "Risk"})

@dataclass
class DailySnapshot:
    """Single-day cumulative-return snapshot."""

    strat_cum_return: float
    bench_cum_return: float

    def __str__(self) -> str:
        return f"Strategy Return: {format_value(self.strat_cum_return, pct=True)}\nBenchmark Return: {format_value(self.bench_cum_return, pct=True)}"

@dataclass
class ConnectorExtras:
    """
    StrategyConnector fields not covered by SeriesMetrics/RelativeMetrics.
    Fields are keyed by column name.
    """

    crash_correlation_to_book: dict[str, float] = field(metadata={"label": "Crash Correlation to Book"})
    intraday_correlation_to_book: float = field(metadata={"label": "Intraday Correlation to Book"})
    incremental_sharpe_marginal: float = field(metadata={"label": "Incremental Sharpe (Marginal)"})
    incremental_sharpe_realised: dict[str, float] = field(metadata={"label": "Incremental Sharpe (Realised)"})
    cvar_monte_carlo: dict[str, float] = field(metadata={"label": "95% cVar (Monte Carlo)", "pct": True})
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
    ann_return = (1 + cum_return) ** (252 / days) - 1  # geometric
    avg_daily_return = float(returns.mean())
    ann_vol = float(returns.std()) * np.sqrt(252)
    max_dd = float(drawdown.min())

    underwater = drawdown < 0
    state_changes = (underwater != underwater.shift()).cumsum()
    max_dd_days = int(drawdown.where(underwater).groupby(state_changes).count().max())

    # recovery: bars from the trough until drawdown first returns to ~0 (else: end of series)
    dd_arr = drawdown.to_numpy()
    trough_idx = int(np.argmin(dd_arr))
    rows = np.arange(len(dd_arr))
    recovered = np.where((rows >= trough_idx) & np.isclose(dd_arr, 0), rows, np.inf)
    recovery_idx = recovered.min()
    recovery_idx = len(dd_arr) if np.isinf(recovery_idx) else recovery_idx
    max_dd_recovery_days = int(recovery_idx - trough_idx)

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
        max_dd_recovery_days=max_dd_recovery_days,
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

    cov_val = returns.cov(reference)
    ret_mean, ref_mean = returns.mean(), reference.mean()

    beta = _safe_div(cov_val, reference.var())
    alpha = float((ret_mean - beta * ref_mean) * 252)
    correlation = _safe_div(cov_val, returns.std() * reference.std())  # avoids pandas .corr()'s unguarded internal np.corrcoef division
    r_squared = float(correlation ** 2)

    tracking_error = (returns - reference).std() * np.sqrt(252)
    information_ratio = _safe_div((ret_mean - ref_mean) * 252, tracking_error)

    residual = returns - (reference * beta + alpha / 252)
    idiosyncratic_risk = float(residual.std()) * np.sqrt(252)

    up_mask = reference > 0
    down_mask = reference < 0
    up_market_capture = _safe_div(returns[up_mask].mean(), reference[up_mask].mean())
    down_market_capture = _safe_div(returns[down_mask].mean(), reference[down_mask].mean())

    ref_lower_q, ref_upper_q = reference.quantile([0.10, 0.90])
    ret_lower_q, ret_upper_q = returns.quantile([0.10, 0.90])
    ref_lower_mask = reference <= ref_lower_q
    ref_upper_mask = reference >= ref_upper_q
    lower_tail_dependency = _safe_div(float(((returns <= ret_lower_q) & ref_lower_mask).sum()), float(ref_lower_mask.sum()))
    upper_tail_dependency = _safe_div(float(((returns >= ret_upper_q) & ref_upper_mask).sum()), float(ref_upper_mask.sum()))

    return RelativeMetrics(
        alpha=alpha,
        beta=float(beta),
        r_squared=r_squared,
        information_ratio=information_ratio,
        idiosyncratic_risk=idiosyncratic_risk,
        correlation=float(correlation),
        up_market_capture=up_market_capture,
        down_market_capture=down_market_capture,
        lower_tail_dependency=lower_tail_dependency,
        upper_tail_dependency=upper_tail_dependency,
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
        return f"{value:,.2f}{suffix}"

    return f"{value}{suffix}"

def dataclass_rows(instances: list[Optional[Any]], reference_cls: type, *, default_section: Optional[str] = None) -> list[tuple[Optional[str], list[tuple[str, list[str]]]]]:
    """
    Turn dataclass fields into (section_title, rows) groups for render_sections.

    Fields carrying "section" metadata are grouped by declaration order.
    Fields without one fall back to default_section.

    instances : list[Optional[Any]]
        Dataclass instances aligned 1:1 to the target table's columns (None = "-" for that column).
    reference_cls : type
        The dataclass type instances share (drives field order/labels/formatting).
    default_section : Optional[str] = None
        Section title used for fields with no "section" metadata of their own.

    Returns list[tuple[Optional[str], list[tuple[str, list[str]]]]]
        (section_title, [(label, formatted_values), ...]) groups.
    """

    groups: list[tuple[Optional[str], list]] = []
    current: Any = object()  # sentinel, guaranteed to differ from any real section value

    for f in dataclasses.fields(reference_cls):
        section = f.metadata.get("section", default_section)

        if section != current:
            groups.append((section, []))
            current = section

        label = f.metadata.get("label", f.name)
        pct = f.metadata.get("pct", False)
        suffix = f.metadata.get("suffix", "")
        values = [format_value(getattr(inst, f.name), pct=pct, suffix=suffix) if inst is not None else "-" for inst in instances]

        groups[-1][1].append((label, values))

    return groups

def merge_groups(*group_lists: list[tuple[Optional[str], list[tuple[str, list[str]]]]]) -> list[tuple[Optional[str], list[tuple[str, list[str]]]]]:
    """
    Merge groups from multiple dataclass_rows() calls, combining sections into blocks.

    group_lists : list[tuple[Optional[str], list[tuple[str, list[str]]]]]
        Any number of dataclass_rows() outputs, in the order they should be merged.

    Returns list[tuple[Optional[str], list[tuple[str, list[str]]]]]
        Combined (section_title, rows) groups, ready for render_sections.
    """

    order: list[Optional[str]] = []
    bucket: dict[Optional[str], list] = {}

    for groups in group_lists:
        for title, rows in groups:
            if title not in bucket:
                order.append(title)
                bucket[title] = []

            bucket[title].extend(rows)

    return [(title, bucket[title]) for title in order]

def render_sections(headers: list[str], groups: list[tuple[Optional[str], list[tuple[str, list[str]]]]], *, show_header: bool = True) -> str:
    """
    Render metric groups as a single pandas-styled table with section titles inserted as divider rows.

    headers : list[str]
        Column header labels.
    groups : list[tuple[Optional[str], list[tuple[str, list[str]]]]]
        (section_title, rows) groups, as produced by dataclass_rows.
    show_header : bool = True
        Whether to print the column header row.

    Returns str
        Formatted table, in the style of pd.DataFrame.to_string().
    """

    index = []
    data = {h: [] for h in headers}

    for i, (title, rows) in enumerate(groups):
        if i > 0:
            index.append("")
            for h in headers:
                data[h].append("")

        if title:
            index.append("> " + title)
            for h in headers:
                data[h].append("")

        for label, values in rows:
            index.append(label)
            for h, v in zip(headers, values):
                data[h].append(v)

    return pd.DataFrame(data, index=index).to_string(header=show_header)
