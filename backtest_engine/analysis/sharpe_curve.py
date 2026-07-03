from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from typing import Optional

def sharpe_curve(df: pd.DataFrame, portfolio: Portfolio, aum: Optional[list[int]] = None, base: Optional[int] = None) -> None:
    aum = aum or [i * (10 ** k) for k in range(4, 12) for i in range(1, 10)] + [1e12]
    sharpes = [portfolio(df, i).sharpe for i in aum]
    
    base = sharpes[aum.index(base) if base in aum else 0] # base: baseline/benchmark portfolio (in $ aum)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(aum, sharpes, color="blue", label="Sharpe")

    ax.axhline(base, color="gray", linestyle="--", linewidth=1, label=f"Base Sharpe ({base:.2f})")
    ax.axhline(base * 0.5, color="red", linestyle="--", linewidth=1, label=f"50% Base ({base * 0.5:.2f})")
    ax.axhline(0, color="black", linestyle="-", linewidth=1, label="Risk Free")

    ax.set_xscale("log")
    ax.set_xlim(min(aum), max(aum))

    ax.set_xlabel("AUM ($)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Strategy Capacity: Sharpe vs AUM", fontweight="bold")

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend()
    
    plt.tight_layout()
    plt.show()