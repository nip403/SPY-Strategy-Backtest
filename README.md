# SPY Strategy Backtest

Comprehensive backtesting engine designed to test the intraday volatility-breakout strategy outlined in SFI paper [Beat the Market](https://github.com/nip403/SPY-Momentum-Strategy-Backtest/blob/main/Beat%20the%20Market%20An%20Effective%20Intraday%20Momentum%20Strategy%20for%20S%26P500%20ETF%20(SPY).pdf). Strategies are subclassed around an extensible `Portfolio` base class, and provides a suite of analysis tools for reporting & evaluation.

## Quickstart, Features, and Example Usage

- [main.py](main.py)

## Architecture

```
backtest_engine/
├── __init__.py
├── core.py                          Portfolio (base class)
├── data.py                          Alpaca API data fetch, caching, preprocessing
├── utils.py                         Toy equity series generator, other helpers
├── extensions/
│   ├── naive_rolling_windows.py     Core Portfolio example variants
│   ├── naive_quarter_interval.py    Core Portfolio example variants
│   └── kissel_impact.py             Core Portfolio variant (Kissel I-Star impact model)
└── analysis/
    ├── tearsheet.py                 Tearsheet
    ├── decomposition.py             Long/short strategy decomposer
    └── connector.py                 Book integration analysis
```

## Dependencies

See [pyproject file](pyproject.toml).

Alpaca API credentials are required for live data pulls (`data.request`); cached parquet files are used automatically when available.

### Todo

- optimise pd -> np
- consider refactoring/consolidating tearsheet/connector/decomposer reporting fields into dataclasses