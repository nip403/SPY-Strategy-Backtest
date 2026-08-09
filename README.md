# SPY Strategy Backtest

Comprehensive backtesting engine designed to test the intraday volatility-breakout strategy outlined in SFI paper [Beat the Market](https://github.com/nip403/SPY-Momentum-Strategy-Backtest/blob/main/Beat%20the%20Market%20An%20Effective%20Intraday%20Momentum%20Strategy%20for%20S%26P500%20ETF%20(SPY).pdf). The `Portfolio` core class is composed from pluggable Strategy, Execution, and CostModel components, and provides a suite of analysis tools for reporting & evaluation.

## Quickstart, Features, and Example Usage

- [main.ipynb](main.ipynb)

## Architecture

```
backtest_engine/
├── __init__.py
├── core.py                          Portfolio core class
├── data.py                          Alpaca API data fetch, caching, preprocessing
├── utils.py                         Toy equity series generator, other helpers
├── components/
│   ├── base.py                      Shared BacktestContext config + Strategy/Execution/CostModel base protocols
│   ├── strategy.py                  Signal generation models
│   ├── execution.py                 Fill/capacity-constraint models
│   └── cost_model.py                Cost models
└── analysis/
    ├── base.py                      Shared AnalysisReport interface
    ├── metrics.py                   Shared metric dataclasses + computation
    ├── tearsheet.py                 Tearsheet
    ├── decomposition.py             Long/short strategy decomposer
    ├── connector.py                 Book integration analysis
    └── capacity.py                  Capacity & alpha decay estimator
```

## Dependencies

See [pyproject file](pyproject.toml).

Alpaca API credentials are required for live data pulls (`data.request`); cached parquet files are used automatically when available.

## Testing

```powershell
py -3.12 -m pytest                                                   # full suite, randomised
py -3.12 -m pytest -m "fast"                                         # fast loop, skips heavier tests
py -3.12 -m pytest --cov=backtest_engine --cov-report=term-missing -ra  # coverage report + extra test summaries
```

### Todo

- end to end recommendation engine (book, bench, portfolio -> rec)
- out of sample testing
- logging, maybe
