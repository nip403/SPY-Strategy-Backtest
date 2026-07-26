# SPY Strategy Backtest

Comprehensive backtesting engine designed to test the intraday volatility-breakout strategy outlined in SFI paper [Beat the Market](https://github.com/nip403/SPY-Momentum-Strategy-Backtest/blob/main/Beat%20the%20Market%20An%20Effective%20Intraday%20Momentum%20Strategy%20for%20S%26P500%20ETF%20(SPY).pdf). The `Portfolio` orchestrator is composed from pluggable Strategy, Execution, and CostModel components, and provides a suite of analysis tools for reporting & evaluation.

## Quickstart, Features, and Example Usage

- [main_new.ipynb](main_new.ipynb)

Variants are chosen by passing component instances to `Portfolio` (`strategy=`, `execution=`, `cost_model=`) rather than subclassing — combinations that used to require a multiple-inheritance diamond (e.g. `PortfolioDynamicCost` + `PortfolioCappedVolumeRollover`) are now just two kwargs.

## Architecture

```
backtest_engine/
├── __init__.py
├── core.py                          Portfolio (orchestrator; composes a Strategy, optional Execution, and CostModel)
├── data.py                          Alpaca API data fetch, caching, preprocessing
├── utils.py                         Toy equity series generator, other helpers
├── components/
│   ├── base.py                      Shared BacktestContext + Strategy/Execution/CostModel protocols
│   ├── strategy.py                  Pluggable signal-generation strategies
│   ├── execution.py                 Pluggable fill/capacity-constraint models
│   └── cost_model.py                Pluggable cost & market-impact models
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
- ~~consider complete refactor of backtest pipeline to favour composition over inheritance~~ done (see `components/`); note `returns_matrix`/`sharpe_curve` still hold *positions* fixed at the construction AUM when composed with an AUM-dependent Execution model (e.g. `CappedVolumeExecution`) and only vectorise the CostModel's cost/impact scaling across the AUM grid — truly re-deriving positions per AUM point is unsolved and would need a different vectorisation strategy