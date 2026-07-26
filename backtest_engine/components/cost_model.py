from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional
from .base import BacktestContext, CostComponent

class FlatCostModel(CostComponent):
    def __init__(self, *, commission: float = 0.0035, slippage: float = 0.001) -> None:
        """
        Initialise a naive per-share cost model.

        commission : float = 0.0035
            Per-share commission cost.
        slippage : float = 0.001
            Per-share slippage cost.
        """

        self.commission = commission
        self.slippage = slippage
        self.frictions = commission + slippage
        
        self._cache = None

    def expense(self, df: pd.DataFrame, ctx: BacktestContext) -> pd.DataFrame:
        """
        Calculate transaction costs return net returns.
        Applies commissions and slippage based on changes in portfolio exposure when positions are adjusted.

        % Cost = delta(position) * leverage / cost/share, incurred at the minute position changes.

        df : pd.DataFrame
            Backtest data containing gross returns and positions.
        ctx : BacktestContext
            Shared portfolio parameters for the backtest run.

        Returns pd.Series
            Net returns after transaction costs.
        """

        self._cache = df["gross_ret"] - (df["position"].diff().abs().fillna(0) * self.frictions / df["close"])
        df["net_ret"] = self._cache

        return df

    def returns_matrix(self, aum: np.ndarray) -> np.ndarray:
        """
        Generate returns across multiple portfolio sizes.
        Costs are AUM-independent, so this trivially tiles the cached net returns.

        aum : np.ndarray
            Portfolio capital values to evaluate.

        Returns np.ndarray
            Matrix of returns with one column per AUM value.
        """

        return np.tile(self._cache.to_numpy()[:, None], (1, len(aum)))

class DynamicCostModel(CostComponent):
    DEFAULT_PARAMS = {
        "a1": 700,
        "a2": 0.5,
        "a3": 1.0,
        "a4": 0.5,
        "b1": 0.85,
        "lookback": 20, # 1 trading mth
    }

    def __init__(self, *, coeff_config: Optional[dict] = None, commission: float = 0.0035) -> None:
        """
        Initialise a cost model built on the Kissel I-Star Market Impact Model.

        coeff_config : Optional[dict] = None
            Custom I* model coefficients overriding defaults. Coefficients are proprietary and not publicly available.

            a1 : float = 700
                Base market impact coefficient controlling overall cost magnitude.
                Default value is outdated by over a decade.
            a2 : float = 0.5
                Trade size exponent applied to ADV participation.
                Square-root scaling of market impact is well-documented in empirical literature.
            a3 : float = 1.0
                Volatility exponent applied to market impact scaling.
                Controls sensitivity to changes in market volatility.
            a4 : float = 0.5
                Temporary impact exponent applied to volume participation.
                Controls the additional cost from trading volume concentration.
            b1 : float = 0.85
                Market-dependent weight between temporary and permanent market impact components.
                Higher values increase sensitivity to temporary impact. Default value is by LLM consensus
            lookback : int = 20
                Number of trading days used to estimate average daily volume and volatility inputs.
        commission : float = 0.0035
            Per-share commission cost, expressed as a fraction of price. Applied in addition to market impact.
        """

        self._config = {**self.DEFAULT_PARAMS, **(coeff_config or {})}

        self.a1 = self._config["a1"]
        self.a2 = self._config["a2"]
        self.a3 = self._config["a3"]
        self.a4 = self._config["a4"]
        self.b1 = self._config["b1"]

        self.lookback_window = self._config["lookback"]
        self.commission = commission
        
        self._cache = None

    def expense(self, df: pd.DataFrame, ctx: BacktestContext) -> pd.DataFrame:
        """
        Calculate returns after applying dynamic transaction costs.
        
        Derivation calculations (factoring out AUM):
            1. Shares = (Delta Leverage x AUM) / Close

            2. Permanent Cost Component = a1 x (Shares / ADV)^a2 x Volatility^a3
                = [a1 x (Delta Leverage / (Close x ADV))^a2 x Volatility^a3] x AUM^a2
                = Baseline Permanent Cost x AUM^a2

            3. Temporary Cost Component = Permanent Cost x (Shares / Volume)^a4
                = [Baseline Permanent Cost x (Delta Leverage / (Close x Volume))^a4] x AUM^(a2 + a4)
                = Baseline Temporary Cost x AUM^(a2 + a4)

            4. Market Impact Cost = [b1 x Baseline Temporary Cost x AUM^(a2 + a4)] + [(1 - b1) x Baseline Permanent Cost x AUM^a2]

        df : pd.DataFrame
            Backtest dataframe containing positions, prices, volume, and returns.
        ctx : BacktestContext
            Shared portfolio parameters for the backtest run.

        Returns pd.Series
            Net returns after market impact and commission costs.
        """

        dates = pd.Series(df.index.date, index=df.index)
        buckets = df.groupby(df.index.date)

        # I* market impact model
        # build cache for net returns tensor, factor out aum for vectorised calcs in returns_matrix
        # isolate aum from i* and mi and cache for aum multiplication later
        delta_leverage = df["position"].diff().abs().fillna(0).to_numpy()
        close = df["close"].to_numpy()
        volume = df["volume"].to_numpy()

        adv = dates.map(buckets["volume"].sum().rolling(self.lookback_window).mean().shift(1)).fillna(0).to_numpy()
        ann_vol = dates.map(buckets["close"].last().pct_change().rolling(self.lookback_window).std().shift(1) * np.sqrt(252)).fillna(0).to_numpy()

        # common factors, failsafe set div by volume=0 scenarios to 0
        denom_adv = close * adv
        adv_term = np.divide(delta_leverage, denom_adv, out=np.zeros_like(delta_leverage), where=(denom_adv != 0))

        denom_vol = close * volume
        participation = np.divide(delta_leverage, denom_vol, out=np.zeros_like(delta_leverage), where=(denom_vol != 0))

        perm = self.a1 * (adv_term ** self.a2) * (ann_vol ** self.a3) * delta_leverage
        temp = perm * (participation ** self.a4)

        commission = np.divide(self.commission * delta_leverage, close, out=np.zeros_like(delta_leverage), where=(close != 0))

        gross = (df["position"].shift(1).fillna(0) * df["ret"].fillna(0)).to_numpy()

        self._cache = pd.DataFrame({
                "gross": gross,
                "perm": perm,
                "temp": temp,
                "commission": commission,
            },
            index=df.index,
        )
        
        df["net_ret"] = pd.Series(self.returns_matrix(np.array([ctx.aum]))[:, 0], index=df.index)

        return df

    def returns_matrix(self, aum: np.ndarray) -> np.ndarray:
        """
        Generate net return scenarios across different portfolio sizes based on precomputed model params.

        aum : np.ndarray
            Array of AUM values used to scale market impact costs.

        Returns np.ndarray
            Matrix of net returns with rows matching timestamps and columns matching AUM values.
        """

        gross = self._cache["gross"].to_numpy()[:, None]
        perm = self._cache["perm"].to_numpy()[:, None]
        temp = self._cache["temp"].to_numpy()[:, None]
        commission = self._cache["commission"].to_numpy()[:, None]

        aum = aum[None, :]
        mkt = (self.b1 * temp * (aum ** (self.a2 + self.a4)) + (1 - self.b1) * perm * (aum ** self.a2)) / 10_000

        # handle division failures upstream - potentially a bad choice
        return np.nan_to_num(gross - mkt - commission, nan=0, posinf=0, neginf=0)
