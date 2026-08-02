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

    def expense(self, df: pd.DataFrame, ctx: BacktestContext, cache: dict) -> pd.DataFrame:
        """
        Calculate transaction costs return net returns.
        Applies commissions and slippage based on changes in portfolio exposure when positions are adjusted.

        % Cost = delta(position) * leverage / cost/share, incurred at the minute position changes.

        df : pd.DataFrame
            Backtest data containing gross returns and positions.
        ctx : BacktestContext
            Shared portfolio parameters for the backtest run.
        cache : dict
            This Portfolio's cache.

        Returns pd.Series
            Net returns after transaction costs.
        """

        cache.setdefault("close", df["close"])

        position_matrix = df[["position"]].to_numpy()
        gross_matrix = df[["gross_ret"]].to_numpy()

        df["net_ret"] = self.returns_matrix(np.array([ctx.aum]), position_matrix, gross_matrix, cache)[:, 0]

        return df

    def returns_matrix(self, aum: np.ndarray, position_matrix: np.ndarray, gross_matrix: np.ndarray, cache: dict) -> np.ndarray:
        """
        Generate returns across multiple portfolio sizes.
        Returns are always recomputed from the position_matrix (rather than assuming delta_leverage is fixed).

        aum : np.ndarray
            Portfolio capital values to evaluate.
        position_matrix : np.ndarray
            Capacity-constrained positions, one column per AUM.
        gross_matrix : np.ndarray
            Gross returns, one column per AUM.
        cache : dict
            This Portfolio's cache, as populated by expense().

        Returns np.ndarray
            Matrix of returns.
        """

        turnover = np.abs(np.diff(position_matrix, axis=0, prepend=position_matrix[:1]))
        close = cache["close"].to_numpy()[:, None]

        return gross_matrix - turnover * self.frictions / close

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

    def expense(self, df: pd.DataFrame, ctx: BacktestContext, cache: dict) -> pd.DataFrame:
        """
        Calculate returns after applying Kissel I-Star for the Portfolio's AUM.
        Precomputes and caches the AUM-/position-independent market-microstructure inputs used by I-Star.

        df : pd.DataFrame
            Backtest dataframe containing positions, prices, volume, and returns.
        ctx : BacktestContext
            Shared portfolio parameters for the backtest run.
        cache : dict
            The Portfolio's cache.

        Returns pd.DataFrame
            df with "net_ret" added: returns after market impact and commission costs.
        """

        # resample bins every day in the span, restrict to days actually present before rolling
        day_key = df.index.floor("D")
        valid_days = day_key.unique()

        adv = df["volume"].resample("D").sum().loc[valid_days].rolling(self.lookback_window).mean().shift(1)
        close = df["close"].resample("D").last().loc[valid_days]
        ann_vol = close.pct_change().rolling(self.lookback_window).std().shift(1) * np.sqrt(252)

        adv = pd.Series(day_key.map(adv), index=df.index).fillna(0)
        ann_vol = pd.Series(day_key.map(ann_vol), index=df.index).fillna(0)

        cache.setdefault("close", df["close"])
        cache.setdefault("volume", df["volume"])
        cache["dynamic_cost"] = pd.DataFrame({"adv": adv, "ann_vol": ann_vol})

        position_matrix = df[["position"]].to_numpy()
        gross_matrix = df[["gross_ret"]].to_numpy()

        df["net_ret"] = self.returns_matrix(np.array([ctx.aum]), position_matrix, gross_matrix, cache)[:, 0]

        return df

    def returns_matrix(self, aum: np.ndarray, position_matrix: np.ndarray, gross_matrix: np.ndarray, cache: dict) -> np.ndarray:
        """
        Generate net return scenarios across different portfolio sizes using the Kissel I-Star market impact model.

        Notation: 
            dL = delta-leverage = |df[position]_t - df[position]_{t-1}|
            A = aum ($)
            P = close ($/share)
            V = adv (average daily volume, trailing)
            v = bar volume
            sigma = ann_vol (annualised volatility).

        Derivation:
            1. Shares = dL x A / P
            
            2. I-Star prices cost in bps of dollar value traded:
                PermCost = a1 x (Shares / V)^a2 x sigma^a3
                TempCost = PermCost x (Shares / v)^a4
                MI (bps) = b1 x TempCost + (1 - b1) x PermCost
                
            3. MI = (MI / 10_000) x dL (as a % impact on return)
            
            4. Substitute Shares = dL x A / P to factor out AUM:
                adv_term = dL / (P x V)
                participation = dL / (P x v)
                
                Note: 
                (Shares / V)^a2 = A^a2 x (dL / (P x V))^a2 = A^a2 x adv_term^a2
                (Shares / v)^a4 = A^a4 x (dL / (P x v))^a4 = A^a4 x participation^a4

                Substituting:
                PermCost = a1 x adv_term^a2 x sigma^a3 x A^a2
                TempCost = PermCost x participation^a4 x A^a4
                         = a1 x adv_term^a2 x sigma^a3 x participation^a4 x A^(a2 + a4)
                              
            5. Factoring market impact:
                MI = [b1 x temp x A^(a2+a4) + (1 - b1) x perm x A^a2] / 10_000
                
                Let (AUM and dL factored out):
                perm_base = a1 x adv_term^a2 x sigma^a3
                temp_base = perm_base x participation^a4
                
                MI = dL x [b1 x temp_base x A^(a2 + a4) + (1 - b1) x perm_base x A^a2] / 10_000

            6. Handle commission (AUM-independent):
                commission % = commission_rate x Shares / A = commission_rate x dL / P

            7. net_ret = gross_ret - MI - commission

        aum : np.ndarray
            Array of AUM values used to scale market impact costs.
        position_matrix : np.ndarray
            Capacity-constrained positions, one column per AUM value.
        gross_matrix : np.ndarray
            Gross returns, one column per AUM value.
        cache : dict
            The portfolio's cache, populated by expense().

        Returns np.ndarray
            Matrix of net returns with rows matching timestamps and columns matching AUM values.
        """

        aum = np.asarray(aum, dtype=float)[None, :]

        delta_leverage = np.abs(np.diff(position_matrix, axis=0, prepend=position_matrix[:1]))

        close = cache["close"].to_numpy()[:, None]
        volume = cache["volume"].to_numpy()[:, None]
        dynamic_cost = cache["dynamic_cost"]
        adv = dynamic_cost["adv"].to_numpy()[:, None]
        ann_vol = dynamic_cost["ann_vol"].to_numpy()[:, None]

        # common factors, failsafe set div by volume=0 scenarios to 0
        denom_adv = close * adv
        adv_term = np.divide(delta_leverage, denom_adv, out=np.zeros_like(delta_leverage), where=(denom_adv != 0))

        denom_vol = close * volume
        participation = np.divide(delta_leverage, denom_vol, out=np.zeros_like(delta_leverage), where=(denom_vol != 0))

        perm = self.a1 * (adv_term ** self.a2) * (ann_vol ** self.a3) * delta_leverage
        temp = perm * (participation ** self.a4)

        commission = np.divide(self.commission * delta_leverage, close, out=np.zeros_like(delta_leverage), where=(close != 0))

        mkt = (self.b1 * temp * (aum ** (self.a2 + self.a4)) + (1 - self.b1) * perm * (aum ** self.a2)) / 10_000

        return gross_matrix - mkt - commission
