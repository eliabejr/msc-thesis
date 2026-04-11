"""
portfolio/strategies.py
=======================
Implements the six portfolio strategies described in Section 4:

  MinVar         – original minimum-variance (constant μ ∝ 1)
  MinVar(JM-XGB) – regime-enhanced minimum-variance
  MV             – naive mean-variance (EWM historical μ)
  MV(JM-XGB)     – regime-enhanced mean-variance
  EW             – equally-weighted (1/N)
  EW(JM-XGB)     – regime-filtered equally-weighted
  60/40          – fixed-mix benchmark

All strategies implement the same interface:
    weights(date, context) → np.ndarray

Open/Closed  : new strategies subclass PortfolioStrategy.
Liskov        : any subclass is a drop-in replacement.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config.settings import (
    ASSETS,
    BENCHMARK_60_40,
    GAMMA_RISK_MV,
    GAMMA_RISK_MV_JMXGB,
    GAMMA_RISK_MINVAR,
    GAMMA_TRADE_MINVAR,
    GAMMA_TRADE_MINVAR_JMXGB,
    GAMMA_TRADE_MV,
    GAMMA_TRADE_MV_JMXGB,
    MIN_BULLISH_ASSETS,
    MINVAR_BEAR_MU,
    MINVAR_BULL_MU,
    MV_BASELINE_HL_DAYS,
    MV_BEAR_CAP,
    WEIGHT_UB,
)
from src.portfolio.covariance import ewm_covariance
from src.portfolio.optimizer import MVOptimizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class PortfolioStrategy(ABC):
    """Abstract base for all portfolio strategies."""

    def __init__(self, assets: list[str]) -> None:
        self.assets = assets
        self.N      = len(assets)

    @abstractmethod
    def weights(
        self,
        date:         pd.Timestamp,
        returns:      pd.DataFrame,
        rf:           pd.Series,
        regime_fc:    Optional[pd.Series] = None,
        opt_lam_ret:  Optional[Dict[str, float]] = None,
        jm_labels:    Optional[Dict[str, np.ndarray]] = None,
        jm_dates:     Optional[Dict[str, pd.DatetimeIndex]] = None,
        w_prev:       Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return portfolio weights (length N) for the *next* period.

        Parameters
        ----------
        date        : current date (t); weights are applied on t+1.
        returns     : all available returns up to and including `date`.
        rf          : risk-free rate series.
        regime_fc   : today's regime forecast per asset (0=bull, 1=bear).
        opt_lam_ret : per-asset regime-conditional return forecast (MV).
        jm_labels   : per-asset array of in-sample JM labels (for MV return).
        jm_dates    : per-asset DatetimeIndex matching jm_labels.
        w_prev      : previous weights (for trading cost).
        """


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class FixMix6040(PortfolioStrategy):
    """Fixed 60/40 benchmark (Table 5)."""

    def __init__(self, assets: list[str]) -> None:
        super().__init__(assets)
        self._w = np.array([BENCHMARK_60_40.get(a, 0.0) for a in assets])

    def weights(self, date, returns, rf, **kwargs) -> np.ndarray:
        return self._w.copy()


class MinVarPortfolio(PortfolioStrategy):
    """
    Original Minimum-Variance portfolio (Section 4.2).

    μ ∝ 1  (constant), no trading-cost term.
    γ^risk = 10, γ^trade = 0.
    """

    def __init__(self, assets: list[str]) -> None:
        super().__init__(assets)
        self._opt = MVOptimizer(
            gamma_risk  = GAMMA_RISK_MINVAR,
            gamma_trade = GAMMA_TRADE_MINVAR,
        )

    def weights(self, date, returns, rf, w_prev=None, **kwargs) -> np.ndarray:
        Sigma = ewm_covariance(returns, end_date=date)
        mu    = np.ones(self.N) * 1e-3   # μ ∝ 1
        return self._opt.solve(mu, Sigma, w_prev)


class MinVarJMXGB(PortfolioStrategy):
    """
    Regime-enhanced Minimum-Variance portfolio (Section 4.2).

    μ_j = 10 bps if bullish, 0 if bearish.
    γ^risk = 10, γ^trade = 1.
    Falls back to 100% risk-free if ≤ 3 assets bullish.
    """

    def __init__(self, assets: list[str]) -> None:
        super().__init__(assets)
        self._opt = MVOptimizer(
            gamma_risk  = GAMMA_RISK_MINVAR,
            gamma_trade = GAMMA_TRADE_MINVAR_JMXGB,
        )

    def weights(self, date, returns, rf, regime_fc=None, w_prev=None, **kwargs) -> np.ndarray:
        if regime_fc is None:
            return MinVarPortfolio(self.assets).weights(date, returns, rf, w_prev=w_prev)

        # NumPy only: a pandas Series μ makes `μ @ w` use pd.Series.__matmul__ and breaks cvxpy.
        if isinstance(regime_fc, pd.Series):
            bull = (regime_fc.reindex(self.assets).fillna(1) == 0).to_numpy(dtype=np.float64)
        else:
            bull = np.asarray((regime_fc == 0).astype(float), dtype=np.float64).reshape(-1)
        n_bull = int(bull.sum())

        if n_bull < MIN_BULLISH_ASSETS:
            return np.zeros(self.N)   # 100% risk-free

        mu    = bull * MINVAR_BULL_MU + (1 - bull) * MINVAR_BEAR_MU
        Sigma = ewm_covariance(returns, end_date=date)
        return self._opt.solve(mu, Sigma, w_prev)


class MVPortfolio(PortfolioStrategy):
    """
    Original Mean-Variance portfolio with naive EWM return forecast (Section 4.3).

    γ^risk = 5, γ^trade = 0.
    """

    def __init__(self, assets: list[str]) -> None:
        super().__init__(assets)
        self._opt = MVOptimizer(
            gamma_risk  = GAMMA_RISK_MV,
            gamma_trade = GAMMA_TRADE_MV,
        )

    def weights(self, date, returns, rf, w_prev=None, **kwargs) -> np.ndarray:
        # EWM mean with 5-year halflife
        hist = returns.loc[:date]
        mu   = hist.ewm(halflife=MV_BASELINE_HL_DAYS, min_periods=252).mean().iloc[-1].values
        mu   = np.nan_to_num(mu, nan=0.0)
        Sigma = ewm_covariance(returns, end_date=date)
        return self._opt.solve(mu, Sigma, w_prev)


class MVJMXGBPortfolio(PortfolioStrategy):
    """
    Regime-enhanced Mean-Variance portfolio (Section 4.3).

    Return forecast = average return of in-sample periods in the same regime
    as the forecast, using the optimal λ JM.

    γ^risk = 10, γ^trade = 1.
    """

    def __init__(self, assets: list[str]) -> None:
        super().__init__(assets)
        self._opt = MVOptimizer(
            gamma_risk  = GAMMA_RISK_MV_JMXGB,
            gamma_trade = GAMMA_TRADE_MV_JMXGB,
        )

    def weights(
        self,
        date,
        returns,
        rf,
        regime_fc     = None,
        opt_lam_ret   = None,
        w_prev        = None,
        **kwargs,
    ) -> np.ndarray:
        if regime_fc is None or opt_lam_ret is None:
            return MVPortfolio(self.assets).weights(date, returns, rf, w_prev=w_prev)

        n_bull = int((regime_fc == 0).sum())
        if n_bull < MIN_BULLISH_ASSETS:
            return np.zeros(self.N)

        # Regime-conditional return forecast (supplied externally)
        mu = np.array([opt_lam_ret.get(a, 0.0) for a in self.assets])
        # Cap bearish forecasts at MV_BEAR_CAP
        for i, a in enumerate(self.assets):
            if regime_fc.get(a, 1) == 1:
                mu[i] = max(mu[i], MV_BEAR_CAP)

        Sigma = ewm_covariance(returns, end_date=date)
        return self._opt.solve(mu, Sigma, w_prev)


class EWPortfolio(PortfolioStrategy):
    """
    Equally-weighted portfolio: 1/N to each risky asset (Section 4.4).
    """

    def weights(self, date, returns, rf, **kwargs) -> np.ndarray:
        return np.ones(self.N) / self.N


class EWJMXGBPortfolio(PortfolioStrategy):
    """
    Regime-filtered equally-weighted portfolio (Section 4.4).

    Distributes 100% weight equally among bullish assets.
    Falls back to 100% risk-free if ≤ 3 assets are bullish.
    """

    def weights(self, date, returns, rf, regime_fc=None, **kwargs) -> np.ndarray:
        if regime_fc is None:
            return np.ones(self.N) / self.N

        bull_mask = (regime_fc == 0).values.astype(float)
        n_bull    = int(bull_mask.sum())

        if n_bull < MIN_BULLISH_ASSETS:
            return np.zeros(self.N)

        return bull_mask / n_bull


# ---------------------------------------------------------------------------
# Strategy registry  (Open/Closed – new strategies just add an entry)
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: Dict[str, type] = {
    "60/40":           FixMix6040,
    "MinVar":          MinVarPortfolio,
    "MinVar(JM-XGB)":  MinVarJMXGB,
    "MV":              MVPortfolio,
    "MV(JM-XGB)":      MVJMXGBPortfolio,
    "EW":              EWPortfolio,
    "EW(JM-XGB)":      EWJMXGBPortfolio,
}


def build_strategy(name: str, assets: list[str]) -> PortfolioStrategy:
    """Factory function – instantiate a strategy by name."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY)}")
    return cls(assets)
