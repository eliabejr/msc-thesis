"""
backtest/engine.py
==================
Daily backtesting engine that applies portfolio strategies over the
out-of-sample testing period (2007-2023).

Responsibilities:
  • Iterate day by day (or rebalance-date by rebalance-date)
  • Feed each strategy with the correct information set (no look-ahead)
  • Accumulate returns, weights, turnover
  • Compute and return performance metrics

Single Responsibility : simulation loop + accounting only.
Dependency Inversion  : receives strategy objects, does not instantiate them.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_metrics
from src.config.settings import ASSETS, TRANSACTION_COST, TRADING_DAYS_YEAR
from src.portfolio.strategies import PortfolioStrategy

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Event-driven daily backtester.

    Parameters
    ----------
    excess_returns : DataFrame  (dates × assets)
    total_returns  : DataFrame  (dates × assets)  — used for P&L
    rf             : Series  daily risk-free rate
    assets         : ordered list of asset names
    transaction_cost : one-way cost fraction (default 5 bps)
    """

    def __init__(
        self,
        excess_returns:   pd.DataFrame,
        total_returns:    pd.DataFrame,
        rf:               pd.Series,
        assets:           List[str] = ASSETS,
        transaction_cost: float     = TRANSACTION_COST,
    ) -> None:
        self.excess_returns   = excess_returns
        self.total_returns    = total_returns
        self.rf               = rf
        self.assets           = assets
        self.N                = len(assets)
        self.tc               = transaction_cost

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        strategy:     PortfolioStrategy,
        test_start:   str,
        test_end:     str,
        regime_forecasts:  Optional[pd.DataFrame] = None,
        regime_cond_rets:  Optional[pd.DataFrame] = None,
    ) -> Dict[str, pd.Series | pd.DataFrame | dict]:
        """
        Run a single strategy over the testing period.

        Parameters
        ----------
        strategy          : a PortfolioStrategy instance
        test_start/end    : ISO date strings
        regime_forecasts  : DataFrame (dates × assets), 0=bull/1=bear
                            (aligned to testing period)
        regime_cond_rets  : DataFrame (dates × assets), regime-conditional
                            daily return forecasts for MV(JM-XGB)

        Returns
        -------
        dict with keys:
          'port_returns'  : pd.Series – daily portfolio total returns
          'weights'       : pd.DataFrame – EOD weights
          'turnover'      : pd.Series – daily one-way turnover
          'metrics'       : dict of annualised performance metrics
        """
        idx   = self.excess_returns.index
        dates = idx[(idx >= test_start) & (idx <= test_end)]

        port_returns: List[float]      = []
        weights_hist: List[np.ndarray] = []
        turnover_hist: List[float]     = []
        date_index:   List[pd.Timestamp] = []

        w_prev = np.ones(self.N) / self.N   # initial equal-weight

        for i, date in enumerate(dates[:-1]):
            # Information available at end of day `date`
            hist_ret  = self.excess_returns.loc[:date]
            hist_tot  = self.total_returns.loc[:date]
            rf_scalar = float(self.rf.get(date, 0.0))

            # Regime forecast for the *next* day
            regime_fc     = self._get_regime_series(regime_forecasts, date)
            regime_cond_r = self._get_return_forecast(regime_cond_rets, date)

            # Compute target weights
            try:
                w_target = strategy.weights(
                    date         = date,
                    returns      = hist_ret,
                    rf           = self.rf,
                    regime_fc    = regime_fc,
                    opt_lam_ret  = regime_cond_r,
                    w_prev       = w_prev,
                )
            except Exception as exc:
                logger.warning("Strategy failed on %s: %s – using previous weights.", date, exc)
                w_target = w_prev.copy()

            w_target = np.clip(w_target, 0, 1)

            # One-way turnover
            to = float(np.abs(w_target - w_prev).sum())
            turnover_hist.append(to)

            # P&L on the *next* day
            next_date = dates[i + 1]
            r_next = self.total_returns.loc[next_date, self.assets].values
            r_next = np.nan_to_num(r_next, nan=0.0)

            # Risk-free allocation
            leverage       = float(w_target.sum())
            rf_weight      = max(1.0 - leverage, 0.0)
            rf_next        = float(self.rf.get(next_date, rf_scalar))

            port_ret       = float(w_target @ r_next) + rf_weight * rf_next
            port_ret      -= to * self.tc        # deduct trading cost

            port_returns.append(port_ret)
            weights_hist.append(w_target.copy())
            date_index.append(next_date)

            # Update previous weights  (mark-to-market drift)
            w_mkt   = w_target * (1 + r_next)
            total_v = float(w_mkt.sum()) + rf_weight * (1 + rf_next)
            if total_v > 0:
                w_prev = w_mkt / total_v
            else:
                w_prev = np.zeros(self.N)

        port_s   = pd.Series(port_returns, index=pd.DatetimeIndex(date_index), name="port_return")
        weights_df = pd.DataFrame(
            np.vstack(weights_hist),
            index=pd.DatetimeIndex(date_index),
            columns=self.assets,
        )
        turnover_s = pd.Series(
            turnover_hist,
            index=pd.DatetimeIndex(date_index),
            name="turnover",
        )

        metrics = compute_metrics(
            port_returns = port_s,
            turnover     = turnover_s,
            weights      = weights_df,
            rf           = self.rf,
        )

        logger.info(
            "Strategy '%s': Sharpe=%.2f  MDD=%.1f%%  Turnover=%.2f",
            strategy.__class__.__name__,
            metrics["Sharpe"],
            metrics["MDD"] * 100,
            metrics["Turnover"],
        )

        return {
            "port_returns": port_s,
            "weights":      weights_df,
            "turnover":     turnover_s,
            "metrics":      metrics,
        }

    def run_all(
        self,
        strategies:        Dict[str, PortfolioStrategy],
        test_start:        str,
        test_end:          str,
        regime_forecasts:  Optional[pd.DataFrame] = None,
        regime_cond_rets:  Optional[pd.DataFrame] = None,
    ) -> Dict[str, dict]:
        """Run a dict of named strategies and collect results."""
        results: Dict[str, dict] = {}
        for name, strat in strategies.items():
            logger.info("Running strategy: %s", name)
            results[name] = self.run(
                strategy          = strat,
                test_start        = test_start,
                test_end          = test_end,
                regime_forecasts  = regime_forecasts,
                regime_cond_rets  = regime_cond_rets,
            )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_regime_series(
        regime_forecasts: Optional[pd.DataFrame],
        date: pd.Timestamp,
    ) -> Optional[pd.Series]:
        if regime_forecasts is None:
            return None
        if date not in regime_forecasts.index:
            # Forward-fill: use last available forecast
            loc = regime_forecasts.index.searchsorted(date, side="right") - 1
            if loc < 0:
                return None
            return regime_forecasts.iloc[loc]
        return regime_forecasts.loc[date]

    @staticmethod
    def _get_return_forecast(
        regime_cond_rets: Optional[pd.DataFrame],
        date: pd.Timestamp,
    ) -> Optional[Dict[str, float]]:
        if regime_cond_rets is None:
            return None
        if date not in regime_cond_rets.index:
            loc = regime_cond_rets.index.searchsorted(date, side="right") - 1
            if loc < 0:
                return None
            row = regime_cond_rets.iloc[loc]
        else:
            row = regime_cond_rets.loc[date]
        return dict(row)
