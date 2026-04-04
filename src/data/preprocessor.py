"""
data/preprocessor.py
====================
Transforms raw prices + FRED series into clean daily returns and
risk-free rates aligned on the same business-day index.

Single Responsibility : compute returns, align, fill gaps.
Dependency Inversion  : accepts DataFrames, not tied to any loader.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from src.config.settings import TRADING_DAYS_YEAR

logger = logging.getLogger(__name__)


def _ffill_and_zero_leading(
    df: pd.DataFrame,
    *,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Forward-fill each column, then set values *before* the first originally-valid
    observation to ``fill_value``.

    yFinance staggered ETF inception leaves leading NaNs that ``ffill`` cannot
    remove. For excess returns, imputing **0** before first trade means “no stake
    in that asset yet” (no deviation from the risk-free leg used as benchmark).
    """
    out = df.ffill()
    for col in out.columns:
        raw = df[col]
        fv = raw.first_valid_index()
        if fv is None:
            continue
        out.loc[out.index < fv, col] = fill_value
    return out.fillna(fill_value)


class DataPreprocessor:
    """Convert raw price/yield frames into analysis-ready returns."""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def prepare(
        self,
        prices: pd.DataFrame,
        fred: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Main preparation pipeline.

        Parameters
        ----------
        prices : daily adjusted-close prices (columns = asset names)
        fred   : daily FRED data with columns  ['rf', 'y2', 'y10', 'vix', ...] (e.g. 'stlfi')

        Returns
        -------
        returns      : daily total returns (fractional, not %)
        risk_free    : daily risk-free rate (annualised ÷ 252)
        macro_raw    : aligned raw FRED series (for feature engineering)
        """
        # --- 1. Compute log returns & convert to simple returns ----------
        log_ret = np.log(prices / prices.shift(1))
        returns = np.exp(log_ret) - 1          # simple daily returns

        # --- 2. Align FRED to trading-day index --------------------------
        fred_aligned = self._align_fred(fred, returns.index)

        # --- 3. Risk-free: annualised yield → daily rate ------------------
        #   DTB3 is % per annum; divide by 100 then by 252
        rf_daily = fred_aligned["rf"] / 100.0 / TRADING_DAYS_YEAR
        rf_daily = rf_daily.ffill().fillna(0.0)

        # --- 4. Excess returns -------------------------------------------
        # returns is already aligned; rf_daily shares the same index
        excess_returns = returns.subtract(rf_daily, axis=0)

        # --- 5. Drop leading NaN rows (first row from log-diff) ----------
        first_valid = excess_returns.dropna(how="all").index[0]
        excess_returns = excess_returns.loc[first_valid:]
        returns        = returns.loc[first_valid:]
        rf_daily       = rf_daily.loc[first_valid:]
        fred_aligned   = fred_aligned.loc[first_valid:]

        # --- 6. Staggered listings: ffill short gaps; 0 excess before first obs. -
        excess_returns = _ffill_and_zero_leading(excess_returns, fill_value=0.0)

        na_count = int(excess_returns.isna().sum().sum())
        if na_count:
            logger.warning("%d NaN values remain in excess returns after cleaning.", na_count)

        logger.info(
            "Returns shape: %s  [%s → %s]",
            excess_returns.shape,
            excess_returns.index[0].date(),
            excess_returns.index[-1].date(),
        )
        return excess_returns, rf_daily, fred_aligned

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _align_fred(fred: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Reindex FRED data to match the equity trading-day calendar.

        FRED may include weekends / holidays; we forward-fill to fill gaps.
        """
        combined = fred.reindex(
            fred.index.union(target_index)
        ).ffill().reindex(target_index)
        return combined
