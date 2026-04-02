"""
features/macro_features.py
==========================
Computes the five cross-asset macro-features used in the XGBoost classifier
(Table 3, Shu et al. 2024).

Feature                       Transformation
──────────────────────────────────────────────────────────────────────────
US Treasury 2-Year Yield      EWMA of first difference          (hl=21)
Yield Curve Slope (10Y-2Y)    EWMA of level                     (hl=10)
Yield Curve Slope (10Y-2Y)    EWMA of first difference          (hl=21)
VIX Index                     EWMA of log-difference            (hl=63)
Stock-Bond Correlation        Rolling correlation               (252-day)
──────────────────────────────────────────────────────────────────────────

Single Responsibility : only feature computation for the macro block.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import (
    CORR_BOND_ASSET,
    CORR_STOCK_ASSET,
    MACRO_CORR_WINDOW,
    MACRO_SLOPE_D_HL,
    MACRO_SLOPE_HL,
    MACRO_VIX_HL,
    MACRO_YIELD_HL,
)


class MacroFeatureBuilder:
    """
    Build the five macro features from aligned FRED data + asset returns.

    Parameters
    ----------
    stock_asset : name of the equity asset for stock-bond correlation
    bond_asset  : name of the bond asset for stock-bond correlation
    """

    def __init__(
        self,
        stock_asset: str = CORR_STOCK_ASSET,
        bond_asset: str  = CORR_BOND_ASSET,
    ) -> None:
        self._stock = stock_asset
        self._bond  = bond_asset

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(
        self,
        fred_aligned: pd.DataFrame,
        excess_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute the five macro features on the trading-day index.

        Parameters
        ----------
        fred_aligned   : DataFrame with columns ['rf', 'y2', 'y10', 'vix']
                         aligned to the equity trading calendar
        excess_returns : DataFrame with all asset excess returns
                         (needed for the stock-bond rolling correlation)

        Returns
        -------
        DataFrame with 5 columns, same index as excess_returns.
        """
        idx = excess_returns.index

        y2    = fred_aligned["y2"].reindex(idx).ffill()
        y10   = fred_aligned["y10"].reindex(idx).ffill()
        vix   = fred_aligned["vix"].reindex(idx).ffill()

        features: dict[str, pd.Series] = {}

        # ── 1. EWMA of 2Y yield first difference ─────────────────────
        dy2 = y2.diff()
        features["y2_diff_ewma"] = dy2.ewm(halflife=MACRO_YIELD_HL, min_periods=1).mean()

        # ── 2. EWMA of yield curve slope (10Y − 2Y) ──────────────────
        slope = y10 - y2
        features["slope_ewma"] = slope.ewm(halflife=MACRO_SLOPE_HL, min_periods=1).mean()

        # ── 3. EWMA of slope first difference ────────────────────────
        dslope = slope.diff()
        features["slope_diff_ewma"] = dslope.ewm(halflife=MACRO_SLOPE_D_HL, min_periods=1).mean()

        # ── 4. EWMA of VIX log-differences ───────────────────────────
        log_vix = np.log(vix.clip(lower=1e-8))
        dlog_vix = log_vix.diff()
        features["vix_logdiff_ewma"] = dlog_vix.ewm(halflife=MACRO_VIX_HL, min_periods=1).mean()

        # ── 5. Rolling stock-bond correlation (1-year window) ─────────
        stock = excess_returns[self._stock] if self._stock in excess_returns else pd.Series(np.nan, index=idx)
        bond  = excess_returns[self._bond]  if self._bond  in excess_returns else pd.Series(np.nan, index=idx)
        features["stock_bond_corr"] = (
            stock.rolling(window=MACRO_CORR_WINDOW, min_periods=MACRO_CORR_WINDOW // 2)
            .corr(bond)
        )

        df = pd.DataFrame(features, index=idx)
        return df
