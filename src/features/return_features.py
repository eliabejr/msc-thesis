"""
features/return_features.py
===========================
Computes the eight asset-specific return features used in the Statistical
Jump Model (Table 2, Shu et al. 2024).

Feature set (per asset, exponentially smoothed):
  1-2.  Downside Deviation (log scale)  – halflives 5, 21
  3-5.  Average Return                  – halflives 5, 10, 21
  6-8.  Sortino Ratio                   – halflives 5, 10, 21

Single Responsibility : only feature computation, no model logic.
Open/Closed           : new features can be added via subclassing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import (
    ASSETS_NO_DD_IN_JM,
    JM_DD_HALFLIVES,
    JM_RET_HALFLIVES,
    JM_SORT_HALFLIVES,
)


class ReturnFeatureBuilder:
    """
    Build the return-feature matrix X used by the Jump Model.

    Parameters
    ----------
    exclude_dd_assets : assets for which DD features are skipped (JM only).
        Defaults to the paper's  ['AggBond', 'Treasury', 'Gold'].
    """

    def __init__(
        self,
        exclude_dd_assets: list[str] | None = None,
    ) -> None:
        self._exclude_dd = (
            exclude_dd_assets
            if exclude_dd_assets is not None
            else ASSETS_NO_DD_IN_JM
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build(
        self,
        excess_returns: pd.Series,
        asset_name: str,
        for_jm: bool = True,
    ) -> pd.DataFrame:
        """
        Compute the feature matrix for a single asset.

        Parameters
        ----------
        excess_returns : daily excess return series for one asset
        asset_name     : used to decide whether to include DD features
        for_jm         : if True, apply the DD-exclusion rule (Table 2 note);
                         if False (XGBoost), always include all 8 features.

        Returns
        -------
        DataFrame of standardised features, same index as excess_returns.
        """
        r = excess_returns.copy()
        features: dict[str, pd.Series] = {}

        # ── Downside Deviation (log scale) ────────────────────────────
        include_dd = not (for_jm and asset_name in self._exclude_dd)
        if include_dd:
            for hl in JM_DD_HALFLIVES:
                dd = self._ewm_downside_dev(r, hl)
                log_dd = np.log(dd.clip(lower=1e-8))   # log-transform
                features[f"log_dd_hl{hl}"] = log_dd

        # ── Average Return ────────────────────────────────────────────
        for hl in JM_RET_HALFLIVES:
            features[f"avg_ret_hl{hl}"] = r.ewm(halflife=hl, min_periods=1).mean()

        # ── Sortino Ratio ─────────────────────────────────────────────
        for hl in JM_SORT_HALFLIVES:
            avg = r.ewm(halflife=hl, min_periods=1).mean()
            dd  = self._ewm_downside_dev(r, hl)
            sortino = avg / dd.clip(lower=1e-8)
            features[f"sortino_hl{hl}"] = sortino

        df = pd.DataFrame(features, index=r.index)
        return self._standardise(df)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ewm_downside_dev(returns: pd.Series, halflife: int) -> pd.Series:
        """Exponentially weighted downside deviation (semi-deviation)."""
        # Only use the negative portion; zeros contribute nothing to downside
        downside = returns.clip(upper=0.0)
        # EWM mean of squared downside deviations → take sqrt
        ewm_var = (downside ** 2).ewm(halflife=halflife, min_periods=1).mean()
        return np.sqrt(ewm_var)

    @staticmethod
    def _standardise(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise features to zero mean / unit variance using the
        *expanding* (cumulative) statistics to avoid look-ahead bias.
        The JM paper standardises features before fitting.
        """
        mu  = df.expanding(min_periods=20).mean()
        std = df.expanding(min_periods=20).std().clip(lower=1e-8)
        return (df - mu) / std
