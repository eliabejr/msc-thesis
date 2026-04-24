"""
models/xgb_classifier.py
========================
Wraps XGBClassifier for regime forecasting (Section 3.3, Shu et al. 2024).

Responsibilities:
  • Train an XGBoost classifier on  {(x̃_t,  ŝ_{t+1})}
  • Predict daily regime probabilities for future periods
  • Apply EWM smoothing to the probability series

Single Responsibility : classifier training + probability smoothing.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier  # type: ignore

from src.config.settings import PROB_SMOOTH_HL, XGB_PARAMS, XGB_THRESHOLD

logger = logging.getLogger(__name__)


class RegimeForecaster:
    """
    XGBoost-based regime forecaster with optional EWM probability smoothing.

    Parameters
    ----------
    asset_name     : used to look up the per-asset smoothing halflife
    smooth_hl      : override the default per-asset halflife (0 = no smoothing)
    xgb_params     : XGBoost hyperparameters (defaults to paper's settings)
    threshold      : probability threshold for predicting the bearish class
    """

    def __init__(
        self,
        asset_name: str,
        smooth_hl:  Optional[int] = None,
        xgb_params: Optional[dict] = None,
        threshold:  float = XGB_THRESHOLD,
    ) -> None:
        self.asset_name = asset_name
        self.smooth_hl  = smooth_hl if smooth_hl is not None else PROB_SMOOTH_HL.get(asset_name, 0)
        self.threshold  = threshold
        self._constant_class: Optional[int] = None

        params = dict(xgb_params or XGB_PARAMS)
        seed = int(params.pop("random_state", 42))
        self._clf = XGBClassifier(
            **params,
            seed=seed,
        )

        self._fitted = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> "RegimeForecaster":
        """
        Fit the classifier.

        Parameters
        ----------
        X_train : feature matrix (return + macro features)
        y_train : regime labels shifted +1  (ŝ_{t+1})
                  0 = bullish, 1 = bearish
        """
        # Drop rows where label is NaN (last row after shift)
        mask = y_train.notna()
        X    = X_train.loc[mask].values
        y    = y_train.loc[mask].values.astype(int)

        # Handle degenerate case (only one class in training)
        if len(np.unique(y)) < 2:
            self._constant_class = int(y[0])
            logger.warning(
                "[%s] Only one class in training labels. "
                "Using constant %s forecast.",
                self.asset_name,
                "bearish" if self._constant_class == 1 else "bullish",
            )
            self._fitted = True
            return self

        self._constant_class = None
        self._clf.fit(X, y)
        self._fitted = True
        return self

    def predict_proba_series(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """
        Return EWM-smoothed probability of the *bearish* regime  (class 1).

        Parameters
        ----------
        X : feature DataFrame; index must be a DatetimeIndex.

        Returns
        -------
        pd.Series with the same index as X.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba_series().")

        if self._constant_class is not None:
            raw_proba = np.full(len(X), float(self._constant_class), dtype=float)
        else:
            raw_proba = self._clf.predict_proba(X.values)[:, 1]   # P(bearish)
        proba_s   = pd.Series(raw_proba, index=X.index, name=f"p_bear_{self.asset_name}")

        if self.smooth_hl and self.smooth_hl > 0:
            proba_s = proba_s.ewm(halflife=self.smooth_hl, min_periods=1).mean()

        return proba_s

    def predict_regime(self, X: pd.DataFrame) -> pd.Series:
        """
        Binary regime forecast: 0=bullish, 1=bearish.
        """
        proba  = self.predict_proba_series(X)
        regime = (proba >= self.threshold).astype(int)
        regime.name = f"regime_{self.asset_name}"
        return regime
