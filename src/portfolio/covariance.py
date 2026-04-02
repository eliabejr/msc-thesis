"""
portfolio/covariance.py
=======================
EWM historical covariance estimation (Sections 4.2 & 4.3).

Single Responsibility : covariance matrix estimation only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.settings import COV_HL_DAYS


def ewm_covariance(
    returns: pd.DataFrame,
    end_date: pd.Timestamp,
    halflife: int = COV_HL_DAYS,
    min_periods: int = 63,
) -> np.ndarray:
    """
    Compute the EWM historical covariance matrix using data up to `end_date`.

    Parameters
    ----------
    returns    : DataFrame of (excess) returns, columns = assets
    end_date   : last date included (strict look-back, no future data)
    halflife   : EWM halflife in trading days  (default 252)
    min_periods: minimum observations needed

    Returns
    -------
    Σ : (N, N) positive-semidefinite covariance matrix.
        Falls back to diagonal if the matrix is singular.
    """
    hist = returns.loc[:end_date]
    if len(hist) < min_periods:
        # Fallback: diagonal covariance
        vols = hist.std(ddof=1).fillna(0.01)
        return np.diag(vols.values ** 2)

    # Pandas EWM covariance
    ewm_cov = hist.ewm(halflife=halflife, min_periods=min_periods).cov().iloc[-len(hist.columns):]
    cov_mat = ewm_cov.values

    # Ensure PSD via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    eigvals = np.maximum(eigvals, 1e-8)
    cov_mat = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return cov_mat
