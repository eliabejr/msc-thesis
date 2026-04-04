"""
portfolio/covariance.py
=======================
EWM historical covariance estimation (Sections 4.2 & 4.3).

Single Responsibility : covariance matrix estimation only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config.settings import COV_HL_DAYS

logger = logging.getLogger(__name__)


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
    cov_mat = ewm_cov.values.astype(np.float64, copy=True)
    cov_mat = (cov_mat + cov_mat.T) * 0.5

    if not np.all(np.isfinite(cov_mat)):
        vols = hist.std(ddof=1).fillna(0.01)
        logger.debug("Non-finite EWM covariance at %s; using diagonal.", end_date)
        return np.diag(vols.values ** 2)

    n = cov_mat.shape[0]
    scale = float(np.trace(cov_mat) / max(n, 1)) or 1.0
    # Diagonal loading helps eigh converge on nearly-singular EWM covariances
    for load in (0.0, 1e-10 * scale, 1e-8 * scale, 1e-6 * scale):
        try:
            sym = cov_mat + load * np.eye(n)
            eigvals, eigvecs = np.linalg.eigh(sym)
            break
        except np.linalg.LinAlgError:
            continue
    else:
        vols = hist.std(ddof=1).fillna(0.01)
        logger.debug("eigh failed after loading at %s; using diagonal.", end_date)
        return np.diag(vols.values ** 2)

    floor = max(1e-8, 1e-12 * scale)
    eigvals = np.maximum(eigvals, floor)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T
