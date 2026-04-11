"""
Formal time-series utilities: HP decomposition, HAC regression, bootstrap Spearman,
and aligning macro series to JM training windows (same rule as Notebook 03).
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.filters.hp_filter import hpfilter

# Recommended λ for **daily** financial series: between monthly (129_600) and
# quarterly (1_600) scaling; see Ravn–Uhlig (2002) on adjusting λ with observation frequency.
# This is a pragmatic default for VIX / stress (not a claim of optimality).
HP_LAMBDA_DAILY_DEFAULT: float = 1_600_000.0


def hp_decompose(
    series: pd.Series,
    lam: float = HP_LAMBDA_DAILY_DEFAULT,
) -> Tuple[pd.Series, pd.Series]:
    """
    Hodrick–Prescott trend / cycle split.

    Parameters
    ----------
    series : daily (or other) frequency; NaNs dropped for the filter then reindexed.
    lam    : smoothing. Default ``HP_LAMBDA_DAILY_DEFAULT`` for daily macro.

    Returns
    -------
    trend, cycle : same index as input (NaN where input NaN or too few points).
    """
    s = series.dropna().astype(float)
    if len(s) < 10:
        nan = pd.Series(np.nan, index=series.index)
        return nan.copy(), nan.copy()
    cyc, tr = hpfilter(s, lamb=lam)
    trend = pd.Series(tr, index=s.index, name=getattr(series, "name", None))
    cycle = pd.Series(cyc, index=s.index, name=getattr(series, "name", None))
    return trend.reindex(series.index), cycle.reindex(series.index)


def macro_mean_in_train_window(
    rebal_date: pd.Timestamp,
    macro: pd.Series,
    trading_index: pd.DatetimeIndex,
    train_years: int,
    how: Literal["mean", "median"] = "mean",
) -> float:
    """
    Average (or median) of ``macro`` over days in
    ``[rebal_date - train_years, rebal_date)`` intersected with ``trading_index``,
    matching the JM training index in Notebook 03.
    """
    train_start = rebal_date - pd.DateOffset(years=train_years)
    mask = (trading_index >= train_start) & (trading_index < rebal_date)
    w = macro.reindex(trading_index[mask]).dropna().astype(float)
    if w.empty:
        return float("nan")
    if how == "median":
        return float(w.median())
    return float(w.mean())


def macro_cycle_at_date(
    macro: pd.Series,
    as_of: pd.Timestamp,
    lam: float = HP_LAMBDA_DAILY_DEFAULT,
) -> float:
    """HP cycle component of ``macro`` at ``as_of`` (last available obs ≤ as_of)."""
    s = macro.loc[:as_of].dropna().astype(float)
    if len(s) < 10:
        return float("nan")
    _, cyc = hpfilter(s, lamb=lam)
    return float(cyc.iloc[-1])


def hac_ols(
    y: np.ndarray,
    x: np.ndarray,
    *,
    maxlags: Optional[int] = None,
) -> Any:
    """
    OLS with Newey–West HAC covariance (statsmodels).

    Parameters
    ----------
    y : shape (n,)
    x : shape (n, k) or (n,) for one regressor; constant is added automatically.
    """
    yv = np.asarray(y, dtype=float).ravel()
    xv = np.asarray(x, dtype=float)
    if xv.ndim == 1:
        xv = xv.reshape(-1, 1)
    if len(yv) != len(xv):
        raise ValueError("y and x must have the same number of rows")
    valid = np.isfinite(yv) & np.all(np.isfinite(xv), axis=1)
    yv, xv = yv[valid], xv[valid]
    if maxlags is None:
        maxlags = max(2, int(np.ceil(len(yv) ** 0.25)))
    X = sm.add_constant(xv, has_constant="add")
    return sm.OLS(yv, X, missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )


def spearman_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    return_pvalue: bool = False,
) -> Union[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """
    Spearman correlation with percentile bootstrap CI on valid paired rows.

    When ``return_pvalue`` is True, also returns the asymptotic two-sided
    p-value from ``scipy.stats.spearmanr`` on the same valid pairs (for multiple
    testing corrections alongside the bootstrap CI).

    Returns
    -------
    rho_hat, lo, hi
        or, if ``return_pvalue``, rho_hat, lo, hi, p_value
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 5:
        if return_pvalue:
            return float("nan"), float("nan"), float("nan"), float("nan")
        return float("nan"), float("nan"), float("nan")
    rho_hat, p_asymp = stats.spearmanr(x, y)
    p_asymp = float(p_asymp) if p_asymp is not None and np.isfinite(p_asymp) else float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = stats.spearmanr(x[idx], y[idx])
        boot[b] = r
    lo, hi = np.quantile(boot, [alpha / 2, 1.0 - alpha / 2])
    if return_pvalue:
        return float(rho_hat), float(lo), float(hi), p_asymp
    return float(rho_hat), float(lo), float(hi)
