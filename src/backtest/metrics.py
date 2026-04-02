"""
backtest/metrics.py
===================
Performance metrics used in Tables 4, 6, 7, 8, 9 of Shu et al. (2024).

Metrics:
  • Annualised excess return
  • Annualised volatility
  • Sharpe ratio
  • Maximum drawdown (MDD)
  • Calmar ratio
  • Annualised turnover
  • Average leverage

Single Responsibility : only metric computation from return series.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.config.settings import TRADING_DAYS_YEAR


def compute_metrics(
    port_returns: pd.Series,
    turnover:     pd.Series,
    weights:      pd.DataFrame,
    rf:           pd.Series,
) -> Dict[str, float]:
    """
    Compute the full set of performance metrics.

    Parameters
    ----------
    port_returns : daily portfolio *total* returns (after costs)
    turnover     : daily one-way turnover (sum of |Δw|)
    weights      : daily portfolio weights (dates × assets)
    rf           : daily risk-free rate

    Returns
    -------
    dict of annualised metrics.
    """
    rf_aligned = rf.reindex(port_returns.index).fillna(0.0)
    excess     = port_returns - rf_aligned

    ann_ret   = float(excess.mean() * TRADING_DAYS_YEAR)
    ann_vol   = float(excess.std(ddof=1) * np.sqrt(TRADING_DAYS_YEAR))
    sharpe    = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0
    mdd       = float(_max_drawdown(port_returns))
    calmar    = abs(ann_ret / mdd) if mdd < -1e-6 else 0.0
    ann_to    = float(turnover.mean() * TRADING_DAYS_YEAR)
    avg_lev   = float(weights.sum(axis=1).mean())

    return {
        "Return":    ann_ret,
        "Volatility": ann_vol,
        "Sharpe":    sharpe,
        "MDD":       mdd,
        "Calmar":    calmar,
        "Turnover":  ann_to,
        "Leverage":  avg_lev,
    }


def _max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown of a total-return series."""
    cum   = (1 + returns).cumprod()
    peak  = cum.cummax()
    dd    = (cum - peak) / peak
    return float(dd.min())


def strategy_table(
    results: Dict[str, Dict[str, float]],
    pct_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Format a dict of {strategy_name: metrics_dict} into a display DataFrame.
    """
    if pct_cols is None:
        pct_cols = ["Return", "Volatility", "MDD"]
    df = pd.DataFrame(results).T
    for c in pct_cols:
        if c in df.columns:
            df[c] = df[c].map(lambda x: f"{x:.1%}")
    for c in ["Sharpe", "Calmar"]:
        if c in df.columns:
            df[c] = df[c].map(lambda x: f"{x:.2f}")
    for c in ["Turnover", "Leverage"]:
        if c in df.columns:
            df[c] = df[c].map(lambda x: f"{x:.2f}")
    return df
