"""
utils/helpers.py
================
Shared utility functions used across multiple modules.

DRY: centralise helpers that would otherwise be duplicated.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Wealth-curve helpers
# ---------------------------------------------------------------------------

def wealth_curve(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    """Convert a daily return series to a cumulative wealth curve."""
    return start_value * (1 + returns).cumprod()


def plot_wealth_curves(
    curves: Dict[str, pd.Series],
    title:  str = "Strategy Wealth Curves",
    log_scale: bool = True,
    figsize: Tuple[int, int] = (12, 5),
    bear_periods: Optional[pd.Series] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot multiple wealth curves on a single axis.

    Parameters
    ----------
    curves       : {label: pd.Series of cumulative wealth}
    title        : plot title
    log_scale    : use log-scale y-axis
    bear_periods : optional binary series (1=bear) to shade bearish periods
    ax           : existing Axes object; creates one if None
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    colors = ["steelblue", "darkorange", "green", "red", "purple", "brown"]
    for (label, curve), color in zip(curves.items(), colors):
        ax.plot(curve.index, curve.values, label=label, color=color, linewidth=1.2)

    if bear_periods is not None:
        _shade_bear_periods(ax, bear_periods)

    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    return ax


def _shade_bear_periods(ax: plt.Axes, bear: pd.Series) -> None:
    """Add shaded regions for bearish forecasts."""
    in_bear    = False
    bear_start = None
    for dt, val in bear.items():
        if val == 1 and not in_bear:
            bear_start = dt
            in_bear    = True
        elif val != 1 and in_bear:
            ax.axvspan(bear_start, dt, alpha=0.15, color="red")
            in_bear = False
    if in_bear and bear_start is not None:
        ax.axvspan(bear_start, bear.index[-1], alpha=0.15, color="red")


# ---------------------------------------------------------------------------
# Return-series helpers
# ---------------------------------------------------------------------------

def regime_conditional_returns(
    excess_returns: pd.Series,
    labels:         np.ndarray,
    dates:          pd.DatetimeIndex,
    regime:         int,
) -> float:
    """
    Average daily excess return during periods classified as `regime`.

    Parameters
    ----------
    excess_returns : full excess return series (aligned to `dates`)
    labels         : JM label array
    dates          : DatetimeIndex corresponding to `labels`
    regime         : which label (0 or 1)

    Returns
    -------
    Mean daily excess return for the requested regime.
    """
    mask = labels == regime
    sub  = excess_returns.reindex(dates)[mask]
    return float(sub.mean()) if len(sub) > 0 else 0.0


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a sensible format."""
    logging.basicConfig(
        level  = level,
        format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt= "%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_results(obj: object, path: str | Path) -> None:
    """Pickle an object to disk."""
    import pickle
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(obj, f)
    logger.info("Saved results to %s", p)


def load_results(path: str | Path) -> object:
    """Unpickle an object from disk."""
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)
