"""Time-series and inference helpers for thesis notebooks."""

from src.analysis.time_series_stats import (
    hp_decompose,
    hac_ols,
    spearman_bootstrap_ci,
    macro_mean_in_train_window,
    macro_cycle_at_date,
)

__all__ = [
    "hp_decompose",
    "hac_ols",
    "spearman_bootstrap_ci",
    "macro_mean_in_train_window",
    "macro_cycle_at_date",
]
