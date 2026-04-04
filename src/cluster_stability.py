"""
Cluster stability diagnostics for JM refits on the biannual schedule used in
λ cross-validation (January / July rebalancing).

For each refit date we report:
  • internal cluster validity indices on the training feature matrix (geometry)
  • (from pairwise comparisons) agreement / ARI between consecutive windows

Kept as ``src.cluster_stability`` (flat module) so notebooks match imports like
``src.utils.helpers`` without an extra sub-package.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

logger = logging.getLogger(__name__)


def mean_run_length(labels: np.ndarray) -> float:
    """Average length of consecutive same-regime episodes."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return 0.0
    run_lens: List[int] = []
    curr, cnt = int(labels[0]), 1
    for l in labels[1:]:
        li = int(l)
        if li == curr:
            cnt += 1
        else:
            run_lens.append(cnt)
            curr, cnt = li, 1
    run_lens.append(cnt)
    return float(np.mean(run_lens))


def pairwise_stability(fit_a: Dict[str, Any], fit_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two JM fits on their overlapping date range (consecutive refits).

    Returns
    -------
    dict with 'agreement', 'ari', 'n_overlap'
    """
    dates_a = pd.DatetimeIndex(fit_a["dates"])
    dates_b = pd.DatetimeIndex(fit_b["dates"])

    overlap = dates_a.intersection(dates_b)
    if len(overlap) < 20:
        return {"agreement": np.nan, "ari": np.nan, "n_overlap": len(overlap)}

    labels_a = pd.Series(fit_a["labels"], index=dates_a).reindex(overlap).values
    labels_b = pd.Series(fit_b["labels"], index=dates_b).reindex(overlap).values

    agreement_direct = float((labels_a == labels_b).mean())
    agreement_swapped = float((labels_a == (1 - labels_b)).mean())
    agreement = max(agreement_direct, agreement_swapped)

    if agreement_swapped > agreement_direct:
        labels_b = 1 - labels_b

    ari = float(adjusted_rand_score(labels_a, labels_b))

    return {"agreement": agreement, "ari": ari, "n_overlap": len(overlap)}


def internal_validity_scores(
    X: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Silhouette (↑), Davies–Bouldin (↓), Calinski–Harabasz (↑).

    Degenerate partitions (single cluster, too few points) yield NaNs.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    out: Dict[str, float] = {
        "silhouette": float("nan"),
        "davies_bouldin": float("nan"),
        "calinski_harabasz": float("nan"),
    }
    if X.shape[0] < 4 or X.shape[0] != len(labels):
        return out
    uniq = np.unique(labels)
    if uniq.size < 2:
        return out
    counts = np.bincount(labels, minlength=int(uniq.max()) + 1)
    if (counts > 0).sum() < 2 or counts[counts > 0].min() < 2:
        try:
            out["davies_bouldin"] = float(davies_bouldin_score(X, labels))
        except Exception:
            pass
        try:
            out["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
        except Exception:
            pass
        return out

    try:
        out["silhouette"] = float(silhouette_score(X, labels, metric="euclidean"))
    except Exception as exc:
        logger.debug("silhouette_score failed: %s", exc)
    try:
        out["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    except Exception as exc:
        logger.debug("davies_bouldin_score failed: %s", exc)
    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    except Exception as exc:
        logger.debug("calinski_harabasz_score failed: %s", exc)
    return out


def build_pairwise_stability_df(
    jm_fits: Dict[str, Dict[Any, Dict[str, Any]]],
    assets: List[str],
) -> pd.DataFrame:
    """Agreement / ARI between each consecutive pair of biannual refits."""
    rows: List[Dict[str, Any]] = []
    for asset in assets:
        rebal_dates = sorted(jm_fits[asset].keys())
        for i in range(1, len(rebal_dates)):
            d_prev, d_curr = rebal_dates[i - 1], rebal_dates[i]
            m = pairwise_stability(jm_fits[asset][d_prev], jm_fits[asset][d_curr])
            rows.append(
                {
                    "Asset": asset,
                    "Date_prev": d_prev,
                    "Date_curr": d_curr,
                    "Agreement": m["agreement"],
                    "ARI": m["ari"],
                    "N_overlap": m["n_overlap"],
                }
            )
    return pd.DataFrame(rows)


def build_internal_validity_df(
    jm_fits: Dict[str, Dict[Any, Dict[str, Any]]],
    assets: List[str],
) -> pd.DataFrame:
    """
    One row per (Asset, Rebal_date): internal CVIs on the JM training matrix.

    Requires each fit dict to contain ``'X'`` (feature matrix rows aligned with
    ``labels``). Re-fit or delete ``stability_jm_fits.pkl`` if ``X`` is missing.
    """
    rows: List[Dict[str, Any]] = []
    for asset in assets:
        for date in sorted(jm_fits[asset].keys()):
            fit = jm_fits[asset][date]
            X = fit.get("X")
            labels = np.asarray(fit["labels"], dtype=int)
            if X is None:
                rows.append(
                    {
                        "Asset": asset,
                        "Rebal_date": date,
                        "silhouette": float("nan"),
                        "davies_bouldin": float("nan"),
                        "calinski_harabasz": float("nan"),
                        "has_features": False,
                    }
                )
                continue
            Xv = np.asarray(X, dtype=float)
            scores = internal_validity_scores(Xv, labels)
            rows.append(
                {
                    "Asset": asset,
                    "Rebal_date": date,
                    "has_features": True,
                    **scores,
                }
            )
    return pd.DataFrame(rows)


def build_centroid_drift_df(
    jm_fits: Dict[str, Dict[Any, Dict[str, Any]]],
    assets: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for asset in assets:
        rebal_dates = sorted(jm_fits[asset].keys())
        for i in range(1, len(rebal_dates)):
            d_prev, d_curr = rebal_dates[i - 1], rebal_dates[i]
            fp, fc = jm_fits[asset][d_prev], jm_fits[asset][d_curr]
            c_prev, c_curr = fp["centroids"], fc["centroids"]
            bull_prev, bull_curr = int(fp["bull_state"]), int(fc["bull_state"])
            drift_bull = float(np.linalg.norm(c_prev[bull_prev] - c_curr[bull_curr]))
            drift_bear = float(
                np.linalg.norm(c_prev[1 - bull_prev] - c_curr[1 - bull_curr])
            )
            rows.append(
                {
                    "Asset": asset,
                    "Date": d_curr,
                    "Drift_Bull": drift_bull,
                    "Drift_Bear": drift_bear,
                    "Drift_Mean": (drift_bull + drift_bear) / 2,
                }
            )
    return pd.DataFrame(rows)


def build_occupancy_df(
    jm_fits: Dict[str, Dict[Any, Dict[str, Any]]],
    assets: List[str],
) -> pd.DataFrame:
    rows = []
    for asset in assets:
        for date, fit in jm_fits[asset].items():
            labels = fit["labels"]
            rows.append(
                {
                    "Asset": asset,
                    "Date": date,
                    "Pct_Bear": float(np.mean(labels)),
                }
            )
    return pd.DataFrame(rows)


def build_persistence_df(
    jm_fits: Dict[str, Dict[Any, Dict[str, Any]]],
    assets: List[str],
) -> pd.DataFrame:
    rows = []
    for asset in assets:
        for date, fit in jm_fits[asset].items():
            lab = np.asarray(fit["labels"])
            mrl = mean_run_length(lab)
            rows.append(
                {
                    "Asset": asset,
                    "Date": date,
                    "Mean_Run_Length": mrl,
                    "Persistence_Ratio": mrl / len(lab) if len(lab) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def summarize_internal_validity_by_asset(cvi_df: pd.DataFrame) -> pd.DataFrame:
    """Asset-level means of per–rebalance CVIs (for a static scorecard)."""
    if "has_features" in cvi_df.columns:
        sub = cvi_df[cvi_df["has_features"]]
        if sub.empty:
            sub = cvi_df
    else:
        sub = cvi_df
    g = sub.groupby("Asset")[
        ["silhouette", "davies_bouldin", "calinski_harabasz"]
    ].mean()
    g.columns = [
        "Silhouette (mean)",
        "Davies–Bouldin (mean)",
        "Calinski–Harabasz (mean)",
    ]
    return g


def assert_fits_have_features(
    jm_fits: Dict[str, Dict[Any, Dict[str, Any]]],
    assets: List[str],
) -> bool:
    for a in assets:
        fits = jm_fits.get(a, {})
        if not fits:
            return False
        for fit in fits.values():
            if fit.get("X") is None:
                return False
    return True
