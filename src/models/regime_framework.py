"""
models/regime_framework.py
==========================
Implements Algorithms 1 and 2 of Shu et al. (2024):

  Algorithm 1 – Generate asset-specific regime forecasts (JM-XGB) for a
                 fixed jump penalty λ over a given prediction window.

  Algorithm 2 – Optimal jump-penalty selection via time-series cross-
                 validation (5-year validation window, biannual updates).

The framework is the central orchestrator: it calls the JumpModel and
RegimeForecaster, but does NOT own portfolio or backtesting logic.

Open/Closed  : penalty grid, update frequency, and eval metric can be
               overridden without modifying this class.
Liskov        : RegimeForecasterBase can replace RegimeForecaster.
"""

from __future__ import annotations

import logging
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.config.settings import (
    ASSETS,
    ASSETS_NO_DD_IN_JM,
    LAMBDA_CANDIDATES,
    REBAL_MONTHS,
    TRADING_DAYS_YEAR,
    TRAIN_YEARS,
    VAL_YEARS,
    XGB_PARAMS,
)
from src.features.macro_features import MacroFeatureBuilder
from src.features.return_features import ReturnFeatureBuilder
from src.models.jump_model import JumpModel
from src.models.xgb_classifier import RegimeForecaster

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rebalance_dates(
    index: pd.DatetimeIndex,
    start: str,
    end:   str,
    months: Tuple[int, ...] = REBAL_MONTHS,
) -> pd.DatetimeIndex:
    """
    Return the first trading day of each rebalance month within [start, end].
    """
    dates = index[(index >= start) & (index <= end)]
    rebal = []
    seen  = set()
    for dt in dates:
        key = (dt.year, dt.month)
        if dt.month in months and key not in seen:
            rebal.append(dt)
            seen.add(key)
    return pd.DatetimeIndex(rebal)


def _sharpe_01_strategy(
    regime_forecast: pd.Series,
    excess_returns: pd.Series,
    rf: pd.Series,
    tc: float = 5e-4,
) -> float:
    """
    Sharpe ratio of the 0/1 strategy (Bulla et al. 2011).

    At the end of day t, use regime_forecast[t] for day t+1.
    0 = bullish → hold risky asset, 1 = bearish → hold risk-free.
    Transaction cost `tc` (one-way) applied at each regime change.
    """
    # Align on common index
    common = regime_forecast.index.intersection(excess_returns.index)
    f   = regime_forecast.reindex(common)
    er  = excess_returns.reindex(common)
    rf_ = rf.reindex(common).fillna(0.0)

    # Strategy return: use *previous* day forecast for *current* day
    bull = (1 - f.shift(1)).fillna(0).clip(0, 1)   # 1 if bullish

    # Switching cost
    switches   = (f.shift(1) != f.shift(2)).fillna(False).astype(float)
    trade_cost = switches * tc

    strat_ret = bull * er - (1 - bull) * rf_ * 0.0 - trade_cost
    # (when bearish the strategy earns the risk-free which we've already
    #  removed, so bearish periods contribute 0 to *excess* return)

    daily_mean = strat_ret.mean()
    daily_std  = strat_ret.std()
    if len(strat_ret) < 5 or not math.isfinite(float(daily_mean)) or not math.isfinite(float(daily_std)):
        return float("-inf")
    if daily_std < 1e-10:
        return 0.0
    return float(daily_mean / daily_std * np.sqrt(TRADING_DAYS_YEAR))


# ---------------------------------------------------------------------------
# Main Framework
# ---------------------------------------------------------------------------

@dataclass
class RegimeFramework:
    """
    Orchestrates jump model fitting, XGBoost training, lambda tuning,
    and the generation of out-of-sample regime forecasts for all assets.

    Parameters
    ----------
    excess_returns : DataFrame  (dates × assets), daily excess returns
    rf             : Series, daily risk-free rates
    fred_aligned   : aligned FRED macro data
    assets         : list of asset names to process
    lambda_grid    : list of jump-penalty candidates
    train_years    : lookback window (years) for model fitting
    val_years      : validation window (years) for λ tuning
    rebal_months   : months when biannual rebalancing occurs
    transaction_cost : one-way cost (fraction)
    """

    excess_returns:   pd.DataFrame
    rf:               pd.Series
    fred_aligned:     pd.DataFrame
    assets:           List[str]              = field(default_factory=lambda: list(ASSETS))
    lambda_grid:      List[float]            = field(default_factory=lambda: list(LAMBDA_CANDIDATES))
    train_years:      int                    = TRAIN_YEARS
    val_years:        int                    = VAL_YEARS
    rebal_months:     Tuple[int, ...]        = tuple(REBAL_MONTHS)
    asset_jobs:       int                    = 1
    transaction_cost: float                  = 5e-4
    xgb_n_jobs:       int                    = 1

    # Built lazily
    _ret_feat_builder:   ReturnFeatureBuilder   = field(init=False, repr=False)
    _macro_feat_builder: MacroFeatureBuilder     = field(init=False, repr=False)
    _macro_features:     Optional[pd.DataFrame]  = field(default=None, init=False, repr=False)
    _asset_features:     Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._ret_feat_builder   = ReturnFeatureBuilder()
        self._macro_feat_builder = MacroFeatureBuilder()

    def _get_macro_features(self) -> pd.DataFrame:
        if self._macro_features is None:
            self._macro_features = self._macro_feat_builder.build(
                self.fred_aligned, self.excess_returns
            )
        return self._macro_features

    def _get_asset_features(self, asset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cached = self._asset_features.get(asset)
        if cached is not None:
            return cached

        macro_feats = self._get_macro_features()
        ret_feats_xgb = self._ret_feat_builder.build(
            self.excess_returns[asset], asset, for_jm=False
        )
        ret_feats_jm = self._ret_feat_builder.build(
            self.excess_returns[asset], asset, for_jm=True
        )
        X_xgb_full = ret_feats_xgb.join(macro_feats, how="left").ffill()
        cached = (X_xgb_full, ret_feats_jm)
        self._asset_features[asset] = cached
        return cached

    @staticmethod
    def _asset_cache_path(cache_dir: Path, cache_prefix: str, asset: str) -> Path:
        safe_asset = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in asset)
        return cache_dir / f"{cache_prefix}_{safe_asset}.pkl"

    @staticmethod
    def _load_asset_result(path: Path) -> Tuple[str, Dict[pd.Timestamp, int], Dict[pd.Timestamp, float]]:
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _save_asset_result(path: Path, payload: Tuple[str, Dict[pd.Timestamp, int], Dict[pd.Timestamp, float]]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(path)

    # ------------------------------------------------------------------
    # Algorithm 1 – Regime forecasts for a fixed λ
    # ------------------------------------------------------------------

    def generate_forecasts_fixed_lambda(
        self,
        lam:          float,
        pred_start:   str,
        pred_end:     str,
        asset:        str,
    ) -> pd.Series:
        """
        Algorithm 1 for one asset over a prediction window [pred_start, pred_end].

        Every six months:
          1. Fit JM with penalty λ on the preceding 11-year window.
          2. Fit XGBoost on the same window (labels shifted +1 day).
          3. Predict daily regimes for the next 6-month block.

        Returns
        -------
        pd.Series  (0=bullish, 1=bearish) for the prediction window.
        """
        idx   = self.excess_returns.index
        dates = idx[(idx >= pred_start) & (idx <= pred_end)]
        if len(dates) == 0:
            return pd.Series(dtype=int)

        rebal = _rebalance_dates(idx, pred_start, pred_end, self.rebal_months)
        forecasts: Dict[pd.Timestamp, int] = {}

        # Pre-compute full feature matrices once per framework/asset. Algorithm 2
        # calls this method many times while scanning lambdas and validation
        # windows, so rebuilding features here dominates runtime and memory churn.
        X_xgb_full, ret_feats_jm = self._get_asset_features(asset)

        for i, rebal_date in enumerate(rebal):
            # Training window: [rebal_date - train_years, rebal_date)
            train_end   = rebal_date - pd.Timedelta(days=1)
            train_start = rebal_date - pd.DateOffset(years=self.train_years)
            train_idx   = idx[(idx >= train_start) & (idx <= train_end)]
            if len(train_idx) < 252:
                logger.debug("Skipping rebal %s – insufficient training data.", rebal_date)
                continue

            # Prediction sub-window
            if i + 1 < len(rebal):
                block_end = rebal[i + 1] - pd.Timedelta(days=1)
            else:
                block_end = pd.Timestamp(pred_end)
            block_idx = idx[(idx >= rebal_date) & (idx <= block_end)]
            if len(block_idx) == 0:
                continue

            # 1. Fit JM
            X_jm_train = ret_feats_jm.reindex(train_idx).dropna()
            jm = JumpModel(jump_pen=lam)
            jm.fit(X_jm_train.values)

            # Bullish = higher *conditional mean* excess return (not cumulative total:
            # cumulative favours long low-drift regimes when λ is large and one state persists).
            er_train = self.excess_returns[asset].reindex(X_jm_train.index)
            stats    = jm.regime_stats(er_train.values)
            bull_state = max(stats, key=lambda k: stats[k]["mean_daily"])
            # Labels: 0=bullish, 1=bearish
            raw_labels = jm.labels_
            jm_labels  = (raw_labels != bull_state).astype(int)

            # 2. Shift labels +1 day for supervised target
            label_s = pd.Series(jm_labels, index=X_jm_train.index).shift(-1)

            # 3. Fit XGBoost
            X_xgb_train = X_xgb_full.reindex(X_jm_train.index)
            valid_mask  = label_s.notna() & X_xgb_train.notna().all(axis=1)
            if valid_mask.sum() < 50:
                continue

            xgb_params = dict(XGB_PARAMS)
            xgb_params["n_jobs"] = self.xgb_n_jobs
            forecaster = RegimeForecaster(asset_name=asset, xgb_params=xgb_params)
            forecaster.fit(
                X_xgb_train.loc[valid_mask],
                label_s.loc[valid_mask],
            )

            # 4. Daily forecasts for the block
            X_xgb_block = X_xgb_full.reindex(block_idx).ffill()
            regime_block = forecaster.predict_regime(X_xgb_block)

            for dt, val in regime_block.items():
                forecasts[dt] = int(val)

        result = pd.Series(forecasts, name=f"regime_{asset}").sort_index()
        return result.reindex(dates).ffill().fillna(1).astype(int)

    # ------------------------------------------------------------------
    # Algorithm 2 – Optimal λ selection + testing-period forecasts
    # ------------------------------------------------------------------

    def _run_single_asset(
        self,
        asset: str,
        rebal: pd.DatetimeIndex,
        test_end: str,
    ) -> Tuple[str, Dict[pd.Timestamp, int], Dict[pd.Timestamp, float]]:
        logger.info(
            "event=asset_start asset=%s rebalances=%d lambda_candidates=%d "
            "xgb_n_jobs=%d",
            asset,
            len(rebal),
            len(self.lambda_grid),
            self.xgb_n_jobs,
        )

        asset_forecasts: Dict[pd.Timestamp, int] = {}
        asset_lams: Dict[pd.Timestamp, float] = {}

        for i, rebal_date in enumerate(rebal):
            # Validation window: [rebal_date - val_years, rebal_date)
            val_end   = rebal_date - pd.Timedelta(days=1)
            val_start = rebal_date - pd.DateOffset(years=self.val_years)

            # OOS block
            if i + 1 < len(rebal):
                block_end = rebal[i + 1] - pd.Timedelta(days=1)
            else:
                block_end = pd.Timestamp(test_end)
            block_start = rebal_date

            logger.info(
                "event=asset_rebalance_start asset=%s rebalance=%s "
                "rebalance_idx=%d rebalances=%d validation_start=%s "
                "validation_end=%s block_start=%s block_end=%s",
                asset,
                rebal_date.date(),
                i + 1,
                len(rebal),
                val_start.date(),
                val_end.date(),
                block_start.date(),
                block_end.date(),
            )

            best_sr  = float("-inf")
            best_lam = self.lambda_grid[0]

            for lam in self.lambda_grid:
                try:
                    fc = self.generate_forecasts_fixed_lambda(
                        lam       = lam,
                        pred_start= str(val_start.date()),
                        pred_end  = str(val_end.date()),
                        asset     = asset,
                    )
                    er_val = self.excess_returns[asset].reindex(fc.index)
                    rf_val = self.rf.reindex(fc.index).fillna(0.0)
                    sr     = _sharpe_01_strategy(
                        fc, er_val, rf_val, tc=self.transaction_cost
                    )
                    if math.isfinite(sr) and sr > best_sr:
                        best_sr  = float(sr)
                        best_lam = float(lam)
                except Exception as exc:
                    logger.warning(
                        "[%s] λ tuning: λ=%s failed at rebal %s: %s",
                        asset, lam, rebal_date.date(), exc,
                    )
                    continue

            if not math.isfinite(best_sr):
                logger.warning(
                    "[%s] No finite validation Sharpe at rebal %s — keeping λ=%.3f",
                    asset, rebal_date.date(), best_lam,
                )

            logger.info(
                "event=asset_rebalance_done asset=%s rebalance=%s "
                "best_lambda=%.6g validation_sharpe=%.6g",
                asset,
                rebal_date.date(),
                best_lam,
                best_sr,
            )
            asset_lams[rebal_date] = best_lam

            # Generate OOS forecasts with optimal λ
            try:
                oos_fc = self.generate_forecasts_fixed_lambda(
                    lam        = best_lam,
                    pred_start = str(block_start.date()),
                    pred_end   = str(block_end.date()),
                    asset      = asset,
                )
                for dt, val in oos_fc.items():
                    asset_forecasts[dt] = int(val)
            except Exception as exc:
                logger.warning("OOS forecast failed for %s at %s: %s", asset, rebal_date, exc)

        logger.info(
            "event=asset_done asset=%s forecast_days=%d lambda_points=%d",
            asset,
            len(asset_forecasts),
            len(asset_lams),
        )
        return asset, asset_forecasts, asset_lams

    def run(
        self,
        test_start: str,
        test_end:   str,
        asset_cache_dir: Optional[str | Path] = None,
        cache_prefix: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """
        Algorithm 2: biannual λ tuning + out-of-sample forecast generation.

        For every 6-month block in [test_start, test_end]:
          1. For each λ, run Algorithm 1 over the 5-year validation window.
          2. Compute 0/1 Sharpe ratio.
          3. Pick optimal λ; generate OOS forecasts for the next 6 months.

        Returns
        -------
        regime_forecasts : DataFrame  (dates × assets), 0=bull / 1=bear
        optimal_lambdas  : dict  asset → Series of time-stamped optimal λ
        """
        idx     = self.excess_returns.index
        rebal   = _rebalance_dates(idx, test_start, test_end, self.rebal_months)

        cache_dir_path = Path(asset_cache_dir) if asset_cache_dir is not None else None
        if cache_dir_path is not None:
            cache_dir_path.mkdir(parents=True, exist_ok=True)
        prefix = cache_prefix or (
            "regime_" + "_".join(str(m) for m in self.rebal_months)
        )
        if cache_dir_path is not None:
            existing = [
                asset for asset in self.assets
                if self._asset_cache_path(cache_dir_path, prefix, asset).is_file()
            ]
            logger.info(
                "event=asset_checkpoint_status prefix=%s cached_assets=%d "
                "total_assets=%d cache_dir=%s",
                prefix,
                len(existing),
                len(self.assets),
                cache_dir_path,
            )

        def run_or_load(asset: str) -> Tuple[str, Dict[pd.Timestamp, int], Dict[pd.Timestamp, float]]:
            if cache_dir_path is None:
                return self._run_single_asset(asset, rebal, test_end)

            path = self._asset_cache_path(cache_dir_path, prefix, asset)
            if path.is_file():
                logger.info(
                    "event=asset_checkpoint_load asset=%s path=%s",
                    asset,
                    path,
                )
                return self._load_asset_result(path)

            result = self._run_single_asset(asset, rebal, test_end)
            self._save_asset_result(path, result)
            logger.info(
                "event=asset_checkpoint_save asset=%s path=%s",
                asset,
                path,
            )
            return result

        n_jobs = max(1, min(len(self.assets), int(self.asset_jobs)))
        logger.info(
            "event=framework_run_start assets=%d rebalances=%d asset_jobs=%d "
            "xgb_n_jobs=%d prefix=%s",
            len(self.assets),
            len(rebal),
            n_jobs,
            self.xgb_n_jobs,
            prefix,
        )
        if n_jobs > 1:
            asset_results = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(run_or_load)(asset)
                for asset in self.assets
            )
        else:
            asset_results = [
                run_or_load(asset)
                for asset in self.assets
            ]

        all_forecasts: Dict[str, Dict[pd.Timestamp, int]] = {
            asset: forecasts for asset, forecasts, _ in asset_results
        }
        optimal_lams:  Dict[str, Dict[pd.Timestamp, float]] = {
            asset: lams for asset, _, lams in asset_results
        }

        regime_df = pd.DataFrame(
            {a: pd.Series(all_forecasts[a]) for a in self.assets}
        ).sort_index()

        opt_lam_series = {
            a: pd.Series(optimal_lams[a], name=f"best_lam_{a}")
            for a in self.assets
        }

        return regime_df, opt_lam_series
