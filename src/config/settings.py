"""
settings.py
===========
Centralised configuration for the Shu et al. (2024) replication.

All hard-coded constants from the paper live here so that every other module
imports from a single source of truth (DRY / Single-Responsibility).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Asset universe  (Table 1 & Appendix A, Shu et al. 2024)
# ---------------------------------------------------------------------------

# Ordered list of the 12 risky asset abbreviations used throughout the paper.
ASSETS: List[str] = [
    "LargeCap", "MidCap", "SmallCap",
    "EAFE", "EM",
    "AggBond", "Treasury", "HighYield", "Corporate",
    "REIT",
    "Commodity", "Gold",
]

# Yahoo-Finance tickers that closely track the indexes used in the paper.
# Note: Bloomberg indexes are not freely available; ETF proxies introduce
# small tracking differences but cover the full 1991-2023 history via
# adjusted close prices.
ASSET_TICKERS: Dict[str, str] = {
    "LargeCap":  "IVV",    # S&P 500
    "MidCap":    "IJH",    # S&P MidCap 400
    "SmallCap":  "IWM",    # Russell 2000
    "EAFE":      "EFA",    # MSCI EAFE
    "EM":        "EEM",    # MSCI EM
    "AggBond":   "AGG",    # Bloomberg US Aggregate Bond
    "Treasury":  "SPTL",   # US Long Treasury (replaces LUTLTRUU)
    "HighYield": "HYG",    # iBoxx Liquid High Yield
    "Corporate": "SPBO",   # Bloomberg US Corporate
    "REIT":      "IYR",    # Dow Jones US Real Estate
    "Commodity": "DBC",    # DBIQ Diversified Commodity
    "Gold":      "GLD",    # LBMA Gold
}

# Backup Yahoo tickers used to splice early history for ETFs with short lives
ASSET_TICKERS_BACKUP: Dict[str, str] = {
    "Treasury":  "^TYX",   # 30-year yield proxy (not ideal, used only as fallback)
    "HighYield": "VWEHX",  # Vanguard High-Yield (mutual fund, available from 1978)
}

# ---------------------------------------------------------------------------
# FRED series for macro features and risk-free rate  (Section 3.3 & paper body)
# ---------------------------------------------------------------------------

FRED_SERIES: Dict[str, str] = {
    "rf":       "DTB3",    # 3-month T-Bill (risk-free rate proxy)
    "y2":       "DGS2",    # 2-year Treasury constant-maturity yield
    "y10":      "DGS10",   # 10-year Treasury constant-maturity yield
    "vix":      "VIXCLS",  # CBOE VIX
    "stlfi":    "STLFSI",  # St. Louis Fed Financial Stress Index (weekly; aligned in preprocessor)
}

# ---------------------------------------------------------------------------
# Sample-period definitions  (Section 3–5)
# ---------------------------------------------------------------------------

DATA_START:       str = "1991-01-01"
DATA_END:         str = "2023-12-31"

# Initial training window ends at the first prediction date
FIRST_PRED_START: str = "2002-01-01"   # start of first validation window

# Out-of-sample testing period
TEST_START:       str = "2007-01-01"
TEST_END:         str = "2023-12-31"

# Validation window used for λ tuning (5-year lookback from TEST_START)
VAL_YEARS:        int = 5              # years
TRAIN_YEARS:      int = 11             # years  (lookback window for JM + XGB)

# ---------------------------------------------------------------------------
# Statistical Jump Model hyper-parameters  (Section 3.2)
# ---------------------------------------------------------------------------

JM_N_STATES:  int = 2      # K=2 (bullish / bearish)
JM_MAX_ITER:  int = 300    # coordinate-descent iterations
JM_TOL:       float = 1e-6 # convergence tolerance

# Jump-penalty search grid: 0 and a log-spaced grid from 0.5 to 100
# "ranging from 0.0 to 100.0, distributed evenly on a logarithmic scale"
import numpy as np
LAMBDA_CANDIDATES: List[float] = (
    [0.0] + list(np.logspace(-1, 2, 30))  # 0, ~0.1, …, 100
)

# Features for the JM (Table 2):
#   EWM Downside Deviation (log scale): halflives 5, 21
#   EWM Average Return:                 halflives 5, 10, 21
#   EWM Sortino Ratio:                  halflives 5, 10, 21
JM_DD_HALFLIVES:     List[int] = [5, 21]
JM_RET_HALFLIVES:    List[int] = [5, 10, 21]
JM_SORT_HALFLIVES:   List[int] = [5, 10, 21]

# Assets for which DD features are excluded from the JM  (footnote 13)
ASSETS_NO_DD_IN_JM: List[str] = ["AggBond", "Treasury", "Gold"]

# ---------------------------------------------------------------------------
# XGBoost hyper-parameters  (Section 3.3)
# ---------------------------------------------------------------------------

XGB_PARAMS: Dict = {
    "n_estimators":    100,
    "max_depth":       6,
    "learning_rate":   0.3,
    "subsample":       1.0,
    "colsample_bytree": 1.0,
    "eval_metric":     "logloss",
    "random_state":    42,
    "n_jobs":          -1,
}

XGB_THRESHOLD: float = 0.5   # probability cut-off for bearish label

# Halflife (days) for EWM smoothing of XGBoost probability output (footnote 14)
# 0 means no smoothing
PROB_SMOOTH_HL: Dict[str, int] = {
    "LargeCap":  8,
    "MidCap":    8,
    "SmallCap":  8,
    "REIT":      8,
    "AggBond":   8,
    "Treasury":  8,
    "Commodity": 4,
    "Gold":      4,
    "Corporate": 2,
    "EM":        0,
    "EAFE":      0,
    "HighYield": 0,
}

# ---------------------------------------------------------------------------
# Macro features  (Table 3)
# ---------------------------------------------------------------------------

MACRO_SLOPE_HL:     int = 10    # EWMA halflife for yield curve slope
MACRO_YIELD_HL:     int = 21    # EWMA halflife for 2Y yield diff
MACRO_SLOPE_D_HL:   int = 21    # EWMA halflife for slope diff
MACRO_VIX_HL:       int = 63    # EWMA halflife for VIX log-diff
MACRO_CORR_WINDOW:  int = 252   # rolling window for stock-bond correlation

# Assets used to compute stock-bond correlation (Section 3.3)
CORR_STOCK_ASSET: str = "LargeCap"
CORR_BOND_ASSET:  str = "AggBond"

# ---------------------------------------------------------------------------
# Portfolio optimisation parameters  (Section 4)
# ---------------------------------------------------------------------------

TRANSACTION_COST_BPS: float = 5.0          # one-way cost in basis points
TRANSACTION_COST:     float = 5e-4         # as a fraction

WEIGHT_UB:    float = 0.40    # per-asset upper bound  (Section 4.1)
LEVERAGE_MAX: float = 1.00    # maximum leverage (long-only, no short RF)

# Risk aversion
GAMMA_RISK_MINVAR:    float = 10.0
GAMMA_RISK_MV:        float = 5.0
GAMMA_RISK_MV_JMXGB: float = 10.0

# Trade aversion
GAMMA_TRADE_MINVAR:         float = 0.0   # original MinVar: no trading cost term
GAMMA_TRADE_MINVAR_JMXGB:  float = 1.0
GAMMA_TRADE_MV:             float = 0.0
GAMMA_TRADE_MV_JMXGB:      float = 1.0

# Return forecast values for MinVar(JM-XGB)  (Section 4.2)
MINVAR_BULL_MU_BPS: float = 10.0          # μ for bullish assets  (bps)
MINVAR_BULL_MU:     float = 10e-4         # as fraction per day
MINVAR_BEAR_MU:     float = 0.0

# MV: cap bearish return forecast (Section 4.3)
MV_BEAR_CAP_BPS: float = -10.0
MV_BEAR_CAP:     float = -10e-4

# MV baseline: EWM halflife for naive return forecast (Section 4.3)
MV_BASELINE_HL_DAYS: int = 252 * 5        # 5-year halflife in trading days

# Covariance estimation halflife  (Section 4.2 & 4.3)
COV_HL_DAYS: int = 252                    # 252-day halflife

# Minimum bullish count to avoid full risk-free allocation  (Section 4)
MIN_BULLISH_ASSETS: int = 4               # if < 4 bullish → 100% risk-free

# 60/40 benchmark weights  (Table 5)
BENCHMARK_60_40: Dict[str, float] = {
    "LargeCap":  0.10,
    "MidCap":    0.05,
    "SmallCap":  0.05,
    "EAFE":      0.05,
    "EM":        0.05,
    "REIT":      0.10,
    "HighYield": 0.10,
    "Commodity": 0.05,
    "Gold":      0.05,
    "Treasury":  0.10,
    "Corporate": 0.10,
    "AggBond":   0.20,
}

# ---------------------------------------------------------------------------
# Backtesting schedule
# ---------------------------------------------------------------------------

REBAL_MONTHS: Tuple[int, ...] = (1, 7)   # January and July (biannual)
TRADING_DAYS_YEAR: int = 252
