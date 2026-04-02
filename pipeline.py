"""
pipeline.py
===========
Command-line entry point that runs the full Shu et al. (2024) replication
pipeline end-to-end:

  1. Download / load data
  2. Run JM-XGB regime framework (Algorithm 1 + 2)
  3. Run all portfolio backtests
  4. Print performance tables

Usage:
    python pipeline.py [--force-download] [--no-cache]

Output:
    results/ directory containing CSVs, pickles, and PNG figures.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

from src.config.settings import (
    ASSETS, ASSET_TICKERS, FRED_SERIES,
    DATA_START, DATA_END, TEST_START, TEST_END,
)
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.regime_framework import RegimeFramework
from src.backtest.engine import BacktestEngine
from src.portfolio.strategies import build_strategy
from src.backtest.metrics import strategy_table
from src.utils.helpers import setup_logging, wealth_curve

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shu et al. (2024) replication pipeline")
    p.add_argument("--force-download", action="store_true",
                   help="Re-download data even if cache exists")
    p.add_argument("--no-cache", action="store_true",
                   help="Do not use any cached intermediate results")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)
    no_cache = args.no_cache

    # ── 1. Data ──────────────────────────────────────────────────────
    logger.info("=== Step 1: Data Loading ===")
    loader = DataLoader()
    prices = loader.load_prices(ASSET_TICKERS, DATA_START, DATA_END,
                                force_download=args.force_download)
    fred   = loader.load_fred(FRED_SERIES,     DATA_START, DATA_END,
                              force_download=args.force_download)

    prep = DataPreprocessor()
    excess_returns, rf_daily, fred_aligned = prep.prepare(prices, fred)
    total_returns = prices.pct_change().reindex(excess_returns.index).ffill()

    logger.info("Returns: %s  [%s → %s]",
                excess_returns.shape,
                excess_returns.index[0].date(),
                excess_returns.index[-1].date())

    # ── 2. Regime Forecasts ──────────────────────────────────────────
    logger.info("=== Step 2: JM-XGB Regime Framework ===")
    regime_cache = RESULTS / "regime_forecasts.pkl"
    lam_cache    = RESULTS / "optimal_lambdas.pkl"

    if not no_cache and regime_cache.exists():
        logger.info("Loading cached regime forecasts.")
        with open(regime_cache, "rb") as f:
            regime_forecasts = pickle.load(f)
        with open(lam_cache, "rb") as f:
            optimal_lambdas = pickle.load(f)
    else:
        framework = RegimeFramework(
            excess_returns = excess_returns,
            rf             = rf_daily,
            fred_aligned   = fred_aligned,
            assets         = ASSETS,
        )
        regime_forecasts, optimal_lambdas = framework.run(TEST_START, TEST_END)
        with open(regime_cache, "wb") as f:
            pickle.dump(regime_forecasts, f)
        with open(lam_cache, "wb") as f:
            pickle.dump(optimal_lambdas, f)

    logger.info("Regime forecasts shape: %s", regime_forecasts.shape)
    logger.info("Mean %% bearish:\n%s",
                (regime_forecasts == 1).mean().round(3).to_string())

    # ── 3. Portfolio Backtests ────────────────────────────────────────
    logger.info("=== Step 3: Portfolio Backtesting ===")
    port_cache = RESULTS / "portfolio_results.pkl"

    if not no_cache and port_cache.exists():
        logger.info("Loading cached portfolio results.")
        with open(port_cache, "rb") as f:
            all_results = pickle.load(f)
    else:
        engine = BacktestEngine(
            excess_returns = excess_returns,
            total_returns  = total_returns,
            rf             = rf_daily,
            assets         = ASSETS,
        )
        strategy_names = [
            "60/40", "MinVar", "MinVar(JM-XGB)", "MV", "MV(JM-XGB)", "EW", "EW(JM-XGB)"
        ]
        strategies  = {n: build_strategy(n, ASSETS) for n in strategy_names}
        all_results = engine.run_all(
            strategies        = strategies,
            test_start        = TEST_START,
            test_end          = TEST_END,
            regime_forecasts  = regime_forecasts,
        )
        with open(port_cache, "wb") as f:
            pickle.dump(all_results, f)

    # ── 4. Results ───────────────────────────────────────────────────
    logger.info("=== Step 4: Results ===")
    metrics = {name: res["metrics"] for name, res in all_results.items()}
    tbl     = strategy_table(metrics)

    print("\n" + "=" * 60)
    print("Table 6 — Portfolio Performance (2007-2023)")
    print("=" * 60)
    print(tbl.to_string())
    print("=" * 60)

    tbl.to_csv(RESULTS / "table6_portfolio_performance.csv")
    logger.info("Results saved to %s/", RESULTS)


if __name__ == "__main__":
    main()
