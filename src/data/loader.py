"""
data/loader.py
==============
Responsible solely for fetching and caching raw price / yield data.

Single Responsibility : download + cache.
Open/Closed           : new sources can be added without touching callers.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Default cache directory (project root / data / raw)
_DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "raw"


class DataLoader:
    """Download and cache daily price + macro data."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_prices(
        self,
        tickers: Dict[str, str],
        start: str,
        end: str,
        force_download: bool = False,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of daily *adjusted close* prices indexed by date.

        Parameters
        ----------
        tickers : mapping  asset_name → Yahoo Finance ticker
        start, end : ISO-8601 date strings
        force_download : bypass cache
        """
        cache_file = self.cache_dir / f"prices_{start}_{end}.pkl"

        if not force_download and cache_file.exists():
            logger.info("Loading prices from cache: %s", cache_file)
            return pd.read_pickle(cache_file)

        logger.info("Downloading prices from Yahoo Finance …")
        raw = yf.download(
            list(tickers.values()),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        # yfinance returns MultiIndex when multiple tickers
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]].rename(columns={"Close": list(tickers.values())[0]})

        # Rename columns: ticker → asset name
        inv = {v: k for k, v in tickers.items()}
        prices = prices.rename(columns=inv)

        # Keep only the assets requested (in order)
        prices = prices[[a for a in tickers.keys() if a in prices.columns]]

        prices.to_pickle(cache_file)
        logger.info("Saved prices to cache: %s", cache_file)
        return prices

    def load_fred(
        self,
        series: Dict[str, str],
        start: str,
        end: str,
        force_download: bool = False,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of FRED daily series.

        Parameters
        ----------
        series : mapping  name → FRED series ID
        start, end : ISO-8601 date strings
        """
        cache_file = self.cache_dir / f"fred_{start}_{end}.pkl"

        if not force_download and cache_file.exists():
            logger.info("Loading FRED data from cache: %s", cache_file)
            return pd.read_pickle(cache_file)

        logger.info("Downloading FRED data …")
        frames: Dict[str, pd.Series] = {}

        try:
            import pandas_datareader.data as web  # type: ignore

            for name, sid in series.items():
                try:
                    s = web.DataReader(sid, "fred", start=start, end=end)
                    frames[name] = s.squeeze()
                    logger.info("  FRED %s (%s): %d observations", name, sid, len(s))
                except Exception as exc:
                    logger.warning("  Could not download %s: %s", sid, exc)
        except ImportError:
            logger.error(
                "pandas_datareader is required for FRED downloads. "
                "Install with: pip install pandas-datareader"
            )
            raise

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df.to_pickle(cache_file)
        logger.info("Saved FRED data to cache: %s", cache_file)
        return df
