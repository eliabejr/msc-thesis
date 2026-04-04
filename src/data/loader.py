"""
data/loader.py
==============
Responsible solely for fetching and caching raw price / yield data.

Single Responsibility : download + cache.
Open/Closed           : new sources can be added without touching callers.
"""

from __future__ import annotations

import io
import logging
import pickle
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# FRED graph CSV (no API key). pandas-datareader is unmaintained and breaks on newer pandas.
_FRED_GRAPH_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
_FRED_UA = "shu2024-replication/1.0 (research; +https://github.com/pydata/pandas-datareader substitute)"


def _fred_series_from_graph_csv(series_id: str, start: str, end: str) -> pd.Series:
    """Download one FRED series via the public graph CSV endpoint."""
    q = urllib.parse.urlencode({"id": series_id, "cosd": start, "coed": end})
    url = f"{_FRED_GRAPH_CSV}?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": _FRED_UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"FRED download failed for {series_id}: {exc}") from exc

    df = pd.read_csv(io.BytesIO(raw), na_values=[".", ""])
    date_col = None
    for cand in ("DATE", "date"):
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        for c in df.columns:
            if str(c).lower() == "observation_date":
                date_col = c
                break
    if date_col is None:
        raise ValueError(
            f"No date column in FRED CSV for {series_id}; columns={list(df.columns)}"
        )
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    value_cols = [c for c in df.columns if c != date_col]
    if not value_cols:
        raise ValueError(f"No value column in FRED CSV for {series_id}")
    val_col = series_id if series_id in df.columns else value_cols[0]
    s = pd.Series(df[val_col].values, index=df[date_col], name=series_id)
    s = s.sort_index().astype(float)
    return s[~s.index.duplicated(keep="last")]

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
            cached = pd.read_pickle(cache_file)
            if set(series.keys()) <= set(cached.columns):
                logger.info("Loading FRED data from cache: %s", cache_file)
                return cached
            logger.info(
                "FRED cache stale (missing columns); re-downloading: %s",
                cache_file,
            )

        logger.info("Downloading FRED data …")
        frames: Dict[str, pd.Series] = {}

        for name, sid in series.items():
            try:
                s = _fred_series_from_graph_csv(sid, start, end)
                frames[name] = s
                logger.info("  FRED %s (%s): %d observations", name, sid, len(s))
            except Exception as exc:
                logger.warning("  Could not download %s: %s", sid, exc)

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df.to_pickle(cache_file)
        logger.info("Saved FRED data to cache: %s", cache_file)
        return df
