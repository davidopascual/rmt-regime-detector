"""
loader.py — Data acquisition and preprocessing for the RMT research pipeline.

This module downloads adjusted daily closing prices for the S&P 500 universe
via yfinance, computes log-returns, and saves the result to disk in a
reproducible format.

Design decisions
----------------
  - We use log-returns r(t) = log(P(t)/P(t-1)) throughout.  Log-returns are
    approximately additive over time and have lighter tails than simple
    returns for typical daily holding periods.
  - Prices are adjusted closes (splits + dividends) from Yahoo Finance.
  - We do NOT perform any outlier winsorization at this stage; that choice
    is left to downstream analysis with explicit documentation.
  - NaN handling: assets missing more than `max_nan_frac` of observations in
    the study period are dropped from the final panel.  Within-window NaN
    handling is the estimator's responsibility (see estimator.py).

Data source
-----------
  yfinance (Yahoo Finance) — free, no API key required.  Data quality
  limitations: survivorship bias in the ticker list (see universe.py),
  corporate action adjustment errors are possible, and Yahoo Finance data
  is not suitable for live trading.  This is a research prototype only.

Limitations
-----------
  [Mathematical Flag per estimator.py design notes]: the estimator applies
  within-window standardisation, so heterogeneous volatility across the
  study period is removed at the window level.  The loader therefore does
  not need to standardize returns — raw log-returns are passed through.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# yfinance is a soft dependency — imported only inside download_returns()
# so that the rest of the pipeline works without it if data is pre-cached.

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def download_returns(
    tickers: list[str],
    start: str = "2000-01-01",
    end: str = "2023-12-31",
    cache_path: Optional[Path] = None,
    force_download: bool = False,
    max_nan_frac: float = 0.20,
) -> pd.DataFrame:
    """
    Download adjusted closes, compute log-returns, cache to disk.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to download.
    start, end : str
        Date range in YYYY-MM-DD format (inclusive).
    cache_path : Path or None
        Path to the cached Parquet file.  Defaults to
        research/data/returns_panel.parquet.
    force_download : bool
        If True, ignore the cache and re-download.
    max_nan_frac : float
        Drop columns (assets) with fraction of NaN log-returns exceeding
        this threshold over the entire study period.

    Returns
    -------
    pd.DataFrame
        Log-returns panel, shape (T, d).
        Index: pd.DatetimeIndex of trading days.
        Columns: ticker symbols.
        NaN: legitimate missing observations (halted trading, listing gaps).
        All-NaN rows (market holidays) are dropped.
    """
    if cache_path is None:
        cache_path = _DEFAULT_DATA_DIR / "returns_panel.parquet"

    cache_path = Path(cache_path)

    if cache_path.exists() and not force_download:
        returns = pd.read_parquet(cache_path)
        print(f"Loaded cached returns from {cache_path} — shape {returns.shape}")
        return returns

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for data download. "
            "Install it with: pip install yfinance"
        ) from exc

    print(f"Downloading adjusted closes for {len(tickers)} tickers "
          f"({start} to {end})…")

    # Download in one batch — yfinance handles multiple tickers efficiently
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance v0.2+ always returns a MultiIndex when len(tickers) > 1.
    # With auto_adjust=True the adjusted close is stored under "Close"
    # (the "Adj Close" field is absent or redundant; "Close" IS adjusted).
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0).unique().tolist()
        price_key = "Close" if "Close" in level0 else level0[0]
        prices = raw[price_key]
    else:
        prices = raw.to_frame() if isinstance(raw, pd.Series) else raw
        prices.columns = tickers[:1]

    # Ensure columns cover all requested tickers (missing ones become NaN)
    prices = prices.reindex(columns=tickers)

    # Log-returns: r(t) = log(P(t) / P(t-1))
    log_returns = np.log(prices / prices.shift(1))   # first row all NaN

    # Drop the first row (all NaN from shift)
    log_returns = log_returns.iloc[1:]

    # Drop rows that are all NaN (market holidays / weekends)
    log_returns = log_returns.dropna(how="all")

    # Drop assets with too many NaN observations
    nan_fracs = log_returns.isna().mean()
    keep = nan_fracs[nan_fracs <= max_nan_frac].index
    n_dropped = len(tickers) - len(keep)
    if n_dropped > 0:
        warnings.warn(
            f"Dropped {n_dropped} assets with NaN fraction > {max_nan_frac}: "
            f"{list(nan_fracs[nan_fracs > max_nan_frac].index)}",
            UserWarning,
            stacklevel=2,
        )
    log_returns = log_returns[keep]

    # Cache to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    log_returns.to_parquet(cache_path)
    print(f"Saved returns panel to {cache_path} — shape {log_returns.shape}")

    return log_returns


def load_returns(
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load a previously cached returns panel from Parquet.

    Parameters
    ----------
    cache_path : Path or None
        Path to cached Parquet file.  Defaults to
        research/data/returns_panel.parquet.

    Returns
    -------
    pd.DataFrame
        Log-returns panel as saved by download_returns().

    Raises
    ------
    FileNotFoundError
        If no cache exists at the given path.
    """
    if cache_path is None:
        cache_path = _DEFAULT_DATA_DIR / "returns_panel.parquet"

    cache_path = Path(cache_path)

    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cached returns found at {cache_path}. "
            "Run download_returns() first."
        )

    returns = pd.read_parquet(cache_path)
    return returns


def returns_to_numpy(
    returns: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> tuple[np.ndarray, pd.DatetimeIndex, list[str]]:
    """
    Slice and convert a returns DataFrame to a NumPy array for the estimator.

    Parameters
    ----------
    returns : pd.DataFrame
        Log-returns panel (output of download_returns or load_returns).
    start, end : str or None
        Optional date slice in YYYY-MM-DD format.

    Returns
    -------
    array : np.ndarray, shape (T, d)
        Returns matrix.  NaN values preserved for the estimator.
    dates : pd.DatetimeIndex
        Row dates corresponding to array rows.
    tickers : list of str
        Column tickers corresponding to array columns.
    """
    if start is not None or end is not None:
        returns = returns.loc[start:end]

    return returns.to_numpy(dtype=float), returns.index, list(returns.columns)
