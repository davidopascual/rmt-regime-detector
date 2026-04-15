"""
universe.py — S&P 500 constituent universe for the RMT research pipeline.

This module defines the historical S&P 500 constituent universe used
throughout the research.  We use a fixed 100-ticker subset drawn from
long-tenured S&P 500 members to avoid survivor-bias artefacts from the
full dynamic index.

Design decisions
----------------
We use a static 100-ticker list rather than a dynamic constituent list for
three reasons:

  1. Survivorship bias: including only stocks present for the ENTIRE study
     period (2000–2023) biases toward historically successful companies.
     The current list partially mitigates this but does not eliminate it.
     Any published results must acknowledge this limitation.

  2. Missing data handling: the estimator already drops assets with NaN
     in each rolling window (see estimator.py), so assets listed here that
     are temporarily missing are handled automatically.

  3. Reproducibility: a fixed list ensures identical results across runs
     without requiring a live Bloomberg/Refinitiv connection.

Limitations
-----------
  - The 100 tickers below are representative long-tenured S&P 500 members
    but the list is not exhaustive and excludes many periods of index
    membership.  Do NOT interpret results as applying to the full S&P 500.
  - Tickers that were renamed, delisted, or merged may have data gaps that
    are silently dropped per-window by the estimator.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 100-ticker static universe
# ---------------------------------------------------------------------------
# Drawn from sectors: Technology, Financials, Healthcare, Consumer Staples,
# Energy, Industrials, Consumer Discretionary, Materials, Utilities, Comm Svcs.
# All were S&P 500 members for substantial portions of 2000–2023.

SP100_TICKERS: list[str] = [
    # Technology (18)
    # XLNX removed: acquired by AMD Feb 2022 — replaced with MRVL (Marvell Tech)
    "AAPL", "MSFT", "INTC", "IBM", "ORCL", "TXN", "QCOM", "HPQ", "ADI",
    "AMAT", "MRVL", "KLAC", "LRCX", "MU", "CSCO", "NVDA", "GLW", "STX",
    # Financials (16)
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "USB", "PNC", "TFC",
    "BK", "STT", "COF", "AIG", "MET", "PRU",
    # Healthcare (14)
    # ANTM renamed to ELV (Elevance Health) Jun 2022 — use ELV for full history
    "JNJ", "PFE", "ABT", "MRK", "LLY", "BMY", "AMGN", "MDT", "UNH",
    "CVS", "HUM", "ELV", "BAX", "BDX",
    # Consumer Staples (10)
    # K (Kellogg's/Kellanova) had timezone issues at download — replaced with KHC
    "PG", "KO", "PEP", "WMT", "COST", "CL", "GIS", "KHC", "HRL", "SYY",
    # Energy (10)
    "XOM", "CVX", "COP", "SLB", "HAL", "OXY", "VLO", "PSX", "MPC", "DVN",
    # Industrials (10)
    "GE", "HON", "MMM", "CAT", "DE", "BA", "EMR", "ETN", "ITW", "ROK",
    # Consumer Discretionary (8)
    "HD", "LOW", "TGT", "MCD", "YUM", "NKE", "F", "GM",
    # Communication Services (6)
    "T", "VZ", "CMCSA", "DIS", "NFLX", "CHTR",
    # Materials (5)
    "DD", "DOW", "LYB", "APD", "NEM",
    # Utilities (3)
    "NEE", "SO", "DUK",
]

assert len(SP100_TICKERS) == 100, f"Expected 100 tickers, got {len(SP100_TICKERS)}"
assert len(set(SP100_TICKERS)) == 100, "Duplicate tickers in SP100_TICKERS"


def get_universe(sector: str | None = None) -> list[str]:
    """
    Return the universe ticker list, optionally filtered to a sector.

    Parameters
    ----------
    sector : str or None
        If None, return all 100 tickers.  If provided, must be one of:
        'technology', 'financials', 'healthcare', 'consumer_staples',
        'energy', 'industrials', 'consumer_discretionary',
        'communication_services', 'materials', 'utilities'.

    Returns
    -------
    list of str
        Ticker symbols in the selected universe.
    """
    if sector is None:
        return list(SP100_TICKERS)

    _SECTOR_MAP: dict[str, list[str]] = {
        "technology": SP100_TICKERS[0:18],
        "financials": SP100_TICKERS[18:34],
        "healthcare": SP100_TICKERS[34:48],
        "consumer_staples": SP100_TICKERS[48:58],
        "energy": SP100_TICKERS[58:68],
        "industrials": SP100_TICKERS[68:78],
        "consumer_discretionary": SP100_TICKERS[78:86],
        "communication_services": SP100_TICKERS[86:92],
        "materials": SP100_TICKERS[92:97],
        "utilities": SP100_TICKERS[97:100],
    }

    sector_lower = sector.lower()
    if sector_lower not in _SECTOR_MAP:
        raise ValueError(
            f"Unknown sector {sector!r}. "
            f"Valid sectors: {sorted(_SECTOR_MAP.keys())}"
        )
    return list(_SECTOR_MAP[sector_lower])
