"""
changepoint.py — CUSUM-based spectral change-point detection.

This module applies a CUSUM (cumulative sum) control chart to the sequence
of KS distances {KS(t)} produced by RollingSpectralEstimator to detect
structural breaks in the covariance eigenvalue distribution.

Theoretical basis
-----------------
  [Page54] Page, E.S. (1954). Continuous inspection schemes.
           Biometrika, 41(1/2), 100-114.

  Page's original Rule 1 (equation from the paper):
      "Take action at the first n such that
       S_n - min_{0 <= i <= n} S_i > h"
  where S_n = sum_{j=1}^{n} x_j and h is the decision interval.

  The equivalent recursive form commonly used in practice:
      C(t) = max(0, C(t-1) + (x(t) - mu - delta))
  is derived from Page's Rule 1 (equivalence shown by Lorden 1971 and
  Moustakides 1986) but was NOT explicitly stated in Page (1954).

  [Mathematical Flag 3]: The recursive CUSUM formula above is from
  post-1954 literature.  Page's original paper derives the rule in terms
  of min-of-cumulative-sums, not the recursive max(0,...) form.  Both
  are equivalent for detecting an upward shift; attribution of the
  recursive form to "Page 1954" is a common but imprecise citation.

Stationarity assumption
-----------------------
CUSUM assumes the in-control (null) process is stationary with known mean
and variance.  Financial time series violate this assumption:

  1. Volatility clustering (ARCH effects): KS(t) is autocorrelated and
     heteroskedastic.  CUSUM false-alarm rates are not calibrated.
  2. Regime-dependent baseline: the "in-control" KS level differs between
     calm and volatile markets.  A single calibration period mu/sigma
     cannot capture this.
  3. Serial dependence in rolling windows: consecutive windows overlap
     (step < window), so KS(t) values are correlated by construction.

These limitations mean that CUSUM alarm times are indicative rather than
statistically rigorous.  Any published results must document the
calibration period dates and acknowledge the stationarity violation.

Implementation notes
--------------------
  - The CUSUM is applied to {KS(t)} because KS is our primary test
    statistic for departures from the MP bulk distribution.
  - delta (the allowance / shift to detect) is set as a multiple of the
    calibration-period std of KS(t).
  - h (the decision interval) is set as a multiple of the calibration-period
    std of KS(t).  There is no analytically correct choice; the defaults
    here are heuristic starting points.
  - The calibration period should be a period of known market calm
    (e.g., 2004-2006).  Its choice materially affects all downstream alarm
    times and must be documented in any published results.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class CusumResult:
    """
    CUSUM statistics for a single time series.

    Attributes
    ----------
    cusum : np.ndarray
        Cumulative sum statistic C(t) at each step t.
    alarms : np.ndarray of int
        Indices (in the input series) at which C(t) > h (alarm threshold).
    mu : float
        In-control mean used for the CUSUM (from calibration period).
    sigma : float
        In-control std used for the CUSUM (from calibration period).
    delta : float
        Allowance parameter (shift to detect) = k_delta * sigma.
    h : float
        Decision interval (alarm threshold) = k_h * sigma.
    k_delta : float
        Multiplier for delta: delta = k_delta * sigma.
    k_h : float
        Multiplier for h: h = k_h * sigma.
    calibration_slice : slice
        The slice of the input series used for calibration.
    """
    cusum: np.ndarray
    alarms: np.ndarray
    mu: float
    sigma: float
    delta: float
    h: float
    k_delta: float
    k_h: float
    calibration_slice: slice


# ---------------------------------------------------------------------------
# CUSUM implementation
# ---------------------------------------------------------------------------


def cusum_detect(
    series: np.ndarray,
    calibration_n: int = 20,
    k_delta: float = 0.5,
    k_h: float = 4.0,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
) -> CusumResult:
    """
    Apply one-sided upper CUSUM to detect upward shifts in `series`.

    The CUSUM statistic (recursive form):
        C(0) = 0
        C(t) = max(0, C(t-1) + (x(t) - mu - delta))

    An alarm fires when C(t) > h.

    Parameters
    ----------
    series : np.ndarray, shape (n,)
        The time series to monitor.  For this pipeline, typically KS(t)
        from a RollingSpectralEstimator run.
    calibration_n : int
        Number of observations at the start of `series` used to estimate
        the in-control mean and std if mu/sigma are not provided.
        Must be >= 4 to get a meaningful std estimate.
        [Warning: see module docstring on stationarity limitations.]
    k_delta : float
        Allowance multiplier: delta = k_delta * sigma.
        Typical value: 0.5 (detects shift of 1*sigma in expected ARL).
    k_h : float
        Decision interval multiplier: h = k_h * sigma.
        Typical value: 4.0 (heuristic; not calibrated for financial data).
    mu : float or None
        If provided, use this as the in-control mean (skip calibration).
    sigma : float or None
        If provided, use this as the in-control std (skip calibration).

    Returns
    -------
    CusumResult

    Notes
    -----
    See module docstring for the stationarity assumption and its violation
    in financial data.  Alarm times should be interpreted cautiously.
    """
    series = np.asarray(series, dtype=float)
    if series.ndim != 1:
        raise ValueError("series must be 1-D")
    n = len(series)

    if n < 2:
        raise ValueError("series must have at least 2 observations")

    if calibration_n < 4:
        raise ValueError("calibration_n must be >= 4 for a meaningful std estimate")

    if calibration_n > n:
        raise ValueError(
            f"calibration_n={calibration_n} > series length={n}"
        )

    cal_slice = slice(0, calibration_n)

    if mu is None or sigma is None:
        cal_data = series[cal_slice]
        cal_mu = float(np.nanmean(cal_data))
        cal_sigma = float(np.nanstd(cal_data, ddof=1))
        if cal_sigma < 1e-12:
            warnings.warn(
                "Calibration-period std is near zero; CUSUM may be unreliable. "
                "Check that the calibration period has non-constant series values.",
                RuntimeWarning,
                stacklevel=2,
            )
            cal_sigma = 1e-6   # prevent division by zero
        if mu is None:
            mu = cal_mu
        if sigma is None:
            sigma = cal_sigma

    delta = k_delta * sigma
    h = k_h * sigma

    # Recursive CUSUM: upward shift detection
    cusum = np.zeros(n)
    c_prev = 0.0
    for t in range(n):
        x = series[t]
        if np.isnan(x):
            # Propagate NaN-free: treat missing as in-control observation
            c_curr = c_prev
        else:
            c_curr = max(0.0, c_prev + (x - mu - delta))
        cusum[t] = c_curr
        c_prev = c_curr

    alarms = np.where(cusum > h)[0]

    return CusumResult(
        cusum=cusum,
        alarms=alarms,
        mu=float(mu),
        sigma=float(sigma),
        delta=delta,
        h=h,
        k_delta=k_delta,
        k_h=k_h,
        calibration_slice=cal_slice,
    )


def cusum_from_snapshots(
    snapshots: list,
    calibration_n: int = 20,
    k_delta: float = 0.5,
    k_h: float = 4.0,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
) -> tuple[CusumResult, np.ndarray]:
    """
    Extract KS distances from SpectralSnapshot list and run CUSUM.

    Parameters
    ----------
    snapshots : list of SpectralSnapshot
        Output of RollingSpectralEstimator.fit().
    calibration_n, k_delta, k_h, mu, sigma
        Passed through to cusum_detect().

    Returns
    -------
    result : CusumResult
        CUSUM statistics.
    center_positions : np.ndarray of int
        center_pos values from the snapshots (x-axis for plotting).
    """
    if len(snapshots) == 0:
        raise ValueError("snapshots list is empty")

    ks_series = np.array([s.ks for s in snapshots])
    center_positions = np.array([s.center_pos for s in snapshots])

    result = cusum_detect(
        ks_series,
        calibration_n=calibration_n,
        k_delta=k_delta,
        k_h=k_h,
        mu=mu,
        sigma=sigma,
    )

    return result, center_positions


# ---------------------------------------------------------------------------
# Two-regime CUSUM on ρ(t) with empirically calibrated threshold
# ---------------------------------------------------------------------------


def _cusum_resettable(
    series: np.ndarray,
    mu: float,
    delta: float,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-sided upper CUSUM that resets to zero after each alarm.

    Proper ARL calibration requires a resettable CUSUM (the non-resettable
    version from cusum_detect accumulates indefinitely, making the alarm
    count grow monotonically for non-stationary series).

    C(0) = 0
    C(t) = max(0, C(t-1) + (x(t) − μ − δ))
    Alarm at t if C(t) > h; then C(t) ← 0 before proceeding.

    Returns
    -------
    cusum : np.ndarray
        CUSUM values (after reset, value is 0.0 at alarm steps).
    alarms : np.ndarray of int
        Indices at which alarms fired.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    cusum = np.zeros(n)
    alarms = []
    c = 0.0
    for t in range(n):
        x = series[t]
        c = max(0.0, c + (x - mu - delta)) if not np.isnan(x) else c
        if c > h:
            alarms.append(t)
            c = 0.0          # reset after alarm
        cusum[t] = c
    return cusum, np.array(alarms, dtype=int)


def _calibrate_h_for_arl(
    cal_series: np.ndarray,
    mu: float,
    delta: float,
    target_arl_windows: int = 24,
) -> float:
    """
    Binary search for h giving approximately 1 alarm per target_arl_windows
    on the calibration series (using the resettable CUSUM).

    'target_arl_windows' = 24 corresponds to 1 false alarm per 2 years
    at step=21 (24 windows × 21 days ≈ 504 trading days ≈ 2 years).

    The target number of alarms is n_cal / target_arl_windows.  We find the
    threshold h such that the resettable CUSUM produces approximately this
    many alarms during the calibration period.  Because the calibration
    series is finite and non-Gaussian, this is an empirical rather than
    analytical calibration.
    """
    n_cal = len(cal_series)
    target_n = n_cal / target_arl_windows   # e.g., 3.5 alarms if cal is 7 years

    # Upper bound on h: max possible non-resetting CUSUM value
    increments = np.nan_to_num(np.asarray(cal_series) - mu - delta, nan=0.0)
    h_hi = float(np.max(np.maximum.accumulate(np.maximum(0.0, np.cumsum(increments))))) * 1.5 + 1e-6
    h_lo = 0.0

    for _ in range(60):
        h_mid = (h_lo + h_hi) / 2
        _, alarms = _cusum_resettable(cal_series, mu, delta, h_mid)
        if len(alarms) > target_n:
            h_lo = h_mid   # too many alarms → raise h
        else:
            h_hi = h_mid   # too few alarms  → lower h

    return (h_lo + h_hi) / 2


def _calibrate_h_percentile(
    cal_series: np.ndarray,
    mu: float,
    delta: float,
    percentile: float = 95.0,
) -> float:
    """
    Distribution-free threshold calibration: h = percentile of CUSUM values
    on the calibration series (non-resettable version, so values accumulate).

    Because the CUSUM is non-resettable here, the percentile captures the
    tail of the statistic over the entire calibration period rather than
    over independent renewal intervals.  This is deliberately conservative
    for a series that trends upward.

    A high percentile (e.g., 95th) means that only the top 5% of the
    calibration CUSUM mass triggers an alarm in the run period, providing
    a distribution-free false-alarm control.

    Parameters
    ----------
    cal_series : np.ndarray
        Calibration-period values of the statistic (e.g., ρ(t)).
    mu : float
        In-control mean (estimated from cal_series mean).
    delta : float
        Allowance = k_delta * sigma.
    percentile : float
        Percentile of the CUSUM distribution to use as threshold (0–100).

    Returns
    -------
    float : calibrated h
    """
    series = np.asarray(cal_series, dtype=float)
    n = len(series)
    cusum = np.zeros(n)
    c = 0.0
    for t in range(n):
        x = series[t]
        c = max(0.0, c + (x - mu - delta)) if not np.isnan(x) else c
        cusum[t] = c
    return float(np.percentile(cusum, percentile))


def two_regime_cusum_alternatives(
    snapshots: list,
    dates,
    k_delta: float = 0.5,
    percentile: float = 95.0,
) -> dict:
    """
    Run two alternative calibration approaches for the Regime 2 (COVID) CUSUM.

    The standard ARL calibration failed for 2013-2019 because the calibration
    period is too volatile in ρ-space, causing h to collapse to zero.  This
    function tries two alternative approaches:

    Approach A — Percentile-based threshold (2013-2019 calibration)
        h = 95th percentile of the non-resettable CUSUM statistic computed
        on the 2013-2019 calibration series.  Distribution-free; does not
        require stationarity.  Robust to slowly drifting baselines.

    Approach B — Shorter calibration window (2016-2019)
        Use 2016-2019 only (the quietest sub-period visible in Figure 2) as
        the calibration period for the ARL-based h search.  If ρ(t) is more
        stationary in this sub-period, the ARL calibration may stabilize.

    Both approaches use the same run period (2013-2024) and crisis target
    (COVID, 2020-02-20).

    Parameters
    ----------
    snapshots : list of SpectralSnapshot
    dates : pd.DatetimeIndex
    k_delta : float
        Allowance multiplier.
    percentile : float
        Percentile for Approach A.

    Returns
    -------
    dict with keys:
        'approach_a' : dict  — Percentile-based (cal 2013-2019)
        'approach_b' : dict  — ARL-based (cal 2016-2019)

    Each inner dict has the same keys as one element of two_regime_cusum().
    """
    snap_dates_all = pd.DatetimeIndex([dates[s.center_pos] for s in snapshots])
    rho_all = np.array([s.rho for s in snapshots])

    run_s   = pd.Timestamp("2013-01-01")
    run_e   = pd.Timestamp("2024-12-31")
    cr_s    = pd.Timestamp("2020-02-20")
    cr_e    = pd.Timestamp("2020-03-23")

    run_mask       = (snap_dates_all >= run_s) & (snap_dates_all <= run_e)
    snap_dates_run = snap_dates_all[run_mask]
    rho_run        = rho_all[run_mask]

    def _alarm_summary(cusum_run, alarms_run):
        alarm_dates = [snap_dates_run[i].date() for i in alarms_run]
        first_alarm_after = None
        detection_delay   = None
        for i in alarms_run:
            ad = snap_dates_run[i]
            if ad >= cr_s:
                first_alarm_after = ad.date()
                detection_delay   = int((ad - cr_s).days)
                break
        false_alarm_dates = [
            snap_dates_run[i].date()
            for i in alarms_run
            if snap_dates_run[i] < cr_s
        ]
        return alarm_dates, first_alarm_after, detection_delay, false_alarm_dates

    # ── Approach A: percentile threshold, calibration 2013-2019 ──────────────
    cal_s_a = pd.Timestamp("2013-01-01")
    cal_e_a = pd.Timestamp("2019-01-01")
    cal_mask_a = (snap_dates_all >= cal_s_a) & (snap_dates_all <= cal_e_a)
    rho_cal_a  = rho_all[cal_mask_a]
    mu_a       = float(np.nanmean(rho_cal_a))
    sigma_a    = float(np.nanstd(rho_cal_a, ddof=1))
    if sigma_a < 1e-6:
        sigma_a = 1e-6
    delta_a    = k_delta * sigma_a
    h_a        = _calibrate_h_percentile(rho_cal_a, mu_a, delta_a, percentile)

    cusum_a, alarms_a = _cusum_resettable(rho_run, mu_a, delta_a, h_a)
    ad_a, fa_a, dd_a, false_a = _alarm_summary(cusum_a, alarms_a)

    approach_a = {
        "name":                     "Approach A (percentile, cal 2013-2019)",
        "cal_start":                "2013-01-01",
        "cal_end":                  "2019-01-01",
        "mu":                       mu_a,
        "sigma":                    sigma_a,
        "delta":                    delta_a,
        "h":                        h_a,
        "h_method":                 f"{percentile}th percentile of calibration CUSUM",
        "snap_dates_run":           snap_dates_run,
        "rho_run":                  rho_run,
        "cusum_run":                cusum_a,
        "alarms_run":               alarms_a,
        "alarm_dates":              ad_a,
        "crisis_label":             "COVID",
        "crisis_start":             cr_s,
        "crisis_end":               cr_e,
        "first_alarm_after_crisis": fa_a,
        "detection_delay_days":     dd_a,
        "false_alarm_count":        len(false_a),
        "false_alarm_dates":        false_a,
    }

    # ── Approach B: ARL calibration, shorter window 2016-2019 ────────────────
    cal_s_b = pd.Timestamp("2016-01-01")
    cal_e_b = pd.Timestamp("2019-01-01")
    cal_mask_b = (snap_dates_all >= cal_s_b) & (snap_dates_all <= cal_e_b)
    rho_cal_b  = rho_all[cal_mask_b]
    mu_b       = float(np.nanmean(rho_cal_b))
    sigma_b    = float(np.nanstd(rho_cal_b, ddof=1))
    if sigma_b < 1e-6:
        sigma_b = 1e-6
    delta_b    = k_delta * sigma_b
    h_b        = _calibrate_h_for_arl(rho_cal_b, mu_b, delta_b, target_arl_windows=24)

    cusum_b, alarms_b = _cusum_resettable(rho_run, mu_b, delta_b, h_b)
    ad_b, fa_b, dd_b, false_b = _alarm_summary(cusum_b, alarms_b)

    approach_b = {
        "name":                     "Approach B (ARL, cal 2016-2019)",
        "cal_start":                "2016-01-01",
        "cal_end":                  "2019-01-01",
        "mu":                       mu_b,
        "sigma":                    sigma_b,
        "delta":                    delta_b,
        "h":                        h_b,
        "h_method":                 "ARL binary search (target ~1 alarm/2yr)",
        "snap_dates_run":           snap_dates_run,
        "rho_run":                  rho_run,
        "cusum_run":                cusum_b,
        "alarms_run":               alarms_b,
        "alarm_dates":              ad_b,
        "crisis_label":             "COVID",
        "crisis_start":             cr_s,
        "crisis_end":               cr_e,
        "first_alarm_after_crisis": fa_b,
        "detection_delay_days":     dd_b,
        "false_alarm_count":        len(false_b),
        "false_alarm_dates":        false_b,
    }

    return {"approach_a": approach_a, "approach_b": approach_b}


def two_regime_cusum(
    snapshots: list,
    dates,
    regime_specs: Optional[list] = None,
    k_delta: float = 0.5,
    target_arl_windows: int = 24,
) -> list[dict]:
    """
    Run two-regime CUSUM on ρ(t) = λ₁(t)/λ+(t), each regime with its own
    calibration period and empirically calibrated alarm threshold h.

    The threshold h is calibrated to give approximately one false alarm per
    two years (target_arl_windows = 24 monthly-step windows) during the
    in-control (calibration) period.

    Parameters
    ----------
    snapshots : list of SpectralSnapshot
    dates : pd.DatetimeIndex
        Row dates of the original returns array.
    regime_specs : list of dict or None
        Each dict has:
          'name'         : str label
          'cal_start'    : str  (YYYY-MM-DD) calibration start
          'cal_end'      : str  calibration end
          'run_start'    : str  CUSUM monitoring start
          'run_end'      : str  monitoring end
          'crisis_label' : str  label for the target crisis
          'crisis_start' : str  expected detection event start
          'crisis_end'   : str  expected detection event end
        Defaults to the two canonical regimes (pre-GFC / post-GFC).
    k_delta : float
        Allowance = k_delta * sigma_rho.  0.5 = detect a 0.5σ upward shift.
    target_arl_windows : int
        Target ARL for threshold calibration.  24 ≈ 1 per 2 years at step=21.

    Returns
    -------
    list of dict, one per regime, with keys:
        name, cal_start, cal_end, run_start, run_end,
        mu, sigma, delta, h, target_arl_windows,
        snap_dates_run  : pd.DatetimeIndex (dates of run-period windows)
        rho_run         : np.ndarray       (ρ(t) in run period)
        cusum_run       : np.ndarray       (CUSUM values in run period)
        alarms_run      : np.ndarray       (alarm indices in run-period array)
        alarm_dates     : list of date     (calendar dates of alarms)
        crisis_label    : str
        crisis_start    : pd.Timestamp
        crisis_end      : pd.Timestamp
        first_alarm_after_crisis : date or None
        detection_delay_days     : int or None  (positive = lag, negative = early)
        false_alarm_count        : int
        false_alarm_dates        : list of date
    """
    if regime_specs is None:
        regime_specs = [
            {
                "name":         "Regime 1 (pre-GFC)",
                "cal_start":    "2004-01-01",
                "cal_end":      "2007-06-01",
                "run_start":    "2004-01-01",
                "run_end":      "2009-12-31",
                "crisis_label": "GFC",
                "crisis_start": "2007-10-09",
                "crisis_end":   "2009-03-09",
            },
            {
                "name":         "Regime 2 (post-GFC calm)",
                "cal_start":    "2013-01-01",
                "cal_end":      "2019-01-01",
                "run_start":    "2013-01-01",
                "run_end":      "2024-12-31",
                "crisis_label": "COVID",
                "crisis_start": "2020-02-20",
                "crisis_end":   "2020-03-23",
            },
        ]

    snap_dates_all = pd.DatetimeIndex([dates[s.center_pos] for s in snapshots])
    rho_all        = np.array([s.rho for s in snapshots])

    results = []

    for spec in regime_specs:
        cal_s  = pd.Timestamp(spec["cal_start"])
        cal_e  = pd.Timestamp(spec["cal_end"])
        run_s  = pd.Timestamp(spec["run_start"])
        run_e  = pd.Timestamp(spec["run_end"])
        cr_s   = pd.Timestamp(spec["crisis_start"])
        cr_e   = pd.Timestamp(spec["crisis_end"])

        # Calibration period
        cal_mask = (snap_dates_all >= cal_s) & (snap_dates_all <= cal_e)
        if cal_mask.sum() < 4:
            warnings.warn(
                f"{spec['name']}: fewer than 4 calibration windows. "
                f"Check cal_start/cal_end.",
                RuntimeWarning,
            )
        rho_cal = rho_all[cal_mask]
        mu      = float(np.nanmean(rho_cal))
        sigma   = float(np.nanstd(rho_cal, ddof=1))
        if sigma < 1e-6:
            sigma = 1e-6
        delta_v = k_delta * sigma

        # Calibrate h empirically
        h = _calibrate_h_for_arl(rho_cal, mu, delta_v, target_arl_windows)

        # Run period
        run_mask = (snap_dates_all >= run_s) & (snap_dates_all <= run_e)
        snap_dates_run = snap_dates_all[run_mask]
        rho_run        = rho_all[run_mask]

        cusum_run, alarms_run = _cusum_resettable(rho_run, mu, delta_v, h)

        # Alarm dates
        alarm_dates = [snap_dates_run[i].date() for i in alarms_run]

        # First alarm after crisis start
        first_alarm_after = None
        detection_delay   = None
        for i in alarms_run:
            ad = snap_dates_run[i]
            if ad >= cr_s:
                first_alarm_after = ad.date()
                detection_delay   = int((ad - cr_s).days)
                break

        # False alarms: alarms that fall entirely outside the crisis window
        false_alarm_indices = [
            i for i in alarms_run
            if not (snap_dates_run[i] >= cr_s and snap_dates_run[i] <= cr_e)
        ]
        # Also exclude any alarms during OTHER known crisis episodes in run period
        # (we count a false alarm only if it's in a truly calm period)
        # For simplicity: alarm is "false" if it occurs before the crisis_start
        # OR after the crisis_end and not during another labeled crisis
        false_alarm_dates = [
            snap_dates_run[i].date()
            for i in alarms_run
            if snap_dates_run[i] < cr_s
        ]

        results.append({
            "name":                       spec["name"],
            "cal_start":                  spec["cal_start"],
            "cal_end":                    spec["cal_end"],
            "run_start":                  spec["run_start"],
            "run_end":                    spec["run_end"],
            "mu":                         mu,
            "sigma":                      sigma,
            "delta":                      delta_v,
            "h":                          h,
            "target_arl_windows":         target_arl_windows,
            "snap_dates_run":             snap_dates_run,
            "rho_run":                    rho_run,
            "cusum_run":                  cusum_run,
            "alarms_run":                 alarms_run,
            "alarm_dates":                alarm_dates,
            "crisis_label":               spec["crisis_label"],
            "crisis_start":               cr_s,
            "crisis_end":                 cr_e,
            "first_alarm_after_crisis":   first_alarm_after,
            "detection_delay_days":       detection_delay,
            "false_alarm_count":          len(false_alarm_dates),
            "false_alarm_dates":          false_alarm_dates,
        })

    return results
