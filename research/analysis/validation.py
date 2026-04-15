"""
validation.py — Crisis event study and ROC-style validation.

This module provides tools to evaluate whether spectral statistics
(KS distance, number of spikes k, detachment ratio rho) from the
RollingSpectralEstimator are informative about known financial crises.

Validation approach
-------------------
We do NOT test whether the estimator predicts crises (it cannot: the
estimator uses only past returns in each window).  Instead we test:

  1. Event study: are KS(t), k(t), rho(t) elevated DURING crisis windows
     compared to calm periods?  This is an in-sample descriptive test.

  2. ROC analysis: for each threshold h on a spectral statistic S(t),
     compute the true positive rate (TPR) and false positive rate (FPR)
     of the binary classifier S(t) > h for predicting membership in a
     pre-defined "crisis calendar".  The AUROC summarizes overall
     discriminative power.

Limitations
-----------
  - The crisis calendar below is constructed by the researcher, which
    introduces look-ahead bias: we choose the windows to label as
    "crisis" AFTER observing that the spectral statistics are elevated.
    Any reported AUROC should be treated as descriptive, not predictive.
  - Overlapping rolling windows create serial correlation in the test
    statistics, invalidating standard ROC confidence intervals.
  - With only ~6 distinct crisis episodes in 2000-2023, statistical
    power is low regardless of AUROC.

Crisis calendar
---------------
Crisis windows are defined by the first and last dates of widely
recognized U.S. equity market stress episodes:

  Episode                        Start        End
  -------------------------------------------------
  Dot-com bust                   2000-03-10   2002-10-09
  Post-9/11 market shock         2001-09-11   2001-09-21
  Global Financial Crisis        2007-10-09   2009-03-09
  European Sovereign Debt        2010-04-23   2010-07-02
  Flash Crash                    2010-05-06   2010-05-06
  China slowdown / oil crash     2015-08-18   2016-02-11
  COVID-19 crash                 2020-02-20   2020-03-23

Sources: NBER business cycle dates, CBOE VIX historical levels,
Wikipedia market correction timelines.  These dates are approximate;
small changes do not materially affect the event study conclusions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Crisis calendar
# ---------------------------------------------------------------------------

CRISIS_EPISODES: list[dict[str, str]] = [
    {"name": "Dot-com bust",            "start": "2000-03-10", "end": "2002-10-09"},
    {"name": "Post-9/11 shock",         "start": "2001-09-11", "end": "2001-09-21"},
    {"name": "Global Financial Crisis", "start": "2007-10-09", "end": "2009-03-09"},
    {"name": "Euro Sovereign Debt",     "start": "2010-04-23", "end": "2010-07-02"},
    {"name": "Flash Crash",             "start": "2010-05-06", "end": "2010-05-06"},
    {"name": "China / Oil crash",       "start": "2015-08-18", "end": "2016-02-11"},
    {"name": "COVID-19 crash",          "start": "2020-02-20", "end": "2020-03-23"},
]


def make_crisis_labels(
    dates: pd.DatetimeIndex,
    episodes: list[dict[str, str]] | None = None,
    expand_days: int = 0,
) -> np.ndarray:
    """
    Create binary crisis labels for each date in `dates`.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Trading day dates to label (typically center_pos dates from snapshots).
    episodes : list of dict or None
        Crisis episodes.  Each dict has keys 'name', 'start', 'end'.
        Defaults to CRISIS_EPISODES.
    expand_days : int
        Expand each episode window by this many calendar days on each side.
        Useful for capturing lagged spectral responses.

    Returns
    -------
    labels : np.ndarray of int, shape (len(dates),)
        1 if the date falls within any crisis episode, 0 otherwise.
    """
    if episodes is None:
        episodes = CRISIS_EPISODES

    labels = np.zeros(len(dates), dtype=int)
    expansion = pd.Timedelta(days=expand_days)

    for ep in episodes:
        start = pd.Timestamp(ep["start"]) - expansion
        end = pd.Timestamp(ep["end"]) + expansion
        mask = (dates >= start) & (dates <= end)
        labels[mask] = 1

    return labels


# ---------------------------------------------------------------------------
# Event study
# ---------------------------------------------------------------------------


@dataclass
class EventStudyResult:
    """
    Summary statistics comparing a spectral statistic during and outside crises.

    Attributes
    ----------
    stat_name : str
        Name of the spectral statistic (e.g., 'ks', 'k', 'rho').
    mean_crisis : float
        Mean value of the statistic during crisis windows.
    mean_calm : float
        Mean value of the statistic during calm (non-crisis) windows.
    ratio : float
        mean_crisis / mean_calm.
    n_crisis : int
        Number of crisis windows.
    n_calm : int
        Number of calm windows.
    p_value : float
        One-sided Mann-Whitney U test p-value (H1: crisis > calm).
        Use with caution: see module docstring on serial correlation.
    """
    stat_name: str
    mean_crisis: float
    mean_calm: float
    ratio: float
    n_crisis: int
    n_calm: int
    p_value: float


def crisis_event_study(
    values: np.ndarray,
    labels: np.ndarray,
    stat_name: str = "statistic",
) -> EventStudyResult:
    """
    Compare a spectral statistic between crisis and calm windows.

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        Spectral statistic time series (e.g., ks, k, rho).
    labels : np.ndarray of int, shape (n,)
        Binary crisis labels (1 = crisis, 0 = calm).
    stat_name : str
        Name for display.

    Returns
    -------
    EventStudyResult
    """
    from scipy import stats

    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if values.shape != labels.shape:
        raise ValueError("values and labels must have the same shape")

    crisis_vals = values[labels == 1]
    calm_vals = values[labels == 0]

    if len(crisis_vals) == 0:
        warnings.warn("No crisis windows found; check crisis_labels.", RuntimeWarning, stacklevel=2)
        p_val = float("nan")
        mu_crisis = float("nan")
    else:
        mu_crisis = float(np.nanmean(crisis_vals))

    if len(calm_vals) == 0:
        warnings.warn("No calm windows found.", RuntimeWarning, stacklevel=2)
        p_val = float("nan")
        mu_calm = float("nan")
    else:
        mu_calm = float(np.nanmean(calm_vals))

    if len(crisis_vals) > 0 and len(calm_vals) > 0:
        # One-sided Mann-Whitney U: H1 crisis > calm
        _, p_val = stats.mannwhitneyu(crisis_vals, calm_vals, alternative="greater")
        p_val = float(p_val)
    else:
        p_val = float("nan")

    ratio = (mu_crisis / mu_calm) if (mu_calm != 0 and not np.isnan(mu_calm)) else float("nan")

    return EventStudyResult(
        stat_name=stat_name,
        mean_crisis=mu_crisis,
        mean_calm=mu_calm,
        ratio=ratio,
        n_crisis=int(np.sum(labels == 1)),
        n_calm=int(np.sum(labels == 0)),
        p_value=p_val,
    )


# ---------------------------------------------------------------------------
# ROC analysis
# ---------------------------------------------------------------------------


@dataclass
class RocResult:
    """
    ROC curve and AUROC for a binary classifier based on a spectral statistic.

    Attributes
    ----------
    stat_name : str
    tpr : np.ndarray
        True positive rates at each threshold.
    fpr : np.ndarray
        False positive rates at each threshold.
    thresholds : np.ndarray
        Threshold values corresponding to each (tpr, fpr) point.
    auroc : float
        Area under the ROC curve (trapezoidal rule).
    """
    stat_name: str
    tpr: np.ndarray
    fpr: np.ndarray
    thresholds: np.ndarray
    auroc: float


def compute_roc(
    values: np.ndarray,
    labels: np.ndarray,
    stat_name: str = "statistic",
) -> RocResult:
    """
    Compute ROC curve for predicting crisis membership from a spectral statistic.

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        Spectral statistic (higher = more anomalous).
    labels : np.ndarray of int, shape (n,)
        Binary crisis labels (1 = positive/crisis, 0 = negative/calm).
    stat_name : str
        Name for display.

    Returns
    -------
    RocResult

    Notes
    -----
    AUROC close to 1 means the statistic tends to be high during crises.
    AUROC = 0.5 is random.  See module docstring for limitations.
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)

    valid = ~np.isnan(values)
    v = values[valid]
    l = labels[valid]

    if len(np.unique(l)) < 2:
        warnings.warn(
            "Only one class present in labels; AUROC undefined. "
            "Check that the date range contains both crisis and calm windows.",
            RuntimeWarning,
            stacklevel=2,
        )
        return RocResult(
            stat_name=stat_name,
            tpr=np.array([0.0, 1.0]),
            fpr=np.array([0.0, 1.0]),
            thresholds=np.array([np.inf, -np.inf]),
            auroc=float("nan"),
        )

    fpr, tpr, thresholds = roc_curve(l, v, pos_label=1)
    auroc = float(roc_auc_score(l, v))

    return RocResult(
        stat_name=stat_name,
        tpr=tpr,
        fpr=fpr,
        thresholds=thresholds,
        auroc=auroc,
    )


def run_full_validation(
    snapshots: list,
    dates: Optional[pd.DatetimeIndex] = None,
    episodes: list[dict[str, str]] | None = None,
    expand_days: int = 0,
) -> dict:
    """
    Run event study and ROC for all primary spectral statistics.

    Parameters
    ----------
    snapshots : list of SpectralSnapshot
        Output of RollingSpectralEstimator.fit().
    dates : pd.DatetimeIndex or None
        Trading day dates for each snapshot (indexed by center_pos).
        If None, spectral timestamps are used directly (less interpretable).
    episodes : list of dict or None
        Crisis episodes.  Defaults to CRISIS_EPISODES.
    expand_days : int
        Passed to make_crisis_labels().

    Returns
    -------
    dict with keys:
        'labels'      : np.ndarray of crisis labels
        'event_study' : dict of stat_name -> EventStudyResult
        'roc'         : dict of stat_name -> RocResult
    """
    if len(snapshots) == 0:
        raise ValueError("snapshots is empty")

    # Extract arrays
    ks_vals  = np.array([s.ks    for s in snapshots])
    k_vals   = np.array([s.k     for s in snapshots], dtype=float)
    rho_vals = np.array([s.rho   for s in snapshots])

    if dates is not None:
        snap_dates = dates[[s.center_pos for s in snapshots]]
    else:
        snap_dates = pd.DatetimeIndex(
            [pd.Timestamp("2000-01-01") + pd.Timedelta(days=s.center_pos)
             for s in snapshots]
        )
        warnings.warn(
            "No dates provided; using synthetic dates from center_pos. "
            "Crisis labeling will be incorrect.",
            UserWarning,
            stacklevel=2,
        )

    labels = make_crisis_labels(snap_dates, episodes=episodes, expand_days=expand_days)

    stats = {
        "ks":  ks_vals,
        "k":   k_vals,
        "rho": rho_vals,
    }

    event_study = {}
    roc = {}

    for name, vals in stats.items():
        event_study[name] = crisis_event_study(vals, labels, stat_name=name)
        roc[name] = compute_roc(vals, labels, stat_name=name)

    return {
        "labels":      labels,
        "event_study": event_study,
        "roc":         roc,
    }


# ---------------------------------------------------------------------------
# ρ(t) decomposition: Effect A (λ₁ rising) vs Effect B (λ+ falling)
# ---------------------------------------------------------------------------


def decompose_rho(
    snapshots: list,
    dates: pd.DatetimeIndex,
    calm_start: str = "2004-01-01",
    calm_end: str = "2007-06-01",
) -> dict:
    """
    Decompose log(ρ(t)) elevation above its calm-period baseline into two effects.

    ρ(t) = λ₁(t) / λ+(t) conflates two simultaneous movements:
      Effect A: λ₁(t) rising above its calm-period mean (market factor strengthening)
      Effect B: λ+(t) falling below its calm-period mean (noise floor collapsing)

    Decomposition:
      log(ρ(t)) = log(λ₁(t)) − log(λ+(t))

    Baseline (calm-period mean):
      log(ρ₀) = log(μ_λ₁) − log(μ_λ+)

    Elevation above baseline:
      log(ρ(t)) − log(ρ₀) = Effect A(t) + Effect B(t)

      where:
        Effect A(t) = log(λ₁(t) / μ_λ₁)    [> 0 when λ₁ above its calm mean]
        Effect B(t) = −log(λ+(t) / μ_λ+)   [> 0 when λ+ below its calm mean]

    Both effects are zero by construction when evaluated at the calm baseline.
    During crises both are typically positive and additive.

    Parameters
    ----------
    snapshots : list of SpectralSnapshot
    dates : pd.DatetimeIndex
        Row dates of the original returns array.
    calm_start, calm_end : str
        Date range (YYYY-MM-DD) of the pre-crisis calibration period.
        Default: 2004-01-01 to 2007-06-01 (pre-GFC calm).

    Returns
    -------
    dict with keys:
        snap_dates       : pd.DatetimeIndex — one date per snapshot
        lambda1          : np.ndarray
        lambda_plus      : np.ndarray
        log_rho          : np.ndarray — log(ρ(t))
        log_rho_baseline : float      — log(ρ) at calm-period means
        effect_a         : np.ndarray — log(λ₁/μ_λ₁)
        effect_b         : np.ndarray — −log(λ+/μ_λ+)
        elevation        : np.ndarray — effect_a + effect_b = log(ρ) − log(ρ₀)
        lambda1_norm     : np.ndarray — λ₁(t) / μ_λ₁
        lp_norm          : np.ndarray — λ+(t) / μ_λ+
        mean_lambda1_calm: float
        mean_lp_calm     : float
        calm_start       : str
        calm_end         : str
    """
    snap_dates = pd.DatetimeIndex([dates[s.center_pos] for s in snapshots])
    lambda1_arr = np.array([s.lambda1     for s in snapshots])
    lp_arr      = np.array([s.lambda_plus for s in snapshots])

    # Calm period mask
    calm_mask = (snap_dates >= pd.Timestamp(calm_start)) & \
                (snap_dates <= pd.Timestamp(calm_end))

    if calm_mask.sum() < 4:
        raise ValueError(
            f"Fewer than 4 snapshots in calm period {calm_start}–{calm_end}. "
            f"Found {calm_mask.sum()}. Check dates."
        )

    mean_lambda1 = float(np.mean(lambda1_arr[calm_mask]))
    mean_lp      = float(np.mean(lp_arr[calm_mask]))

    lambda1_norm = lambda1_arr / mean_lambda1
    lp_norm      = lp_arr      / mean_lp

    effect_a  = np.log(lambda1_norm)           # positive when λ₁ above baseline
    effect_b  = -np.log(lp_norm)               # positive when λ+ below baseline
    elevation = effect_a + effect_b            # = log(ρ(t)) - log(ρ₀)

    log_rho           = np.log(lambda1_arr / lp_arr)
    log_rho_baseline  = float(np.log(mean_lambda1 / mean_lp))

    return {
        "snap_dates":        snap_dates,
        "lambda1":           lambda1_arr,
        "lambda_plus":       lp_arr,
        "log_rho":           log_rho,
        "log_rho_baseline":  log_rho_baseline,
        "effect_a":          effect_a,
        "effect_b":          effect_b,
        "elevation":         elevation,
        "lambda1_norm":      lambda1_norm,
        "lp_norm":           lp_norm,
        "mean_lambda1_calm": mean_lambda1,
        "mean_lp_calm":      mean_lp,
        "calm_start":        calm_start,
        "calm_end":          calm_end,
    }


def decompose_rho_at_crises(
    decomp: dict,
    episodes: Optional[list] = None,
) -> list[dict]:
    """
    Evaluate the ρ decomposition at the peak-ρ window within each crisis episode.

    For each crisis episode that falls within the data range, find the snapshot
    window with the highest ρ elevation and report the Effect A / Effect B split.

    Parameters
    ----------
    decomp : dict
        Output of decompose_rho().
    episodes : list of dict or None
        Crisis episodes (start/end/name).  Defaults to CRISIS_EPISODES.

    Returns
    -------
    list of dict, one per in-range episode, with keys:
        name, start, end, peak_date, log_rho, log_rho_baseline, elevation,
        effect_a, effect_b, frac_a, frac_b, lambda1, lambda_plus,
        lambda1_norm, lp_norm
    """
    if episodes is None:
        episodes = CRISIS_EPISODES

    snap_dates = decomp["snap_dates"]
    elevation  = decomp["elevation"]
    results    = []

    for ep in episodes:
        ep_start = pd.Timestamp(ep["start"])
        ep_end   = pd.Timestamp(ep["end"])

        # Mask snapshots within this episode
        mask = (snap_dates >= ep_start) & (snap_dates <= ep_end)
        if mask.sum() == 0:
            # Episode outside data range — skip silently
            continue

        # Peak elevation window within episode
        ep_elevations = elevation.copy()
        ep_elevations[~mask] = -np.inf
        peak_idx = int(np.argmax(ep_elevations))

        ea   = float(decomp["effect_a"][peak_idx])
        eb   = float(decomp["effect_b"][peak_idx])
        elev = float(elevation[peak_idx])
        total = ea + eb

        frac_a = ea / total if total > 1e-12 else float("nan")
        frac_b = eb / total if total > 1e-12 else float("nan")

        results.append({
            "name":              ep["name"],
            "ep_start":          ep_start.date(),
            "ep_end":            ep_end.date(),
            "peak_date":         snap_dates[peak_idx].date(),
            "log_rho":           float(decomp["log_rho"][peak_idx]),
            "log_rho_baseline":  decomp["log_rho_baseline"],
            "elevation":         elev,
            "effect_a":          ea,
            "effect_b":          eb,
            "frac_a":            frac_a,
            "frac_b":            frac_b,
            "lambda1":           float(decomp["lambda1"][peak_idx]),
            "lambda_plus":       float(decomp["lambda_plus"][peak_idx]),
            "lambda1_norm":      float(decomp["lambda1_norm"][peak_idx]),
            "lp_norm":           float(decomp["lp_norm"][peak_idx]),
        })

    return results


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals for AUROC
# ---------------------------------------------------------------------------


def auroc_bootstrap_ci(
    values: np.ndarray,
    labels: np.ndarray,
    stat_name: str = "statistic",
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Compute a bootstrap percentile confidence interval for AUROC.

    Uses paired resampling (values and labels resampled together) so that
    the class balance is preserved on average across bootstrap samples.

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        Spectral statistic values.
    labels : np.ndarray of int, shape (n,)
        Binary crisis labels (1 = crisis, 0 = calm).
    stat_name : str
        Name for display.
    n_resamples : int
        Number of bootstrap resamples.  2000 is sufficient for 95% CI.
    confidence_level : float
        Nominal confidence level (e.g., 0.95 for 95% CI).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        stat_name     : str
        auroc         : float  — point estimate (on original data)
        ci_lower      : float  — lower confidence bound
        ci_upper      : float  — upper confidence bound
        confidence_level : float
        n_resamples   : int
        n_valid       : int    — number of non-NaN observations used
        note          : str    — interpretive note

    Notes
    -----
    Bootstrap CIs on AUROC are naive in the presence of serial correlation
    (overlapping rolling windows).  The resulting intervals are likely
    anti-conservative (too narrow).  They should be reported with this
    caveat.  With only ~7 distinct crisis episodes the effective sample
    size is much smaller than n_valid; the intervals are wide regardless.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(random_state)

    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)

    valid = ~np.isnan(values)
    v = values[valid]
    l = labels[valid]
    n = len(v)

    # Point estimate
    if len(np.unique(l)) < 2:
        return {
            "stat_name":        stat_name,
            "auroc":            float("nan"),
            "ci_lower":         float("nan"),
            "ci_upper":         float("nan"),
            "confidence_level": confidence_level,
            "n_resamples":      n_resamples,
            "n_valid":          n,
            "note":             "Only one class present; AUROC undefined.",
        }

    point_auroc = float(roc_auc_score(l, v))

    # Bootstrap resamples
    boot_aurocs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        v_b = v[idx]
        l_b = l[idx]
        if len(np.unique(l_b)) < 2:
            # Resample happened to draw only one class — skip
            continue
        try:
            boot_aurocs.append(float(roc_auc_score(l_b, v_b)))
        except Exception:
            continue

    boot_aurocs = np.array(boot_aurocs)
    alpha = 1.0 - confidence_level
    ci_lo = float(np.percentile(boot_aurocs, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_aurocs, 100 * (1 - alpha / 2)))

    note = (
        "Bootstrap CI uses paired resampling (i.i.d. assumption). "
        "Serial correlation in rolling windows makes these intervals "
        "anti-conservative (likely too narrow). "
        f"Effective n ~= number of distinct crisis episodes (~7), not {n}."
    )

    return {
        "stat_name":        stat_name,
        "auroc":            point_auroc,
        "ci_lower":         ci_lo,
        "ci_upper":         ci_hi,
        "confidence_level": confidence_level,
        "n_resamples":      len(boot_aurocs),
        "n_valid":          n,
        "note":             note,
    }


def run_auroc_bootstrap(
    snapshots: list,
    ks_full_vals: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    episodes: Optional[list] = None,
    expand_days: int = 0,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict:
    """
    Compute bootstrap AUROC CIs for all four primary spectral statistics.

    Parameters
    ----------
    snapshots : list of SpectralSnapshot
    ks_full_vals : np.ndarray
        Pre-extracted ks_full values (shape = len(snapshots)).
    dates : pd.DatetimeIndex or None
    episodes : list of dict or None
    expand_days : int
    n_resamples, confidence_level, random_state
        Passed to auroc_bootstrap_ci().

    Returns
    -------
    dict mapping stat_name -> auroc_bootstrap_ci result dict
    """
    ks_vals  = np.array([s.ks  for s in snapshots])
    k_vals   = np.array([s.k   for s in snapshots], dtype=float)
    rho_vals = np.array([s.rho for s in snapshots])

    if dates is not None:
        snap_dates = dates[[s.center_pos for s in snapshots]]
    else:
        snap_dates = pd.DatetimeIndex(
            [pd.Timestamp("2000-01-01") + pd.Timedelta(days=s.center_pos)
             for s in snapshots]
        )

    labels = make_crisis_labels(snap_dates, episodes=episodes, expand_days=expand_days)

    stats = {
        "ks":      ks_vals,
        "k":       k_vals,
        "rho":     rho_vals,
        "ks_full": ks_full_vals,
    }

    return {
        name: auroc_bootstrap_ci(
            vals, labels,
            stat_name=name,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            random_state=random_state,
        )
        for name, vals in stats.items()
    }


# ---------------------------------------------------------------------------
# Extension 1: Velocity statistics  dρ/dt  and  dk/dt
# ---------------------------------------------------------------------------

def compute_velocity_stats(
    snapshots: list,
    dates: pd.DatetimeIndex,
    episodes: Optional[list] = None,
    expand_days: int = 21,
    calm_start: str = "2004-01-01",
    calm_end:   str = "2007-06-01",
) -> dict:
    """
    Compute first-difference velocity statistics dρ/dt and dk/dt and their
    discriminative power vs the crisis calendar.

    For each snapshot t:
        dρ(t) = ρ(t) − ρ(t−1)    [NaN at t=0]
        dk(t) = k(t) − k(t−1)    [NaN at t=0]

    A 3-window centered moving average is applied to both to reduce noise
    (endpoint windows use 2-window or 1-window averages).

    AUROC is computed for both the raw and smoothed velocity series against
    the same crisis-calendar labels used for level statistics.

    Parameters
    ----------
    snapshots : list of SpectralSnapshot
    dates : pd.DatetimeIndex
    episodes : list of dict or None
        Crisis episodes.  Defaults to CRISIS_EPISODES.
    expand_days : int
        Window expansion for crisis labels (consistent with level-stat AUROC).
    calm_start, calm_end : str
        Calm-period dates for the lead/lag threshold calibration.

    Returns
    -------
    dict with keys:
        snap_dates  : pd.DatetimeIndex
        rho         : np.ndarray
        k           : np.ndarray
        d_rho       : np.ndarray   (raw first difference)
        d_k         : np.ndarray   (raw first difference)
        d_rho_smooth: np.ndarray   (3-window centered MA)
        d_k_smooth  : np.ndarray   (3-window centered MA)
        labels      : np.ndarray
        auroc_d_rho        : float
        auroc_d_k          : float
        auroc_d_rho_smooth : float
        auroc_d_k_smooth   : float
        lead_lag_d_rho     : dict  (per-crisis first-crossing lead/lag in windows)
        lead_lag_d_k       : dict
    """
    snap_dates = pd.DatetimeIndex([dates[s.center_pos] for s in snapshots])
    rho = np.array([s.rho for s in snapshots])
    k   = np.array([s.k   for s in snapshots], dtype=float)

    # First differences
    d_rho = np.full(len(rho), np.nan)
    d_k   = np.full(len(k),   np.nan)
    d_rho[1:] = np.diff(rho)
    d_k[1:]   = np.diff(k)

    # 3-window centered moving average
    def _smooth3(x):
        s = np.full_like(x, np.nan)
        for i in range(len(x)):
            win = [v for v in [
                x[i-1] if i > 0 else np.nan,
                x[i],
                x[i+1] if i < len(x)-1 else np.nan,
            ] if not np.isnan(v)]
            if win:
                s[i] = float(np.mean(win))
        return s

    d_rho_smooth = _smooth3(d_rho)
    d_k_smooth   = _smooth3(d_k)

    # Crisis labels
    labels = make_crisis_labels(snap_dates, episodes=episodes, expand_days=expand_days)

    def _auroc(vals):
        roc = compute_roc(vals, labels, stat_name="")
        return roc.auroc

    return {
        "snap_dates":        snap_dates,
        "rho":               rho,
        "k":                 k,
        "d_rho":             d_rho,
        "d_k":               d_k,
        "d_rho_smooth":      d_rho_smooth,
        "d_k_smooth":        d_k_smooth,
        "labels":            labels,
        "auroc_d_rho":       _auroc(d_rho),
        "auroc_d_k":         _auroc(d_k),
        "auroc_d_rho_smooth": _auroc(d_rho_smooth),
        "auroc_d_k_smooth":   _auroc(d_k_smooth),
    }


# ---------------------------------------------------------------------------
# Extension 2: Eigenvector subspace overlap  O(t)
# ---------------------------------------------------------------------------

def compute_subspace_overlap(
    snapshots: list,
    dates: pd.DatetimeIndex,
    k: int = 5,
    episodes: Optional[list] = None,
    expand_days: int = 21,
    calm_start: str = "2004-01-01",
    calm_end:   str = "2007-06-01",
    lead_lag_threshold_sigma: float = 1.5,
    lead_lag_look_back_days:  int   = 126,
) -> dict:
    """
    Compute consecutive-window eigenvector subspace overlap O(t) for top-k
    eigenvectors and the instability measure 1−O(t).

    O(t) = ||V_k(t)ᵀ V_k(t−1)||²_F / k

    where V_k(t) is the d×k matrix of the top-k eigenvectors at window t and
    the Frobenius norm gives the total squared projection mass.  O(t)=1 means
    the subspace is identical; O(t)=0 means orthogonal.

    When d(t) ≠ d(t−1), the overlap is computed in the intersection of the
    active asset sets for both windows.  If the intersection has fewer than
    k assets, O(t) is set to NaN.

    IMPORTANT: SpectralSnapshot.eigvecs_top must be populated (i.e., the
    estimator must have been run with store_top_k_eigvecs >= k).

    Lead/lag analysis
    -----------------
    For each of the measured crisis episodes (excluding Flash Crash which is
    a single day), we report the number of 21-day windows by which 1−O(t)
    first crosses a threshold (calm mean + lead_lag_threshold_sigma * calm std)
    BEFORE the crisis start date, versus the same crossing for ρ(t).
    A positive lead = statistic alarms before crisis; negative = lags.

    Parameters
    ----------
    snapshots : list of SpectralSnapshot (must have eigvecs_top populated)
    dates : pd.DatetimeIndex
    k : int
        Number of top eigenvectors to use.
    episodes, expand_days : crisis labeling params
    calm_start, calm_end : calibration period for threshold
    lead_lag_threshold_sigma : float
        Threshold = calm_mean + this * calm_std
    lead_lag_look_back_days : int
        How many calendar days before crisis start to look for first crossing.

    Returns
    -------
    dict with keys:
        snap_dates    : pd.DatetimeIndex
        overlap       : np.ndarray   O(t), NaN where undefined
        instability   : np.ndarray   1 − O(t)
        labels        : np.ndarray
        auroc_instab  : float        AUROC of 1−O(t)
        lead_lag      : list of dict  per-crisis lead/lag analysis
        k             : int          subspace dimension used
        n_nan_windows : int          windows where O(t) is NaN (d mismatch)
    """
    if any(s.eigvecs_top is None for s in snapshots):
        raise ValueError(
            "eigvecs_top is not populated in snapshots. "
            "Re-run RollingSpectralEstimator with store_top_k_eigvecs >= %d." % k
        )

    snap_dates = pd.DatetimeIndex([dates[s.center_pos] for s in snapshots])
    n          = len(snapshots)
    overlap    = np.full(n, np.nan)

    for t in range(1, n):
        s_cur  = snapshots[t]
        s_prev = snapshots[t - 1]

        # Find common assets
        cols_cur  = s_cur.active_cols
        cols_prev = s_prev.active_cols

        if cols_cur is None or cols_prev is None:
            continue

        common = np.intersect1d(cols_cur, cols_prev)
        if len(common) < k:
            continue   # too few common assets → NaN

        # Map common assets to row indices in each eigvec matrix
        idx_cur  = np.searchsorted(cols_cur,  common)
        idx_prev = np.searchsorted(cols_prev, common)

        V_cur  = s_cur.eigvecs_top[idx_cur,  :k]   # (n_common, k)
        V_prev = s_prev.eigvecs_top[idx_prev, :k]   # (n_common, k)

        # Overlap = ||V_cur^T V_prev||^2_F / k
        M = V_cur.T @ V_prev   # (k, k)
        overlap[t] = float(np.sum(M ** 2)) / k

    instability = 1.0 - np.where(np.isnan(overlap), np.nan, overlap)

    labels = make_crisis_labels(snap_dates, episodes=episodes, expand_days=expand_days)
    auroc_instab = compute_roc(instability, labels, stat_name="instability").auroc

    # ── Lead/lag analysis ───────────────────────────────────────────────────
    # Calm period thresholds
    calm_mask = (snap_dates >= pd.Timestamp(calm_start)) & \
                (snap_dates <= pd.Timestamp(calm_end))

    rho_arr = np.array([s.rho for s in snapshots])

    def _threshold(arr, mask):
        cal = arr[mask & ~np.isnan(arr)]
        if len(cal) < 4:
            return np.nan
        return float(np.nanmean(cal)) + lead_lag_threshold_sigma * float(np.nanstd(cal, ddof=1))

    thr_instab = _threshold(instability, calm_mask)
    thr_rho    = _threshold(rho_arr,     calm_mask)

    # Excluded single-day events from lead/lag (cannot meaningfully lead)
    SKIP_EPISODES = {"Flash Crash", "Post-9/11 shock"}
    eps = episodes if episodes is not None else CRISIS_EPISODES

    lead_lag = []
    for ep in eps:
        if ep["name"] in SKIP_EPISODES:
            continue
        cr_s = pd.Timestamp(ep["start"])
        cr_e = pd.Timestamp(ep["end"])

        # Search window: [crisis_start - look_back, crisis_end]
        look_back = pd.Timedelta(days=lead_lag_look_back_days)
        win_start = cr_s - look_back

        def _first_crossing(arr, threshold):
            """First snapshot in [win_start, cr_e] where arr > threshold."""
            if np.isnan(threshold):
                return None
            for i, (d_i, v) in enumerate(zip(snap_dates, arr)):
                if d_i < win_start:
                    continue
                if d_i > cr_e:
                    break
                if not np.isnan(v) and v > threshold:
                    return d_i
            return None

        date_instab = _first_crossing(instability, thr_instab)
        date_rho    = _first_crossing(rho_arr,     thr_rho)

        def _lead_windows(dt):
            """Lead in 21-day windows: positive = leads crisis, negative = lags."""
            if dt is None:
                return None
            days = (cr_s - dt).days   # positive if dt < cr_s (leads)
            return round(days / 21, 1)

        lead_lag.append({
            "name":           ep["name"],
            "crisis_start":   cr_s.date(),
            "date_instab":    date_instab.date() if date_instab else None,
            "date_rho":       date_rho.date()    if date_rho    else None,
            "lead_instab_windows": _lead_windows(date_instab),
            "lead_rho_windows":    _lead_windows(date_rho),
        })

    return {
        "snap_dates":    snap_dates,
        "overlap":       overlap,
        "instability":   instability,
        "labels":        labels,
        "auroc_instab":  auroc_instab,
        "lead_lag":      lead_lag,
        "k":             k,
        "n_nan_windows": int(np.sum(np.isnan(overlap))),
    }


# ---------------------------------------------------------------------------
# Extension 3: Granger causality — does ρ/k/O predict future volatility?
# ---------------------------------------------------------------------------

def _ols_fit(y: np.ndarray, X: np.ndarray):
    """OLS via lstsq; returns (beta, RSS, df_resid)."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss   = float(resid @ resid)
    df    = len(y) - X.shape[1]
    return beta, rss, df


def compute_granger(
    snapshots: list,
    dates: pd.DatetimeIndex,
    R: np.ndarray,
    instability: Optional[np.ndarray] = None,
    max_lag: int = 3,
) -> dict:
    """
    Test whether spectral statistics Granger-cause realized volatility.

    Realized volatility (proxy):
        RV(t) = mean_i |r_i(t)| averaged over all assets in the window's
        center day.  One value per snapshot window.

    For each predictor X ∈ {ρ, k, 1−O(t)} and lag p ∈ {1, 2, 3}:

    Restricted:    RV(t) = α + Σ_{j=1}^p β_j RV(t−j) + ε
    Unrestricted:  RV(t) = α + Σ_{j=1}^p β_j RV(t−j) + Σ_{j=1}^p γ_j X(t−j) + ε

    F-statistic: ((RSS_r − RSS_u) / p) / (RSS_u / df_u)
    H₀: all γ_j = 0 (X does not Granger-cause RV)

    Parameters
    ----------
    snapshots : list of SpectralSnapshot
    dates : pd.DatetimeIndex
    R : np.ndarray, shape (T_total, d_total)
        Raw returns matrix (same as passed to the estimator).
    instability : np.ndarray or None
        1−O(t) series from compute_subspace_overlap().  If None, Granger
        test for O is skipped.
    max_lag : int
        Maximum lag p to test.

    Returns
    -------
    dict mapping predictor name → dict mapping lag → {f_stat, p_val, df1, df2}
    Plus key 'rv' : np.ndarray  (realized volatility series aligned to snapshots)
    """
    from scipy import stats as scipy_stats

    snap_dates = pd.DatetimeIndex([dates[s.center_pos] for s in snapshots])
    n          = len(snapshots)

    # Build RV(t): mean |return| across all assets on the CENTER day of each window.
    rv = np.full(n, np.nan)
    for i, s in enumerate(snapshots):
        day_returns = R[s.center_pos, :]
        valid = day_returns[~np.isnan(day_returns)]
        if len(valid) > 0:
            rv[i] = float(np.mean(np.abs(valid)))

    rho_arr = np.array([s.rho for s in snapshots])
    k_arr   = np.array([s.k   for s in snapshots], dtype=float)

    predictors = {"rho": rho_arr, "k": k_arr}
    if instability is not None:
        predictors["O_instab"] = instability

    results = {}

    for name, x_arr in predictors.items():
        results[name] = {}
        for p in range(1, max_lag + 1):
            T_eff = n - p
            if T_eff < p + 5:
                continue

            # Align: y = RV[p:], lagged RV and X are [p-j-1 : T-j-1]
            y = rv[p:]
            lags_rv = np.column_stack(
                [rv[p - j - 1 : n - j - 1] for j in range(p)]
            )
            lags_x  = np.column_stack(
                [x_arr[p - j - 1 : n - j - 1] for j in range(p)]
            )

            # Remove rows with any NaN
            X_r  = np.column_stack([np.ones(T_eff), lags_rv])
            X_ur = np.column_stack([np.ones(T_eff), lags_rv, lags_x])
            valid_mask = (
                ~np.isnan(y) &
                ~np.any(np.isnan(X_ur), axis=1)
            )
            y_v   = y[valid_mask]
            Xr_v  = X_r[valid_mask]
            Xur_v = X_ur[valid_mask]

            n_v = len(y_v)
            if n_v < X_ur.shape[1] + 5:
                continue

            _, rss_r, df_r = _ols_fit(y_v, Xr_v)
            _, rss_u, df_u = _ols_fit(y_v, Xur_v)

            if df_u <= 0 or rss_u < 1e-20:
                continue

            f_stat = ((rss_r - rss_u) / p) / (rss_u / df_u)
            p_val  = float(scipy_stats.f.sf(f_stat, p, df_u))

            results[name][p] = {
                "f_stat": float(f_stat),
                "p_val":  p_val,
                "df1":    p,
                "df2":    df_u,
                "n":      n_v,
            }

    results["rv"] = rv
    return results


# ---------------------------------------------------------------------------
# Extension 4: Combined multivariate detector (logistic regression)
# ---------------------------------------------------------------------------

def compute_combined_detector(
    snapshots: list,
    dates: pd.DatetimeIndex,
    ks_full_vals: np.ndarray,
    instability: Optional[np.ndarray] = None,
    episodes: Optional[list] = None,
    train_end: str = "2012-12-31",
    test_start: str = "2013-01-01",
    expand_days: int = 21,
    C_reg: float = 1.0,
    random_state: int = 42,
) -> dict:
    """
    Train an L2-regularized logistic regression detector on 2004–2012 and
    evaluate out-of-sample on 2013–2024.

    Features: ρ(t), k(t), dρ(t), dk(t), 1−O(t) [if provided], r_eff(t)
    Label: binary crisis calendar

    All features are standardized using train-set mean and std (no look-ahead).

    Parameters
    ----------
    snapshots : list of SpectralSnapshot
    dates : pd.DatetimeIndex
    ks_full_vals : np.ndarray  (not used as feature but kept for consistency)
    instability : np.ndarray or None
        1−O(t) from compute_subspace_overlap(); if None, feature is excluded.
    episodes : list or None
    train_end : str   Last date of training period (inclusive)
    test_start : str  First date of test period (inclusive)
    expand_days : int  Label expansion
    C_reg : float     Inverse regularization strength for LogisticRegression
    random_state : int

    Returns
    -------
    dict with keys:
        feature_names   : list of str
        train_auroc     : float
        test_auroc      : float
        test_roc        : RocResult
        coef            : np.ndarray  (standardized feature coefficients)
        intercept       : float
        n_train, n_test : int
        n_crisis_train, n_crisis_test : int
        false_alarm_rate_at_70tpr : float  (FPR at the TPR=0.70 operating point)
        rho_only_test_auroc : float  (benchmark: ρ alone, same test set)
        feature_importance : list of (name, coef) sorted by |coef|
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    snap_dates = pd.DatetimeIndex([dates[s.center_pos] for s in snapshots])
    labels_all = make_crisis_labels(snap_dates, episodes=episodes, expand_days=expand_days)

    rho  = np.array([s.rho   for s in snapshots])
    k    = np.array([s.k     for s in snapshots], dtype=float)
    reff = np.array([s.r_eff for s in snapshots])

    # Velocity features
    d_rho = np.full(len(rho), np.nan)
    d_k   = np.full(len(k),   np.nan)
    d_rho[1:] = np.diff(rho)
    d_k[1:]   = np.diff(k)

    # Build feature matrix
    feature_names = ["rho", "k", "d_rho", "d_k", "r_eff"]
    feat_arrays   = [rho, k, d_rho, d_k, reff]

    if instability is not None:
        feature_names.append("instability_1mO")
        feat_arrays.append(instability)

    X_all = np.column_stack(feat_arrays)   # (n, n_feats)
    y_all = labels_all

    # Train / test split by date
    train_mask = snap_dates <= pd.Timestamp(train_end)
    test_mask  = snap_dates >= pd.Timestamp(test_start)

    X_train_raw = X_all[train_mask]
    y_train     = y_all[train_mask]
    X_test_raw  = X_all[test_mask]
    y_test      = y_all[test_mask]

    # Drop NaN rows from train (NaN from d_rho/d_k at t=0, or instability)
    valid_train = ~np.any(np.isnan(X_train_raw), axis=1)
    X_train_raw = X_train_raw[valid_train]
    y_train     = y_train[valid_train]

    valid_test = ~np.any(np.isnan(X_test_raw), axis=1)
    X_test_raw = X_test_raw[valid_test]
    y_test     = y_test[valid_test]

    if len(np.unique(y_train)) < 2:
        warnings.warn("Training set has only one class — cannot train logistic regression.",
                      RuntimeWarning, stacklevel=2)
        return {"error": "insufficient class balance in training set"}

    # Standardize using train statistics (no look-ahead)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    clf = LogisticRegression(C=C_reg, penalty="l2", solver="lbfgs",
                             max_iter=1000, random_state=random_state)
    clf.fit(X_train, y_train)

    prob_train = clf.predict_proba(X_train)[:, 1]
    prob_test  = clf.predict_proba(X_test)[:, 1]

    from sklearn.metrics import roc_auc_score, roc_curve
    train_auroc = float(roc_auc_score(y_train, prob_train)) \
        if len(np.unique(y_train)) > 1 else float("nan")
    test_auroc  = float(roc_auc_score(y_test,  prob_test))  \
        if len(np.unique(y_test))  > 1 else float("nan")

    fpr_arr, tpr_arr, thr_arr = roc_curve(y_test, prob_test, pos_label=1) \
        if len(np.unique(y_test)) > 1 else (np.array([0.,1.]), np.array([0.,1.]), np.array([0.]))

    test_roc = RocResult(
        stat_name="combined_logistic",
        tpr=tpr_arr, fpr=fpr_arr, thresholds=thr_arr, auroc=test_auroc,
    )

    # FPR at TPR ≈ 0.70 operating point
    tpr_target = 0.70
    if len(tpr_arr) > 1:
        idx70 = np.searchsorted(tpr_arr, tpr_target)
        idx70 = min(idx70, len(fpr_arr) - 1)
        far70 = float(fpr_arr[idx70])
    else:
        far70 = float("nan")

    # Benchmark: ρ alone on test set (same rows, no NaN masking needed)
    rho_test_idx = np.where(test_mask)[0][valid_test]
    rho_test_vals = rho[rho_test_idx] if len(rho_test_idx) == len(y_test) else rho[test_mask][valid_test]
    rho_only_auroc = float(roc_auc_score(y_test, rho_test_vals)) \
        if len(np.unique(y_test)) > 1 else float("nan")

    coef = clf.coef_[0]
    feature_importance = sorted(
        zip(feature_names, coef.tolist()), key=lambda x: abs(x[1]), reverse=True
    )

    return {
        "feature_names":              feature_names,
        "train_auroc":                train_auroc,
        "test_auroc":                 test_auroc,
        "test_roc":                   test_roc,
        "coef":                       coef,
        "intercept":                  float(clf.intercept_[0]),
        "n_train":                    int(valid_train.sum()),
        "n_test":                     int(valid_test.sum()),
        "n_crisis_train":             int(np.sum(y_train)),
        "n_crisis_test":              int(np.sum(y_test)),
        "false_alarm_rate_at_70tpr":  far70,
        "rho_only_test_auroc":        rho_only_auroc,
        "feature_importance":         feature_importance,
        "scaler_mean":                scaler.mean_,
        "scaler_std":                 scaler.scale_,
    }
