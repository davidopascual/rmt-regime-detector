"""
plots.py — Publication-quality figures for the RMT spectral research.

Six figures, all matplotlib-only (no seaborn, no plotly):

  Figure 1  — Rolling σ²(t), λ+(t), λ₁(t) time series
  Figure 2  — Rolling k(t) and ρ(t) time series
  Figure 3  — Rolling KS statistics with CUSUM statistic and alarm markers
  Figure 4  — ESD vs. MP density at four dates (calm, pre-crisis, Lehman, COVID)
  Figure 5  — ROC curves for ks_full, k, rho vs. crisis calendar
  Figure 6  — Effective rank r_eff(t) and condition number κ(t) time series

Style
-----
All figures use a clean matplotlib style.  Line widths, font sizes, and
colour choices follow journal guidelines for double-column figures.

Usage
-----
    from research.analysis.plots import all_figures
    figs = all_figures(snapshots, cusum_result, roc_results, dates)
    figs['fig4'].savefig('fig4_esd_comparison.pdf', bbox_inches='tight')
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from research.core.mp_theory import mp_density, mp_cdf, bulk_edges, full_ks_distance_from_mp


matplotlib.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.2,
    "figure.constrained_layout.use": True,
})

C_BLUE   = "#1f77b4"
C_ORANGE = "#ff7f0e"
C_RED    = "#d62728"
C_GREEN  = "#2ca02c"
C_GREY   = "#7f7f7f"
C_PURPLE = "#9467bd"


# ---------------------------------------------------------------------------
# Helper: map center_pos indices to dates
# ---------------------------------------------------------------------------

def _xaxis(snapshots: list, dates=None) -> tuple[np.ndarray, str]:
    if dates is not None:
        try:
            xs = np.array([dates[s.center_pos] for s in snapshots])
            return xs, "Date"
        except (IndexError, KeyError):
            pass
    xs = np.array([s.center_pos for s in snapshots])
    return xs, "Observation index"


def _shade_crises(ax, xs: np.ndarray, alpha: float = 0.13) -> None:
    """
    Shade known crisis periods as light-red vertical bands.
    xs must be a numpy datetime64 array (the x-axis values of the plot).
    Shading is clipped to [xs[0], xs[-1]] to avoid extending beyond the data.
    """
    from research.analysis.validation import CRISIS_EPISODES
    xs_lo = pd.Timestamp(xs[0])
    xs_hi = pd.Timestamp(xs[-1])
    for ep in CRISIS_EPISODES:
        ep_s = max(pd.Timestamp(ep["start"]), xs_lo)
        ep_e = min(pd.Timestamp(ep["end"]),   xs_hi)
        if ep_s < ep_e:
            ax.axvspan(ep_s, ep_e, alpha=alpha, color=C_RED, zorder=0, lw=0)


def _nearest_snapshot(snapshots, dates, target_str: str) -> int:
    """Return index into snapshots list nearest to target_str date."""
    target = pd.Timestamp(target_str)
    snap_dates = pd.DatetimeIndex([dates[s.center_pos] for s in snapshots])
    diffs = np.abs((snap_dates - target).total_seconds().values)
    return int(np.argmin(diffs))


# ---------------------------------------------------------------------------
# Figure 1: Rolling σ²(t), λ+(t), λ₁(t), ρ(t) — 4-panel with crisis shading
# ---------------------------------------------------------------------------

def figure1_rolling_spectral(
    snapshots: list,
    dates=None,
) -> plt.Figure:
    """
    Four-panel plot of σ²(t), λ+(t), λ₁(t), and ρ(t) with crisis shading.

    Panel 1: σ²(t) — noise floor estimate
    Panel 2: λ+(t) — MP bulk upper edge
    Panel 3: λ₁(t) — largest eigenvalue (market mode)
    Panel 4: ρ(t) = λ₁/λ+ — detachment ratio (dimensionless)
              reference line at ρ=1, COVID peak annotated
    """
    xs, xlabel = _xaxis(snapshots, dates)

    sigma2_vals  = np.array([s.sigma2      for s in snapshots])
    lp_vals      = np.array([s.lambda_plus for s in snapshots])
    lambda1_vals = np.array([s.lambda1     for s in snapshots])
    rho_vals     = np.array([s.rho         for s in snapshots])

    fig, axes = plt.subplots(4, 1, figsize=(8.0, 9.0), sharex=True)
    ax1, ax2, ax3, ax4 = axes

    # Panel 1: σ²(t)
    ax1.plot(xs, sigma2_vals, color=C_BLUE, lw=1.2)
    ax1.set_ylabel(r"$\hat{\sigma}^2(t)$")
    ax1.set_title("Rolling spectral parameters with crisis periods (shaded)")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    _shade_crises(ax1, xs)

    # Panel 2: λ+(t)
    ax2.plot(xs, lp_vals, color=C_ORANGE, lw=1.2)
    ax2.set_ylabel(r"$\lambda_+(t)$")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    _shade_crises(ax2, xs)

    # Panel 3: λ₁(t)
    ax3.plot(xs, lambda1_vals, color=C_GREEN, lw=1.2)
    ax3.set_ylabel(r"$\lambda_1(t)$")
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    _shade_crises(ax3, xs)

    # Panel 4: ρ(t) = λ₁/λ+
    ax4.plot(xs, rho_vals, color=C_RED, lw=1.2)
    ax4.axhline(1.0, color=C_GREY, ls="--", lw=0.8, label=r"$\rho = 1$")
    ax4.set_ylabel(r"$\rho(t) = \lambda_1 / \lambda_+$" "\n(dimensionless)")
    ax4.set_xlabel(xlabel)
    ax4.legend(loc="upper left", fontsize=7)
    _shade_crises(ax4, xs)

    # Annotate COVID peak
    covid_peak_idx = int(np.argmax(rho_vals))
    covid_peak_x   = xs[covid_peak_idx]
    covid_peak_y   = rho_vals[covid_peak_idx]
    ax4.annotate(
        f"COVID peak\n\u03c1 = {covid_peak_y:.0f}",
        xy=(covid_peak_x, covid_peak_y),
        xytext=(0, -30),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=7,
        color=C_RED,
        arrowprops=dict(arrowstyle="-", color=C_RED, lw=0.8),
    )

    return fig


# ---------------------------------------------------------------------------
# Figure 2: Rolling k(t) and ρ(t)
# ---------------------------------------------------------------------------

def figure2_spikes_and_detachment(
    snapshots: list,
    dates=None,
) -> plt.Figure:
    """
    Two-panel plot of k(t) (spike count) and ρ(t) = λ₁/λ+ (detachment ratio).
    """
    xs, xlabel = _xaxis(snapshots, dates)

    k_vals   = np.array([s.k   for s in snapshots])
    rho_vals = np.array([s.rho for s in snapshots])

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.5), sharex=True)
    ax1, ax2 = axes

    ax1.step(xs, k_vals, color=C_PURPLE, where="post")
    ax1.set_ylabel(r"$k(t)$  [spike count]")
    ax1.set_title(r"Spike count $k(t)$ and detachment ratio $\rho(t)$")
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax2.plot(xs, rho_vals, color=C_RED)
    ax2.axhline(1.0, color=C_GREY, ls="--", lw=0.8)
    ax2.set_ylabel(r"$\rho(t) = \lambda_1/\lambda_+$")
    ax2.set_xlabel(xlabel)

    return fig


# ---------------------------------------------------------------------------
# Figure 3: KS statistics with CUSUM overlay
# ---------------------------------------------------------------------------

def figure3_ks_cusum(
    snapshots: list,
    cusum_result,
    dates=None,
) -> plt.Figure:
    """
    Three-panel: bulk KS(t), full KS(t), and CUSUM C(t) with alarms.
    """
    xs, xlabel = _xaxis(snapshots, dates)

    ks_bulk  = np.array([s.ks      for s in snapshots])
    ks_full  = np.array([s.ks_full for s in snapshots])
    cusum    = cusum_result.cusum
    alarms   = cusum_result.alarms
    h        = cusum_result.h

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 6.0), sharex=True)
    ax1, ax2, ax3 = axes

    ax1.plot(xs, ks_bulk, color=C_BLUE)
    ax1.axhline(cusum_result.mu, color=C_GREY, ls=":", lw=0.8,
                label=r"$\mu_{\rm cal}$ (bulk)")
    ax1.set_ylabel("Bulk KS(t)")
    ax1.set_title("KS statistics and CUSUM change-point detection")
    ax1.legend(loc="upper right")

    ax2.plot(xs, ks_full, color=C_PURPLE, label=r"Full KS = $k(t)/d$  approx.")
    ax2.set_ylabel("Full KS(t)")
    ax2.legend(loc="upper right")
    if len(alarms) > 0:
        ax2.scatter(xs[alarms], ks_full[alarms], color=C_RED, zorder=5,
                    s=25, marker="^", label="CUSUM alarm")

    ax3.plot(xs, cusum, color=C_ORANGE)
    ax3.axhline(h, color=C_RED, ls="--", lw=1.0, label=f"$h = {h:.3f}$")
    ax3.set_ylabel(r"CUSUM $C(t)$")
    ax3.set_xlabel(xlabel)
    ax3.legend(loc="upper right")

    return fig


# ---------------------------------------------------------------------------
# Figure 4: ESD vs. MP density at four dates
# ---------------------------------------------------------------------------

def figure4_esd_comparison(
    snapshots: list,
    dates=None,
    date_labels: Optional[list[tuple[str, str]]] = None,
    n_bins: int = 40,
    x_clip_factor: float = 3.0,
) -> tuple[plt.Figure, list[dict]]:
    """
    2×2 panel comparing the empirical spectral distribution to the MP density
    at four significant dates.

    Parameters
    ----------
    snapshots : list of SpectralSnapshot
    dates : pd.DatetimeIndex or None
        Row dates of the original returns array (for label lookup).
    date_labels : list of (target_date_str, panel_label) or None
        Four (date, label) pairs.  Defaults to:
          (calm 2006, pre-GFC 2007, Lehman Oct 2008, COVID Apr 2020)
    n_bins : int
        Histogram bin count per panel.
    x_clip_factor : float
        X-axis clipped to [0, x_clip_factor * lambda+] to keep bulk visible.
        Spike eigenvalues outside this range are annotated separately.

    Returns
    -------
    fig : plt.Figure
    panel_stats : list of dict
        One dict per panel with keys: label, date, ks_bulk, ks_full, k, sigma2,
        lambda_plus, lambda1, n_spikes_outside_clip.
        Printed explicitly to stdout by the caller.
    """
    if date_labels is None:
        date_labels = [
            ("2006-07-05", "Calm (Jul 2006)"),
            ("2007-02-05", "Pre-crisis (Feb 2007)"),
            ("2008-11-03", "Lehman window (Nov 2008)"),
            ("2020-04-09", "COVID crash (Apr 2020)"),
        ]

    if len(date_labels) != 4:
        raise ValueError("date_labels must contain exactly 4 (date, label) tuples")

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    panel_stats = []

    for ax, (target_date, panel_label) in zip(axes.flat, date_labels):
        if dates is not None:
            idx = _nearest_snapshot(snapshots, dates, target_date)
        else:
            idx = 0
        snap = snapshots[idx]

        snap_date_str = str(pd.Timestamp(dates[snap.center_pos]).date()) if dates is not None else f"t={snap.center_pos}"

        lm  = snap.lambda_minus
        lp  = snap.lambda_plus
        s2  = snap.sigma2
        g   = snap.gamma
        eigs = snap.eigenvalues   # descending

        # Clip x-axis to [0, x_clip_factor * lp]
        x_max = x_clip_factor * lp
        bulk_for_hist = eigs[eigs <= x_max]
        n_spikes_outside = int(np.sum(eigs > x_max))

        # Histogram (bulk + small spikes)
        ax.hist(
            bulk_for_hist, bins=n_bins, density=True,
            color=C_BLUE, alpha=0.45, label="Empirical ESD",
            edgecolor="none", range=(0, x_max),
        )

        # MP density curve
        x_lo = max(lm * 0.5, 1e-6)
        x_curve = np.linspace(x_lo, lp * 1.05, 500)
        y_curve = np.array([mp_density(xi, g, s2) for xi in x_curve])
        ax.plot(x_curve, y_curve, color=C_ORANGE, lw=1.8,
                label=r"MP$(\hat{\sigma}^2)$")

        # Bulk edge markers
        ax.axvline(lp, color=C_RED, ls="--", lw=1.0, label=r"$\lambda_+$")
        if lm > 1e-6:
            ax.axvline(lm, color=C_GREY, ls=":", lw=0.8)

        # Spike annotation: show top-3 spike eigenvalues as dashed lines
        spikes_in_range = eigs[(eigs > lp) & (eigs <= x_max)]
        for sp_val in spikes_in_range[:3]:
            ax.axvline(sp_val, color=C_GREEN, ls="-", lw=0.8, alpha=0.7)

        # Full and bulk KS
        ks_b = snap.ks
        ks_f = snap.ks_full

        title = (f"{panel_label}\n"
                 f"$\\hat{{\\sigma}}^2$={s2:.3f}, $\\lambda_+$={lp:.3f}, "
                 f"$k$={snap.k}")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(r"Eigenvalue $\lambda$")
        ax.set_ylabel("Density")
        ax.set_xlim(0, x_max)

        if n_spikes_outside > 0:
            ax.text(0.97, 0.92, f"+{n_spikes_outside} spike(s)\noutside clip",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, color=C_GREEN)

        panel_stats.append({
            "label":               panel_label,
            "snap_date":           snap_date_str,
            "ks_bulk":             ks_b,
            "ks_full":             ks_f,
            "k":                   snap.k,
            "sigma2":              s2,
            "lambda_plus":         lp,
            "lambda1":             snap.lambda1,
            "n_spikes_outside_clip": n_spikes_outside,
        })

    # Shared legend on first panel
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Empirical Spectral Distribution vs. Marcenko-Pastur at Four Dates",
                 fontsize=10)
    return fig, panel_stats


# ---------------------------------------------------------------------------
# Figure 5: ROC curves
# ---------------------------------------------------------------------------

def figure5_roc_curves(
    roc_results: dict,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Overlay ROC curves for ks_full, k, and rho vs. crisis calendar.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
    else:
        fig = ax.get_figure()

    colours = {"ks_full": C_BLUE, "ks": C_PURPLE, "k": C_ORANGE, "rho": C_RED}
    labels  = {
        "ks_full": "Full KS",
        "ks":      "Bulk KS",
        "k":       r"Spike count $k$",
        "rho":     r"Detachment $\rho$",
    }

    for name, roc in roc_results.items():
        c = colours.get(name, C_GREY)
        lbl = labels.get(name, name)
        auroc_str = f"{roc.auroc:.3f}" if not np.isnan(roc.auroc) else "N/A"
        ax.plot(roc.fpr, roc.tpr, color=c, label=f"{lbl} (AUROC={auroc_str})")

    ax.plot([0, 1], [0, 1], color=C_GREY, ls="--", lw=0.8, label="Random (0.500)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC: spectral statistics vs. crisis calendar")
    ax.legend(loc="lower right")
    return fig


# ---------------------------------------------------------------------------
# Figure 6: Effective rank and condition number
# ---------------------------------------------------------------------------

def figure6_reff_kappa(
    snapshots: list,
    dates=None,
) -> plt.Figure:
    """
    Two-panel plot of r_eff(t) (effective rank) and κ(t) (condition number).
    """
    xs, xlabel = _xaxis(snapshots, dates)

    r_eff_vals = np.array([s.r_eff for s in snapshots])
    kappa_vals = np.array([s.kappa for s in snapshots])

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.5), sharex=True)
    ax1, ax2 = axes

    ax1.plot(xs, r_eff_vals, color=C_BLUE)
    ax1.set_ylabel(r"$r_{\rm eff}(t)$  [effective rank]")
    ax1.set_title("Effective rank and condition number")

    ax2.plot(xs, kappa_vals, color=C_RED)
    ax2.set_ylabel(r"$\kappa(t)$  [condition number]")
    ax2.set_xlabel(xlabel)
    ax2.set_yscale("log")

    return fig


# ---------------------------------------------------------------------------
# Figure 7: ρ(t) decomposition — Effect A vs Effect B
# ---------------------------------------------------------------------------

def figure7_rho_decomposition(
    decomp: dict,
) -> plt.Figure:
    """
    Two-panel figure showing the log(ρ) decomposition into Effect A and Effect B.

    Panel 1 (top): λ₁(t)/μ_λ₁ and λ+(t)/μ_λ+ normalized series on the same axis.
      Values above 1 indicate the quantity is above its calm baseline.
    Panel 2 (bottom): Stacked area chart of Effect A and Effect B contributions
      to log(ρ) elevation above the calm-period baseline.
    """
    snap_dates = decomp["snap_dates"]
    xs         = snap_dates.values  # numpy datetime64

    lambda1_norm = decomp["lambda1_norm"]
    lp_norm      = decomp["lp_norm"]
    effect_a     = decomp["effect_a"]
    effect_b     = decomp["effect_b"]
    elevation    = decomp["elevation"]

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    ax1, ax2 = axes

    # Panel 1: normalized series
    ax1.plot(xs, lambda1_norm, color=C_GREEN, lw=1.2,
             label=r"$\lambda_1(t)\,/\,\mu_{\lambda_1}$")
    ax1.plot(xs, lp_norm, color=C_ORANGE, lw=1.2,
             label=r"$\lambda_+(t)\,/\,\mu_{\lambda_+}$")
    ax1.axhline(1.0, color=C_GREY, ls="--", lw=0.7)
    ax1.set_ylabel("Normalized level\n(calm baseline = 1)")
    ax1.set_title(r"$\rho(t)$ decomposition: Effect A ($\lambda_1$ rising) vs Effect B ($\lambda_+$ falling)")
    ax1.legend(loc="upper right", fontsize=7)
    _shade_crises(ax1, xs)

    # Panel 2: stacked area of effect_a and effect_b
    # Only show positive contributions (both are positive during crises)
    ea_pos = np.maximum(effect_a, 0.0)
    eb_pos = np.maximum(effect_b, 0.0)

    ax2.stackplot(
        xs,
        ea_pos,
        eb_pos,
        labels=[
            r"Effect A: $\log(\lambda_1/\mu_{\lambda_1})$",
            r"Effect B: $-\log(\lambda_+/\mu_{\lambda_+})$",
        ],
        colors=[C_GREEN, C_ORANGE],
        alpha=0.70,
    )
    ax2.plot(xs, elevation, color=C_RED, lw=1.0, ls="--",
             label=r"Total elevation $\log\rho(t)-\log\rho_0$")
    ax2.axhline(0.0, color=C_GREY, ls=":", lw=0.7)
    ax2.set_ylabel(r"$\log\rho$ elevation above calm baseline")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper right", fontsize=7)
    _shade_crises(ax2, xs)

    return fig


# ---------------------------------------------------------------------------
# Figure 8: Two-regime CUSUM on ρ(t)
# ---------------------------------------------------------------------------

def figure8_two_regime_cusum(
    regime_results: list,
) -> plt.Figure:
    """
    Two-panel figure showing the two-regime CUSUM on ρ(t).

    Each panel corresponds to one regime (GFC / COVID).
    Shows: ρ(t) series, CUSUM C(t), calibrated threshold h,
    crisis shading, and alarm markers.
    """
    n_regimes = len(regime_results)
    fig, axes = plt.subplots(n_regimes, 2, figsize=(12.0, 4.5 * n_regimes),
                             squeeze=False)

    for row, res in enumerate(regime_results):
        xs         = res["snap_dates_run"].values
        rho_run    = res["rho_run"]
        cusum_run  = res["cusum_run"]
        alarms     = res["alarms_run"]
        h          = res["h"]
        mu         = res["mu"]
        crisis_s   = res["crisis_start"]
        crisis_e   = res["crisis_end"]
        label      = res["crisis_label"]
        name       = res["name"]
        delay      = res["detection_delay_days"]
        fa_count   = res["false_alarm_count"]
        first_det  = res["first_alarm_after_crisis"]

        ax_rho  = axes[row, 0]
        ax_cusum = axes[row, 1]

        # Left: ρ(t) series with crisis band
        ax_rho.plot(xs, rho_run, color=C_RED, lw=1.0)
        ax_rho.axhline(mu, color=C_GREY, ls="--", lw=0.8,
                       label=f"cal. mean = {mu:.2f}")
        ax_rho.axvspan(crisis_s, crisis_e, alpha=0.20, color=C_RED,
                       zorder=0, lw=0, label=label)
        if len(alarms) > 0:
            ax_rho.scatter(xs[alarms], rho_run[alarms], color=C_ORANGE,
                           s=30, zorder=5, marker="^", label="CUSUM alarm")
        ax_rho.set_ylabel(r"$\rho(t) = \lambda_1/\lambda_+$")
        ax_rho.set_title(f"{name}: ρ(t) and alarms")
        ax_rho.legend(loc="upper left", fontsize=7)

        # Right: CUSUM C(t) with threshold h
        ax_cusum.plot(xs, cusum_run, color=C_BLUE, lw=1.0, label=r"CUSUM $C(t)$")
        ax_cusum.axhline(h, color=C_RED, ls="--", lw=1.1,
                         label=f"$h = {h:.3f}$ (ARL≈{res['target_arl_windows']})")
        ax_cusum.axvspan(crisis_s, crisis_e, alpha=0.13, color=C_RED,
                         zorder=0, lw=0)
        if len(alarms) > 0:
            ax_cusum.scatter(xs[alarms], cusum_run[alarms], color=C_ORANGE,
                             s=30, zorder=5, marker="^")

        # Annotate first detection after crisis
        det_text = (f"1st alarm post-{label}: {first_det}\ndelay {delay}d"
                    if first_det is not None else f"No alarm during {label}")
        ax_cusum.text(0.02, 0.96, det_text, transform=ax_cusum.transAxes,
                      va="top", ha="left", fontsize=7, color=C_ORANGE)
        fa_text = f"False alarms (pre-crisis): {fa_count}"
        ax_cusum.text(0.02, 0.84, fa_text, transform=ax_cusum.transAxes,
                      va="top", ha="left", fontsize=7, color=C_GREY)

        ax_cusum.set_ylabel(r"CUSUM $C(t)$")
        ax_cusum.set_title(f"{name}: CUSUM statistic")
        ax_cusum.legend(loc="upper right", fontsize=7)

    for ax in axes.flat:
        ax.set_xlabel("Date")

    fig.suptitle(r"Two-regime CUSUM on $\rho(t)$ — GFC and COVID detection",
                 fontsize=11)
    return fig


# ---------------------------------------------------------------------------
# Convenience: generate all figures
# ---------------------------------------------------------------------------

def all_figures(
    snapshots: list,
    cusum_result,
    roc_results: dict,
    dates=None,
    date_labels_fig4: Optional[list[tuple[str, str]]] = None,
    decomp_result: Optional[dict] = None,
    regime_results: Optional[list] = None,
) -> tuple[dict[str, plt.Figure], list[dict]]:
    """
    Generate all publication figures (6 core + up to 2 supplementary).

    Parameters
    ----------
    snapshots : list of SpectralSnapshot
    cusum_result : CusumResult
    roc_results : dict
        Output of validation.run_full_validation()['roc'].
    dates : pd.DatetimeIndex or None
    date_labels_fig4 : list of (date_str, label) or None
        Four (date, label) pairs for Figure 4.  Defaults to canonical dates.
    decomp_result : dict or None
        Output of validation.decompose_rho().  If provided, Figure 7 is generated.
    regime_results : list or None
        Output of changepoint.two_regime_cusum().  If provided, Figure 8 is generated.

    Returns
    -------
    figs : dict mapping 'fig1' ... 'fig6' (and 'fig7', 'fig8' if inputs provided)
        to plt.Figure objects.
    fig4_stats : list of dicts with per-panel KS values etc.
    """
    figs = {}
    figs["fig1"] = figure1_rolling_spectral(snapshots, dates=dates)
    figs["fig2"] = figure2_spikes_and_detachment(snapshots, dates=dates)
    figs["fig3"] = figure3_ks_cusum(snapshots, cusum_result, dates=dates)
    figs["fig4"], fig4_stats = figure4_esd_comparison(
        snapshots, dates=dates, date_labels=date_labels_fig4
    )
    figs["fig5"] = figure5_roc_curves(roc_results)
    figs["fig6"] = figure6_reff_kappa(snapshots, dates=dates)

    if decomp_result is not None:
        figs["fig7"] = figure7_rho_decomposition(decomp_result)

    if regime_results is not None:
        figs["fig8"] = figure8_two_regime_cusum(regime_results)

    return figs, fig4_stats
