"""
reproduce_figures.py — Reproduce all figures from the RMT spectral paper.

Downloads SP100 returns (2003–2024), runs the rolling Marchenko-Pastur
pipeline, and saves all 8 publication figures to research/data/figures/.

Usage
-----
    python reproduce_figures.py

The first run downloads ~22 years of daily returns via yfinance and caches
them to research/data/returns_panel.parquet. Subsequent runs load from cache
and complete in under 2 minutes.

Output
------
    research/data/figures/fig1_sp100.png  — Rolling σ², λ+, λ₁, ρ (4-panel)
    research/data/figures/fig2_sp100.png  — Spike count k(t) and ρ(t)
    research/data/figures/fig3_sp100.png  — KS statistics and CUSUM
    research/data/figures/fig4_sp100.png  — ESD vs MP at four dates
    research/data/figures/fig5_sp100.png  — ROC curves
    research/data/figures/fig6_sp100.png  — Effective rank and condition number
    research/data/figures/fig7_sp100.png  — ρ(t) decomposition (Effect A vs B)
    research/data/figures/fig8_sp100.png  — Two-regime CUSUM (GFC and COVID)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from research.core.loader import load_returns, returns_to_numpy
from research.core.estimator import RollingSpectralEstimator
from research.core.changepoint import cusum_from_snapshots, two_regime_cusum
from research.analysis.validation import (
    run_full_validation,
    crisis_event_study,
    compute_roc,
    decompose_rho,
    decompose_rho_at_crises,
)
from research.analysis.plots import all_figures

os.makedirs("research/data/figures", exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading returns...")
returns = load_returns()
R, dates, tickers = returns_to_numpy(returns)
print(f"  {R.shape[0]} trading days × {R.shape[1]} assets  "
      f"({dates[0].date()} to {dates[-1].date()})")

# ── Rolling spectral estimator ────────────────────────────────────────────────
print("Running rolling Marchenko-Pastur estimator (window=252, step=21)...")
est = RollingSpectralEstimator(window=252, step=21, min_assets=30)
snapshots = est.fit(R)
print(f"  {len(snapshots)} snapshots  |  "
      f"mean γ = {np.mean([s.gamma for s in snapshots]):.3f}  |  "
      f"mean σ² = {np.mean([s.sigma2 for s in snapshots]):.3f}")

# ── CUSUM (bulk KS, for Figure 3) ────────────────────────────────────────────
cusum_result, _ = cusum_from_snapshots(snapshots, calibration_n=20,
                                        k_delta=0.5, k_h=4.0)

# ── Validation (ROC curves, for Figure 5) ────────────────────────────────────
print("Running validation...")
ks_f = np.array([s.ks_full for s in snapshots])
validation = run_full_validation(snapshots, dates=dates, expand_days=21)
validation["event_study"]["ks_full"] = crisis_event_study(
    ks_f, validation["labels"], "ks_full"
)
validation["roc"]["ks_full"] = compute_roc(ks_f, validation["labels"], "ks_full")

# ── ρ(t) decomposition (for Figure 7) ────────────────────────────────────────
decomp = decompose_rho(snapshots, dates)

# ── Two-regime CUSUM on ρ(t) (for Figure 8) ──────────────────────────────────
regime_results = two_regime_cusum(snapshots, dates)

# ── Generate and save all figures ────────────────────────────────────────────
print("Generating figures...")
figs, _ = all_figures(
    snapshots=snapshots,
    cusum_result=cusum_result,
    roc_results=validation["roc"],
    dates=dates,
    date_labels_fig4=[
        ("2006-07-05", "Calm (Jul 2006)"),
        ("2007-02-05", "Pre-crisis (Feb 2007)"),
        ("2008-11-03", "Lehman window (Nov 2008)"),
        ("2020-04-09", "COVID crash (Apr 2020)"),
    ],
    decomp_result=decomp,
    regime_results=regime_results,
)

for name, fig in figs.items():
    path = f"research/data/figures/{name}_sp100.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)

print("\nDone.")
