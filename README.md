# RMT Spectral Analysis of S&P 100 Returns (2003–2024)

Rolling Marchenko-Pastur pipeline for detecting structural breaks in the
covariance eigenvalue distribution of daily equity returns.

## What this does

Each week, fit the Marchenko-Pastur (MP) bulk law to the empirical spectral
distribution (ESD) of a 252-day rolling return covariance matrix. Extract
the noise floor $\hat{\sigma}^2(t)$, bulk upper edge $\lambda_+(t)$, spike
count $k(t)$, and detachment ratio $\rho(t) = \lambda_1 / \lambda_+$. Run
crisis event studies and ROC analysis against a hand-labeled crisis calendar.

## References

| Reference | Used for |
|-----------|----------|
| Marčenko, Pastur (1967) | MP density and bulk edges $\lambda_\pm$ |
| Laloux et al. (1999) | Noise-dressing interpretation |
| Plerou et al. (2002) | Rolling window methodology |
| Baik, Ben Arous, Péché (2005) | BBP spike detection threshold $\theta_c$ |
| Ledoit, Wolf (2004) | Shrinkage estimation context |
| Bun, Bouchaud, Potters (2017) | Self-consistent $\hat{\sigma}^2$ estimator |
| Page (1954) | CUSUM change-point detection |

## Structure

```
research/
  core/
    mp_theory.py       MP density, CDF, BBP threshold, self-consistent σ²
    estimator.py       RollingSpectralEstimator, SpectralSnapshot
    changepoint.py     CUSUM detection (standard + two-regime on ρ(t))
    loader.py          yfinance download, parquet cache, NaN filtering
    universe.py        93-ticker S&P 100 universe (post-NaN-filter)
  analysis/
    validation.py      Crisis event study, ROC/AUROC, ρ decomposition
    plots.py           8 publication figures (matplotlib only)
  tests/
    test_mp_theory.py  45 tests
    test_estimator.py  26 tests
  notebooks/
    01_exploration.ipynb
    02_main_results.ipynb
```

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce figures

```bash
python reproduce_figures.py
```

Downloads ~22 years of daily returns on first run (~2 min), caches to
`research/data/returns_panel.parquet`. All 8 figures save to
`research/data/figures/`.

## Tests

```bash
python -m pytest research/tests/ -v
```

71 passed, 0 failed.

## Empirical results (SP100, 2003–2024)

**ρ decomposition.** The detachment ratio $\rho(t) = \lambda_1 / \lambda_+$
can be written in log-space as

$$\log \rho(t) = \underbrace{\log(\lambda_1 / \mu_{\lambda_1})}_{\text{Effect A}} + \underbrace{(-\log(\lambda_+ / \mu_{\lambda_+}))}_{\text{Effect B}}$$

where the calm-period baselines $\mu_{\lambda_1}, \mu_{\lambda_+}$ are
estimated from 2004–2007. At every crisis peak, Effect B (noise floor
collapse, $\lambda_+$ falling) contributes 51–63% of the total log-$\rho$
elevation; Effect A ($\lambda_1$ rising) contributes 37–49%.

**ROC/AUROC.** Against a 21-day-expanded crisis calendar:

| Statistic | AUROC | 95% CI (bootstrap) |
|-----------|-------|---------------------|
| $\rho(t)$ | 0.669 | [0.589, 0.745] |
| Spike count $k(t)$ | 0.566 | [0.464, 0.662] |
| Full KS | 0.566 | [0.466, 0.661] |
| Bulk KS | 0.486 | [0.384, 0.586] |

Bulk KS falls below chance (AUROC < 0.5) because the residual bulk becomes
*more* MP-like during crises as $\hat{\sigma}^2$ collapses and spikes are
stripped out. Full KS ≈ $k/d$ and correctly elevates during crises.

**GFC detection.** CUSUM calibrated on 2004–2007 calm detects the GFC with
a 56 calendar day lag and 2 false alarms. The detector alarms on
2007-10-04, five days before the labeled peak of 2007-10-09.

**Post-GFC structural break.** Bulk KS drifts monotonically upward 2004→2024.
A CUSUM applied to this series crossed threshold in 2007 and never reset,
indicating a permanent structural break rather than an episodic regime shift.
The 2013–2019 period is insufficiently stationary in $\rho$-space for
standard ARL-calibrated CUSUM.

## Methodological notes

**BBP threshold vs. $\lambda_+$.**
The BBP detection threshold and bulk upper edge are distinct:

$$\theta_c = \sigma^2(1 + \sqrt{\gamma}), \qquad \lambda_+ = \sigma^2(1 + \sqrt{\gamma})^2$$

so $\lambda_+ / \theta_c = 1 + \sqrt{\gamma} > 1$ always. A population spike
$\theta$ is detectable iff $\theta > \theta_c$. Below $\theta_c$ the sample
spike merges with the bulk; above $\theta_c$ the sample spike lands above
$\lambda_+$, not at $\theta_c$.

The BBP theorem is proven for complex Gaussian matrices. Application to real
financial returns (non-Gaussian, autocorrelated, heavy-tailed) is a
heuristic approximation.

**Standardization.** Within-window standardization uses $\text{ddof}=0$ to
ensure $\text{Tr}(S) = d$ exactly, consistent with the MP normalization.
This analyzes correlation structure, not covariance structure; heterogeneous
volatility across assets is removed.

**Survivorship bias.** The ticker universe contains long-tenured constituents
only. Seven tickers were dropped for >20% NaN (spinoffs post-2003: KHC, PSX,
MPC, GM, CHTR, DOW, LYB), leaving $d = 93$ assets and $\gamma = 93/252 \approx 0.37$.

**CUSUM stationarity.** Standard CUSUM assumes a stationary in-control
process. Financial returns violate this through volatility clustering,
regime-dependent baselines, and serial correlation from overlapping windows.
Alarm times are indicative, not statistically calibrated.
